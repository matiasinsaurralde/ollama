package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"time"

	"github.com/ollama/ollama/types/model"
)

type Manifest struct {
	SchemaVersion          int     `json:"schemaVersion"`
	MediaType              string  `json:"mediaType"`
	Config                 Layer   `json:"config"`
	Layers                 []Layer `json:"layers"`
	IsAppleFoundationModel bool

	filepath string
	fi       os.FileInfo
	digest   string
}

func (m *Manifest) Size() (size int64) {
	for _, layer := range append(m.Layers, m.Config) {
		size += layer.Size
	}

	return
}

func (m *Manifest) Remove() error {
	if err := os.Remove(m.filepath); err != nil {
		return err
	}

	manifests, err := GetManifestPath()
	if err != nil {
		return err
	}

	return PruneDirectory(manifests)
}

func (m *Manifest) RemoveLayers() error {
	for _, layer := range append(m.Layers, m.Config) {
		if layer.Digest != "" {
			if err := layer.Remove(); errors.Is(err, os.ErrNotExist) {
				slog.Debug("layer does not exist", "digest", layer.Digest)
			} else if err != nil {
				return err
			}
		}
	}

	return nil
}

func ParseNamedManifest(n model.Name) (*Manifest, error) {
	if !n.IsFullyQualified() {
		return nil, model.Unqualified(n)
	}

	manifests, err := GetManifestPath()
	if err != nil {
		return nil, err
	}

	p := filepath.Join(manifests, n.Filepath())

	var m Manifest
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	sha256sum := sha256.New()
	if err := json.NewDecoder(io.TeeReader(f, sha256sum)).Decode(&m); err != nil {
		return nil, err
	}

	m.filepath = p
	m.fi = fi
	m.digest = hex.EncodeToString(sha256sum.Sum(nil))

	return &m, nil
}

func WriteManifest(name model.Name, config Layer, layers []Layer) error {
	manifests, err := GetManifestPath()
	if err != nil {
		return err
	}

	p := filepath.Join(manifests, name.Filepath())
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}

	f, err := os.Create(p)
	if err != nil {
		return err
	}
	defer f.Close()

	m := Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config:        config,
		Layers:        layers,
	}

	return json.NewEncoder(f).Encode(m)
}

func Manifests(continueOnError bool) (map[model.Name]*Manifest, error) {
	manifests, err := GetManifestPath()
	if err != nil {
		return nil, err
	}

	// TODO(mxyng): use something less brittle
	matches, err := filepath.Glob(filepath.Join(manifests, "*", "*", "*", "*"))
	if err != nil {
		return nil, err
	}

	ms := make(map[model.Name]*Manifest)
	for _, match := range matches {
		fi, err := os.Stat(match)
		if err != nil {
			return nil, err
		}

		if !fi.IsDir() {
			rel, err := filepath.Rel(manifests, match)
			if err != nil {
				if !continueOnError {
					return nil, fmt.Errorf("%s %w", match, err)
				}
				slog.Warn("bad filepath", "path", match, "error", err)
				continue
			}

			n := model.ParseNameFromFilepath(rel)
			if !n.IsValid() {
				if !continueOnError {
					return nil, fmt.Errorf("%s %w", rel, err)
				}
				slog.Warn("bad manifest name", "path", rel)
				continue
			}

			m, err := ParseNamedManifest(n)
			if err != nil {
				if !continueOnError {
					return nil, fmt.Errorf("%s %w", n, err)
				}
				slog.Warn("bad manifest", "name", n, "error", err)
				continue
			}

			ms[n] = m
		}
	}

	// Add models
	appleFoundationModelEnabled := os.Getenv("OLLAMA_ENABLE_APPLE_FOUNDATION_MODEL") == "1"
	slog.Info("check for Apple Foundation Models", slog.Bool("enabled", appleFoundationModelEnabled))
	if appleFoundationModelEnabled {
		manifests, err := createAppleManifests()
		if err != nil {
			if !continueOnError {
				return nil, fmt.Errorf("failed to get models: %w", err)
			}
			slog.Warn("failed to get models", "error", err)
		} else {
			for name, manifest := range manifests {
				ms[name] = manifest
			}
		}
	}

	return ms, nil
}

// createAppleManifests creates manifests for Apple Foundation Models:
func createAppleManifests() (map[model.Name]*Manifest, error) {
	manifests := make(map[model.Name]*Manifest)
	name := model.ParseName("apple")
	if !name.IsValid() {
		return nil, fmt.Errorf("invalid model name")
	}

	// Create a minimal config layer
	config := ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
		RootFS: RootFS{
			Type: "layers",
		},
		ModelFormat:   "apple",
		ModelFamily:   "apple",
		ModelType:     "apple",
		FileType:      "apple",
		ModelFamilies: []string{"apple"},
	}

	fakeDigest := GenerateFakeDigest("apple")

	configData, err := json.Marshal(config)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal config: %w", err)
	}

	// Create config layer
	configLayer := Layer{
		Digest:    fakeDigest,
		MediaType: "application/vnd.ollama.image.config",
		Size:      int64(len(configData)),
	}

	// Create a placeholder model layer
	modelLayer := Layer{
		Digest:    fakeDigest,
		MediaType: "application/vnd.ollama.image.applefoundationmodel",
		Size:      1024 * 1024, // 1MB placeholder
	}

	// Create template layer
	templateData := []byte(`{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>`)

	templateLayer := Layer{
		Digest:    fakeDigest,
		MediaType: "application/vnd.ollama.image.template",
		Size:      int64(len(templateData)),
	}

	manifest := &Manifest{
		SchemaVersion:          2,
		MediaType:              "application/vnd.docker.distribution.manifest.v2+json",
		Config:                 configLayer,
		Layers:                 []Layer{modelLayer, templateLayer},
		IsAppleFoundationModel: true,
	}

	// Set file info using reflection
	manifestValue := reflect.ValueOf(manifest).Elem()
	filepathField := manifestValue.FieldByName("filepath")
	fiField := manifestValue.FieldByName("fi")
	// digestField := manifestValue.FieldByName("digest")

	if filepathField.IsValid() && filepathField.CanSet() {
		filepathField.SetString("")
	}
	if fiField.IsValid() && fiField.CanSet() {
		fiField.Set(reflect.ValueOf(&fakeFileInfo{
			name:    name.Filepath(),
			size:    manifest.Size(),
			mode:    0644,
			modTime: time.Now(),
		}))
	}

	manifests[name] = manifest
	slog.Info("registered Apple Foundation Model")

	return manifests, nil
}

func GenerateFakeDigest(name string) string {
	hash := sha256.Sum256([]byte("apple:" + name))
	return hex.EncodeToString(hash[:])
}

// fakeFileInfo implements os.FileInfo for the Apple Foundation Model
type fakeFileInfo struct {
	name    string
	size    int64
	mode    os.FileMode
	modTime time.Time
}

func (f *fakeFileInfo) Name() string       { return f.name }
func (f *fakeFileInfo) Size() int64        { return f.size }
func (f *fakeFileInfo) Mode() os.FileMode  { return f.mode }
func (f *fakeFileInfo) ModTime() time.Time { return f.modTime }
func (f *fakeFileInfo) IsDir() bool        { return false }
func (f *fakeFileInfo) Sys() interface{}   { return nil }
