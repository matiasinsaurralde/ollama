package renderers

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// Validates the GO-SIDE half of boundary finding D1 (media-marker desync) end to
// end: a vision renderer + the llama_server.go media-rewrite loop produce a
// /completion request whose #image-markers != #multimodal payloads. The C++/mtmd
// crash consequence is NOT exercised here (needs a real llama-server binary);
// this test only proves the desynced request is what the Go daemon builds.

// mirrors llm/llama_server.go:1565-1577 exactly (marker value is irrelevant;
// upstream uses a crypto-random per-process marker).
const fakeMarker = "<__ollama_media_TESTMARKER__>"

// pocMedia mirrors llm.MediaData{ID,Data} (avoids an import cycle). ID is the
// per-message index, matching llm.NewMediaData(i, data) at llm/server.go:192.
type pocMedia struct {
	ID   int
	Data []byte
}

func rewriteLikeLlamaServer(prompt string, media []pocMedia) (markerCount int, payloadCount int) {
	for _, m := range media {
		marker := fmt.Sprintf("[img-%d]", m.ID)
		prompt = strings.Replace(prompt, marker, fakeMarker, 1)
	}
	return strings.Count(prompt, fakeMarker), len(media)
}

func TestPoC_D1_MarkerDesync(t *testing.T) {
	// One user message: content contains the literal "[img-" (attacker text)
	// plus one attached image.
	msgs := []api.Message{{
		Role:    "user",
		Content: "please read [img-", // contains "[img-" but NOT the binding tag [img-0]
		Images:  []api.ImageData{[]byte{0x1, 0x2, 0x3}},
	}}
	media := []pocMedia{{ID: 0, Data: msgs[0].Images[0]}} // chatPrompt->NewMediaData(0,data)

	r := &GlmOcrRenderer{useImgTags: true}
	prompt, err := r.Render(msgs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	markerCount, payloadCount := rewriteLikeLlamaServer(prompt, media)
	t.Logf("rendered prompt = %q", prompt)
	t.Logf("markers substituted into prompt = %d ; multimodal payloads = %d", markerCount, payloadCount)

	if strings.Contains(prompt, "[img-0]") {
		t.Fatalf("expected NO binding tag [img-0] to be inserted (desync source), but found one")
	}
	if markerCount == payloadCount {
		t.Fatalf("expected desync (markers != payloads); got equal counts %d — bug not reproduced", markerCount)
	}
	t.Logf("D1 CONFIRMED (Go side): request to llama-server carries %d payloads but %d markers "+
		"— this is the mismatch mtmd receives", payloadCount, markerCount)
}

// Control: the SAME message without "[img-" in the text renders correctly —
// one binding tag inserted, counts match. Proves the desync is caused by the
// attacker's "[img-" text, not by having an image.
func TestPoC_D1_Control_NoDesyncNormally(t *testing.T) {
	msgs := []api.Message{{
		Role:    "user",
		Content: "please read this",
		Images:  []api.ImageData{[]byte{0x1, 0x2, 0x3}},
	}}
	media := []pocMedia{{ID: 0, Data: msgs[0].Images[0]}}

	r := &GlmOcrRenderer{useImgTags: true}
	prompt, err := r.Render(msgs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	markerCount, payloadCount := rewriteLikeLlamaServer(prompt, media)
	t.Logf("rendered prompt = %q ; markers=%d payloads=%d", prompt, markerCount, payloadCount)
	if markerCount != payloadCount {
		t.Fatalf("control failed: expected matched counts, got markers=%d payloads=%d", markerCount, payloadCount)
	}
}
