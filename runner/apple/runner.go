package proprietary

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/ollama/ollama/llm"
)

// Server implements the Apple Foundation Model runner
type Server struct {
	modelName string
	status    llm.ServerStatus
	progress  float32
	ready     sync.WaitGroup
	mu        sync.Mutex
	log       *slog.Logger
}

// NewServer creates a new Apple Foundation Model server
func NewServer(modelName string) *Server {
	s := &Server{
		modelName: modelName,
		status:    llm.ServerStatusLoadingModel,
		log:       slog.With("model", modelName),
	}
	s.ready.Add(1)

	// Simulate loading
	go func() {
		defer s.ready.Done()
		time.Sleep(100 * time.Millisecond) // Simulate load time
		s.mu.Lock()
		s.status = llm.ServerStatusReady
		s.progress = 1.0
		s.mu.Unlock()
		slog.Info("loaded")
	}()

	return s
}

// WaitUntilRunning waits for the server to be ready
func (s *Server) WaitUntilRunning(ctx context.Context) error {
	done := make(chan struct{})
	go func() {
		s.ready.Wait()
		close(done)
	}()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-done:
		return nil
	}
}

// Ping checks if the server is responsive
func (s *Server) Ping(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.status != llm.ServerStatusReady {
		return fmt.Errorf("server not ready, status: %s", s.status)
	}
	return nil
}

// Close shuts down the server
func (s *Server) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.status = llm.ServerStatusError
	s.log.Info("closed")
	return nil
}

// Pid returns the process ID (not applicable for proprietary models)
func (s *Server) Pid() int {
	return -1 // No actual process for proprietary models
}

// EstimatedVRAM returns estimated VRAM usage
func (s *Server) EstimatedVRAM() uint64 {
	return 0 // Proprietary models don't use VRAM
}

// EstimatedTotal returns estimated total memory usage
func (s *Server) EstimatedTotal() uint64 {
	return 1024 * 1024 // 1MB placeholder
}

// EstimatedVRAMByGPU returns estimated VRAM usage per GPU
func (s *Server) EstimatedVRAMByGPU(gpuID string) uint64 {
	return 0 // Proprietary models don't use GPU VRAM
}

// Completion handles text generation requests
func (s *Server) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	s.ready.Wait()
	response := fmt.Sprintf("response from the model '%s'. Your prompt was: %s", s.modelName, req.Prompt)
	// Call the callback function with the response
	fn(llm.CompletionResponse{
		Content:    response,
		DoneReason: llm.DoneReasonStop,
		Done:       true,
	})

	return nil
}

// Embedding handles embedding requests
func (s *Server) Embedding(ctx context.Context, input string) ([]float32, error) {
	s.ready.Wait()

	// Return a placeholder embedding
	embedding := make([]float32, 384) // 384-dimensional placeholder
	for i := range embedding {
		embedding[i] = 0.1 // Placeholder values
	}

	return embedding, nil
}

// Tokenize handles tokenization requests
func (s *Server) Tokenize(ctx context.Context, content string) ([]int, error) {
	s.ready.Wait()

	// For now, return a simple character-based tokenization
	tokens := make([]int, len(content))
	for i, char := range content {
		tokens[i] = int(char)
	}

	return tokens, nil
}

// Detokenize handles detokenization requests
func (s *Server) Detokenize(ctx context.Context, tokens []int) (string, error) {
	s.ready.Wait()

	// TODO: Replace with your actual detokenization logic
	// For now, return a simple character-based detokenization
	content := make([]rune, len(tokens))
	for i, token := range tokens {
		content[i] = rune(token)
	}

	return string(content), nil
}

// Health returns server health status
func (s *Server) Health(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()

	status := &llm.ServerStatusResponse{
		Status:   s.status,
		Progress: s.progress,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}
