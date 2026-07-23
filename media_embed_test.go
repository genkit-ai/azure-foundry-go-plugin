// Copyright 2026 Xavier Portilla Edo
// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package azureaifoundry

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func testPluginForServer(server *httptest.Server) *AzureAIFoundry {
	return &AzureAIFoundry{
		client: openai.NewClient(
			option.WithBaseURL(server.URL+"/"),
			option.WithAPIKey("test"),
		),
		initted: true,
	}
}

func TestGenerateImagesReturnsMediaParts(t *testing.T) {
	tests := []struct {
		name       string
		response   string
		wantMedia  string
		wantFormat string
	}{
		{
			name:       "URL",
			response:   `{"created":1,"data":[{"url":"https://example.com/generated.png"}]}`,
			wantMedia:  "https://example.com/generated.png",
			wantFormat: "url",
		},
		{
			name:       "base64",
			response:   `{"created":1,"data":[{"b64_json":"aW1hZ2U="}]}`,
			wantMedia:  "data:image/png;base64,aW1hZ2U=",
			wantFormat: "b64_json",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/images/generations" {
					t.Errorf("path = %q, want %q", r.URL.Path, "/images/generations")
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tt.response)
			}))
			defer server.Close()

			resp, err := testPluginForServer(server).generateImages(
				context.Background(),
				"dall-e-3",
				&ai.ModelRequest{
					Messages: []*ai.Message{ai.NewUserTextMessage("draw a landscape")},
					Config: map[string]any{
						"response_format": tt.wantFormat,
					},
				},
			)
			if err != nil {
				t.Fatalf("generateImages() error = %v", err)
			}
			if resp.Media() != tt.wantMedia {
				t.Fatalf("Media() = %q, want %q", resp.Media(), tt.wantMedia)
			}
			if len(resp.Message.Content) != 1 || !resp.Message.Content[0].IsMedia() {
				t.Fatalf("content = %#v, want one media part", resp.Message.Content)
			}
			if !resp.Message.Content[0].IsImage() {
				t.Fatalf("content = %#v, want an image part", resp.Message.Content[0])
			}
			if got := resp.Message.Content[0].ContentType; got != "image/png" {
				t.Fatalf("ContentType = %q, want %q", got, "image/png")
			}
		})
	}
}

func TestGenerateSpeechReturnsMediaPartForEachFormat(t *testing.T) {
	mimeTypes := map[string]string{
		"mp3":  "audio/mpeg",
		"opus": "audio/ogg",
		"aac":  "audio/aac",
		"flac": "audio/flac",
		"wav":  "audio/wav",
		"pcm":  "audio/pcm",
	}

	for format, mimeType := range mimeTypes {
		t.Run(format, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", mimeType)
				_, _ = w.Write([]byte("audio"))
			}))
			defer server.Close()

			resp, err := testPluginForServer(server).generateSpeech(
				context.Background(),
				"tts-1",
				&ai.ModelRequest{
					Messages: []*ai.Message{ai.NewUserTextMessage("hello")},
					Config: map[string]any{
						"response_format": format,
					},
				},
			)
			if err != nil {
				t.Fatalf("generateSpeech() error = %v", err)
			}
			wantMedia := "data:" + mimeType + ";base64,YXVkaW8="
			if resp.Media() != wantMedia {
				t.Fatalf("Media() = %q, want %q", resp.Media(), wantMedia)
			}
			if !resp.Message.Content[0].IsAudio() {
				t.Fatalf("content = %#v, want an audio part", resp.Message.Content[0])
			}
			if got := resp.Message.Content[0].ContentType; got != mimeType {
				t.Fatalf("ContentType = %q, want %q", got, mimeType)
			}
		})
	}
}

type embeddingRequestBody struct {
	Input          []string `json:"input"`
	Model          string   `json:"model"`
	Dimensions     *int64   `json:"dimensions,omitempty"`
	EncodingFormat string   `json:"encoding_format"`
}

func TestEmbedBatchesInputsAndPreservesOrder(t *testing.T) {
	var got embeddingRequestBody
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"model":"text-embedding-3-small",
			"data":[
				{"object":"embedding","index":2,"embedding":[3]},
				{"object":"embedding","index":0,"embedding":[1]},
				{"object":"embedding","index":1,"embedding":[2]}
			],
			"usage":{"prompt_tokens":3,"total_tokens":3}
		}`)
	}))
	defer server.Close()

	dimensions := int64(256)
	resp, err := testPluginForServer(server).embed(
		context.Background(),
		"text-embedding-3-small",
		&ai.EmbedRequest{
			Input: []*ai.Document{
				ai.DocumentFromText("first", nil),
				{Content: []*ai.Part{ai.NewTextPart(""), ai.NewMediaPart("image/png", "ignored")}},
				{Content: []*ai.Part{ai.NewTextPart("sec"), ai.NewTextPart("ond")}},
				ai.DocumentFromText("third", nil),
			},
			Options: &EmbeddingConfig{
				Dimensions:     &dimensions,
				EncodingFormat: "float",
			},
		},
	)
	if err != nil {
		t.Fatalf("embed() error = %v", err)
	}

	if strings.Join(got.Input, ",") != "first,second,third" {
		t.Fatalf("input = %#v, want first, second, third", got.Input)
	}
	if got.Model != "text-embedding-3-small" {
		t.Fatalf("model = %q, want %q", got.Model, "text-embedding-3-small")
	}
	if got.Dimensions == nil || *got.Dimensions != dimensions {
		t.Fatalf("dimensions = %v, want %d", got.Dimensions, dimensions)
	}
	if got.EncodingFormat != "float" {
		t.Fatalf("encoding_format = %q, want %q", got.EncodingFormat, "float")
	}
	if len(resp.Embeddings) != 3 {
		t.Fatalf("len(Embeddings) = %d, want 3", len(resp.Embeddings))
	}
	for i, embedding := range resp.Embeddings {
		want := float32(i + 1)
		if len(embedding.Embedding) != 1 || embedding.Embedding[0] != want {
			t.Fatalf("Embeddings[%d] = %#v, want [%v]", i, embedding.Embedding, want)
		}
	}
}

func TestEmbedAcceptsMapConfig(t *testing.T) {
	var got embeddingRequestBody
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"model":"text-embedding-3-small",
			"data":[{"object":"embedding","index":0,"embedding":[1]}],
			"usage":{"prompt_tokens":1,"total_tokens":1}
		}`)
	}))
	defer server.Close()

	_, err := testPluginForServer(server).embed(
		context.Background(),
		"text-embedding-3-small",
		&ai.EmbedRequest{
			Input: []*ai.Document{ai.DocumentFromText("hello", nil)},
			Options: map[string]any{
				"dimensions":      float64(128),
				"encoding_format": "float",
			},
		},
	)
	if err != nil {
		t.Fatalf("embed() error = %v", err)
	}
	if got.Dimensions == nil || *got.Dimensions != 128 {
		t.Fatalf("dimensions = %v, want 128", got.Dimensions)
	}
	if got.EncodingFormat != "float" {
		t.Fatalf("encoding_format = %q, want %q", got.EncodingFormat, "float")
	}
}

func TestEmbedChunksAtEndpointLimit(t *testing.T) {
	var mu sync.Mutex
	var batchSizes []int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body embeddingRequestBody
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		mu.Lock()
		batchSizes = append(batchSizes, len(body.Input))
		mu.Unlock()

		data := make([]map[string]any, len(body.Input))
		for i := range body.Input {
			data[i] = map[string]any{
				"object":    "embedding",
				"index":     i,
				"embedding": []float64{float64(i)},
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"model":  "text-embedding-3-small",
			"data":   data,
			"usage": map[string]int{
				"prompt_tokens": len(body.Input),
				"total_tokens":  len(body.Input),
			},
		})
	}))
	defer server.Close()

	input := make([]*ai.Document, maxEmbeddingBatchSize+1)
	for i := range input {
		input[i] = ai.DocumentFromText(fmt.Sprintf("document-%d", i), nil)
	}
	resp, err := testPluginForServer(server).embed(
		context.Background(),
		"text-embedding-3-small",
		&ai.EmbedRequest{Input: input},
	)
	if err != nil {
		t.Fatalf("embed() error = %v", err)
	}
	if fmt.Sprint(batchSizes) != "[2048 1]" {
		t.Fatalf("batch sizes = %v, want [2048 1]", batchSizes)
	}
	if len(resp.Embeddings) != len(input) {
		t.Fatalf("len(Embeddings) = %d, want %d", len(resp.Embeddings), len(input))
	}
	if got := resp.Embeddings[maxEmbeddingBatchSize].Embedding[0]; got != 0 {
		t.Fatalf("first value in second batch = %v, want 0", got)
	}
}

func TestEmbedAllEmptyMakesNoRequest(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		requests++
	}))
	defer server.Close()

	resp, err := testPluginForServer(server).embed(
		context.Background(),
		"text-embedding-3-small",
		&ai.EmbedRequest{
			Input: []*ai.Document{
				nil,
				{},
				ai.DocumentFromText("", nil),
			},
		},
	)
	if err != nil {
		t.Fatalf("embed() error = %v", err)
	}
	if requests != 0 {
		t.Fatalf("requests = %d, want 0", requests)
	}
	if len(resp.Embeddings) != 0 {
		t.Fatalf("len(Embeddings) = %d, want 0", len(resp.Embeddings))
	}
}

func TestEmbedRejectsInvalidConfigBeforeRequest(t *testing.T) {
	zero := int64(0)
	negative := int64(-1)
	tests := []struct {
		name    string
		options any
		want    string
	}{
		{
			name:    "zero dimensions",
			options: EmbeddingConfig{Dimensions: &zero},
			want:    "dimensions must be greater than zero",
		},
		{
			name:    "negative dimensions map",
			options: map[string]any{"dimensions": negative},
			want:    "dimensions must be greater than zero",
		},
		{
			name:    "base64 encoding",
			options: EmbeddingConfig{EncodingFormat: "base64"},
			want:    `unsupported embedding encoding_format "base64"`,
		},
		{
			name:    "unknown encoding",
			options: map[string]any{"encoding_format": "binary"},
			want:    `unsupported embedding encoding_format "binary"`,
		},
		{
			name:    "malformed config",
			options: map[string]any{"dimensions": "many"},
			want:    "invalid embedding config",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requests := 0
			server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				requests++
			}))
			defer server.Close()

			_, err := testPluginForServer(server).embed(
				context.Background(),
				"text-embedding-3-small",
				&ai.EmbedRequest{
					Input:   []*ai.Document{ai.DocumentFromText("hello", nil)},
					Options: tt.options,
				},
			)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("embed() error = %v, want containing %q", err, tt.want)
			}
			if requests != 0 {
				t.Fatalf("requests = %d, want 0", requests)
			}
		})
	}
}

func TestEmbedRejectsMalformedResponseIndexes(t *testing.T) {
	tests := []struct {
		name string
		data string
		want string
	}{
		{
			name: "missing",
			data: `[{"object":"embedding","index":0,"embedding":[1]}]`,
			want: "response is missing index 1",
		},
		{
			name: "duplicate",
			data: `[
				{"object":"embedding","index":0,"embedding":[1]},
				{"object":"embedding","index":0,"embedding":[2]}
			]`,
			want: "duplicate response index 0",
		},
		{
			name: "out of range",
			data: `[{"object":"embedding","index":2,"embedding":[1]}]`,
			want: "response index 2 is out of range",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprintf(
					w,
					`{"object":"list","model":"test","data":%s,"usage":{"prompt_tokens":2,"total_tokens":2}}`,
					tt.data,
				)
			}))
			defer server.Close()

			_, err := testPluginForServer(server).embed(
				context.Background(),
				"text-embedding-3-small",
				&ai.EmbedRequest{
					Input: []*ai.Document{
						ai.DocumentFromText("first", nil),
						ai.DocumentFromText("second", nil),
					},
				},
			)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("embed() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}
