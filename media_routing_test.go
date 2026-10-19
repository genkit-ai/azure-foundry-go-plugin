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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func decodeJSONBody(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Errorf("decode request body: %v", err)
	}
	return body
}

func assertBodyFields(t *testing.T, body map[string]any, want map[string]any) {
	t.Helper()
	for key, value := range want {
		if body[key] != value {
			t.Errorf("request %s = %#v, want %#v", key, body[key], value)
		}
	}
}

func TestGenerateImagesSendsDefaultsAndConfig(t *testing.T) {
	tests := []struct {
		name   string
		config any
		want   map[string]any
	}{
		{
			name: "defaults",
			want: map[string]any{
				"model":           "dall-e-3",
				"prompt":          "draw a cat",
				"n":               float64(1),
				"size":            "1024x1024",
				"quality":         "standard",
				"style":           "vivid",
				"response_format": "url",
			},
		},
		{
			name: "map config",
			config: map[string]any{
				"n":               2,
				"size":            "1792x1024",
				"quality":         "hd",
				"style":           "natural",
				"response_format": "b64_json",
			},
			want: map[string]any{
				"n":               float64(2),
				"size":            "1792x1024",
				"quality":         "hd",
				"style":           "natural",
				"response_format": "b64_json",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assertBodyFields(t, decodeJSONBody(t, r), tt.want)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"created":1,"data":[{"url":"https://example.com/a.png"},{"url":"https://example.com/b.png"}]}`)
			}))
			defer server.Close()

			resp, err := testPluginForServer(server).generateImages(
				context.Background(),
				"dall-e-3",
				&ai.ModelRequest{
					Messages: []*ai.Message{ai.NewUserTextMessage("draw a cat")},
					Config:   tt.config,
				},
			)
			if err != nil {
				t.Fatalf("generateImages() error = %v", err)
			}
			if len(resp.Message.Content) != 2 {
				t.Fatalf("content parts = %d, want one media part per image", len(resp.Message.Content))
			}
		})
	}
}

func TestGenerateSpeechSendsDefaultsAndConfig(t *testing.T) {
	tests := []struct {
		name   string
		config any
		want   map[string]any
	}{
		{
			name: "defaults",
			want: map[string]any{
				"model":           "tts-1",
				"input":           "hello there",
				"voice":           "alloy",
				"response_format": "mp3",
				"speed":           float64(1),
			},
		},
		{
			name: "map config",
			config: map[string]any{
				"voice":           "nova",
				"response_format": "wav",
				"speed":           1.5,
			},
			want: map[string]any{
				"voice":           "nova",
				"response_format": "wav",
				"speed":           1.5,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/audio/speech" {
					t.Errorf("path = %q, want %q", r.URL.Path, "/audio/speech")
				}
				assertBodyFields(t, decodeJSONBody(t, r), tt.want)
				_, _ = w.Write([]byte("audio"))
			}))
			defer server.Close()

			_, err := testPluginForServer(server).generateSpeech(
				context.Background(),
				"tts-1",
				&ai.ModelRequest{
					Messages: []*ai.Message{ai.NewUserTextMessage("hello there")},
					Config:   tt.config,
				},
			)
			if err != nil {
				t.Fatalf("generateSpeech() error = %v", err)
			}
		})
	}
}

func TestGenerateImagesAndSpeechRejectEmptyPrompts(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("request sent for an empty prompt")
	}))
	defer server.Close()
	plugin := testPluginForServer(server)
	input := &ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("  ")}}

	if _, err := plugin.generateImages(context.Background(), "dall-e-3", input); err == nil {
		t.Error("generateImages() error = nil, want an empty prompt error")
	}
	if _, err := plugin.generateSpeech(context.Background(), "tts-1", input); err == nil {
		t.Error("generateSpeech() error = nil, want an empty input error")
	}
}

type transcriptionUpload struct {
	model          string
	filename       string
	audio          string
	language       string
	prompt         string
	responseFormat string
	temperature    string
}

func newTranscriptionServer(t *testing.T, response string, got *transcriptionUpload) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/transcriptions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/audio/transcriptions")
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("ParseMultipartForm() error = %v", err)
			return
		}
		file, header, err := r.FormFile("file")
		if err != nil {
			t.Errorf("FormFile() error = %v", err)
			return
		}
		defer func() { _ = file.Close() }()
		audio, err := io.ReadAll(file)
		if err != nil {
			t.Errorf("read uploaded audio: %v", err)
			return
		}
		*got = transcriptionUpload{
			model:          r.FormValue("model"),
			filename:       header.Filename,
			audio:          string(audio),
			language:       r.FormValue("language"),
			prompt:         r.FormValue("prompt"),
			responseFormat: r.FormValue("response_format"),
			temperature:    r.FormValue("temperature"),
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, response)
	}))
	t.Cleanup(server.Close)
	return server
}

func TestTranscribeAudioFromRequestUploadsAudioAndConfig(t *testing.T) {
	var got transcriptionUpload
	server := newTranscriptionServer(t, `{"text":"hello world"}`, &got)

	resp, err := testPluginForServer(server).transcribeAudioFromRequest(
		context.Background(),
		"whisper-1",
		&ai.ModelRequest{
			Messages: []*ai.Message{ai.NewUserMessage(
				// base64("audio-bytes")
				ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8tYnl0ZXM="),
			)},
			Config: map[string]any{
				"language":        "es",
				"prompt":          "Spanish podcast",
				"response_format": "verbose_json",
				"temperature":     0.2,
			},
		},
	)
	if err != nil {
		t.Fatalf("transcribeAudioFromRequest() error = %v", err)
	}
	if resp.Text() != "hello world" {
		t.Errorf("Text() = %q, want %q", resp.Text(), "hello world")
	}

	want := transcriptionUpload{
		model:          "whisper-1",
		filename:       "audio.wav",
		audio:          "audio-bytes",
		language:       "es",
		prompt:         "Spanish podcast",
		responseFormat: "verbose_json",
		temperature:    "0.2",
	}
	if got != want {
		t.Errorf("upload = %+v, want %+v", got, want)
	}
}

func TestTranscribeAudioFromRequestFilenameFromMediaType(t *testing.T) {
	tests := map[string]string{
		"audio/mp3":  "audio.mp3",
		"audio/mpeg": "audio.mp3",
		"audio/wav":  "audio.wav",
		"audio/opus": "audio.opus",
	}
	for mediaType, wantFilename := range tests {
		t.Run(mediaType, func(t *testing.T) {
			var got transcriptionUpload
			server := newTranscriptionServer(t, `{"text":"ok"}`, &got)

			_, err := testPluginForServer(server).transcribeAudioFromRequest(
				context.Background(),
				"whisper-1",
				&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserMessage(
					ai.NewMediaPart(mediaType, "data:"+mediaType+";base64,YQ=="),
				)}},
			)
			if err != nil {
				t.Fatalf("transcribeAudioFromRequest() error = %v", err)
			}
			if got.filename != wantFilename {
				t.Errorf("filename = %q, want %q", got.filename, wantFilename)
			}
			if got.responseFormat != "json" {
				t.Errorf("response_format = %q, want the json default", got.responseFormat)
			}
		})
	}
}

func TestTranscribeAudioFromRequestErrors(t *testing.T) {
	tests := []struct {
		name    string
		message *ai.Message
		wantErr string
	}{
		{
			name:    "no media",
			message: ai.NewUserTextMessage("transcribe this"),
			wantErr: "no audio data found",
		},
		{
			name:    "media without base64 data",
			message: ai.NewUserMessage(ai.NewMediaPart("audio/mp3", "https://example.com/a.mp3")),
			wantErr: "no audio data found",
		},
		{
			name:    "malformed base64",
			message: ai.NewUserMessage(ai.NewMediaPart("audio/mp3", "data:audio/mp3;base64,not-base64!")),
			wantErr: "failed to decode audio",
		},
	}

	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("request sent for invalid audio input")
	}))
	defer server.Close()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := testPluginForServer(server).transcribeAudioFromRequest(
				context.Background(),
				"whisper-1",
				&ai.ModelRequest{Messages: []*ai.Message{tt.message}},
			)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

func TestMediaHelpersRequireInit(t *testing.T) {
	plugin := &AzureAIFoundry{}
	ctx := context.Background()

	if _, err := plugin.generateImagesInternal(ctx, "dall-e-3", &ImageGenerationRequest{Prompt: "x"}); err == nil {
		t.Error("generateImagesInternal() error = nil, want not initialized")
	}
	if _, err := plugin.generateSpeechInternal(ctx, "tts-1", &TTSRequest{Input: "x"}); err == nil {
		t.Error("generateSpeechInternal() error = nil, want not initialized")
	}
	if _, err := plugin.transcribeAudioInternal(ctx, "whisper-1", &STTRequest{Audio: []byte("x")}); err == nil {
		t.Error("transcribeAudioInternal() error = nil, want not initialized")
	}
}

func TestGenerateTextRoutesByModelType(t *testing.T) {
	tests := []struct {
		name     string
		model    ModelDefinition
		message  *ai.Message
		wantPath string
	}{
		{
			name:     "explicit chat overrides a misleading name",
			model:    ModelDefinition{Name: "transcribe-router", Type: ModelTypeChat},
			message:  ai.NewUserTextMessage("hi"),
			wantPath: "/chat/completions",
		},
		{
			name:     "explicit image",
			model:    ModelDefinition{Name: "my-images", Type: ModelTypeImage},
			message:  ai.NewUserTextMessage("draw"),
			wantPath: "/images/generations",
		},
		{
			name:     "explicit text-to-speech",
			model:    ModelDefinition{Name: "voice", Type: ModelTypeTextToSpeech},
			message:  ai.NewUserTextMessage("say"),
			wantPath: "/audio/speech",
		},
		{
			name:     "explicit speech-to-text",
			model:    ModelDefinition{Name: "ears", Type: ModelTypeSpeechToText},
			message:  ai.NewUserMessage(ai.NewMediaPart("audio/mp3", "data:audio/mp3;base64,YQ==")),
			wantPath: "/audio/transcriptions",
		},
		{
			name:     "name-based image fallback",
			model:    ModelDefinition{Name: "dall-e-3"},
			message:  ai.NewUserTextMessage("draw"),
			wantPath: "/images/generations",
		},
		{
			name:     "name-based speech fallback",
			model:    ModelDefinition{Name: "tts-1-hd"},
			message:  ai.NewUserTextMessage("say"),
			wantPath: "/audio/speech",
		},
		{
			name:     "name-based transcription fallback",
			model:    ModelDefinition{Name: "gpt-4o-transcribe"},
			message:  ai.NewUserMessage(ai.NewMediaPart("audio/mp3", "data:audio/mp3;base64,YQ==")),
			wantPath: "/audio/transcriptions",
		},
		{
			name:     "name-based chat fallback",
			model:    ModelDefinition{Name: "gpt-4o"},
			message:  ai.NewUserTextMessage("hi"),
			wantPath: "/chat/completions",
		},
	}

	responses := map[string]string{
		"/chat/completions":     chatCompletionResponse,
		"/images/generations":   `{"created":1,"data":[{"url":"https://example.com/a.png"}]}`,
		"/audio/speech":         "audio",
		"/audio/transcriptions": `{"text":"ok"}`,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotPath string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				response, ok := responses[r.URL.Path]
				if !ok {
					t.Errorf("unexpected request path %q", r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
					return
				}
				if r.URL.Path != "/audio/speech" {
					w.Header().Set("Content-Type", "application/json")
				}
				_, _ = io.WriteString(w, response)
			}))
			defer server.Close()

			_, err := testPluginForServer(server).generateText(
				context.Background(),
				tt.model,
				&ai.ModelRequest{Messages: []*ai.Message{tt.message}},
				nil,
			)
			if err != nil {
				t.Fatalf("generateText() error = %v", err)
			}
			if gotPath != tt.wantPath {
				t.Errorf("path = %q, want %q", gotPath, tt.wantPath)
			}
		})
	}
}

func TestGenerateTextRejectsUnsupportedModelType(t *testing.T) {
	_, err := (&AzureAIFoundry{}).generateText(
		context.Background(),
		ModelDefinition{Name: "gpt-4o", Type: "video"},
		&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("hi")}},
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "unsupported model type") {
		t.Fatalf("error = %v, want an unsupported model type error", err)
	}
}
