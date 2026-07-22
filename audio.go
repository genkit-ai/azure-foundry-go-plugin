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
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go/v3"
)

// fileReader wraps a bytes.Reader to provide a filename for multipart uploads
type fileReader struct {
	*bytes.Reader
	name string
}

// Name returns the filename for multipart form uploads
func (f *fileReader) Name() string {
	return f.name
}

// TTSRequest represents a text-to-speech request
type TTSRequest struct {
	Input          string  // The text to synthesize
	Voice          string  // Voice: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
	ResponseFormat string  // Format: "mp3", "opus", "aac", "flac", "wav", "pcm"
	Speed          float64 // Speed (0.25 to 4.0)
}

// TTSResponse represents the text-to-speech response
type TTSResponse struct {
	Audio []byte // The audio data
}

// generateSpeechInternal converts text to speech using TTS models
func (a *AzureAIFoundry) generateSpeechInternal(ctx context.Context, modelName string, req *TTSRequest) (*TTSResponse, error) {
	a.mu.Lock()
	if !a.initted {
		a.mu.Unlock()
		return nil, fmt.Errorf("azureaifoundry: client not initialized")
	}
	client := a.client
	a.mu.Unlock()

	// Build TTS parameters
	params := openai.AudioSpeechNewParams{
		Model: openai.SpeechModel(modelName),
		Input: req.Input,
		Voice: openai.AudioSpeechNewParamsVoiceUnion{
			OfString: openai.String(req.Voice),
		},
	}

	if req.ResponseFormat != "" {
		params.ResponseFormat = openai.AudioSpeechNewParamsResponseFormat(req.ResponseFormat)
	}
	if req.Speed > 0 {
		params.Speed = openai.Float(req.Speed)
	}

	// Generate speech
	resp, err := client.Audio.Speech.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("speech generation failed: %w", err)
	}

	// Read all audio data from the response body
	audioData, err := io.ReadAll(resp.Body)
	if closeErr := resp.Body.Close(); closeErr != nil {
		return nil, fmt.Errorf("failed to close response body: %w", closeErr)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to read audio data: %w", err)
	}

	return &TTSResponse{
		Audio: audioData,
	}, nil
}

// STTRequest represents a speech-to-text request
type STTRequest struct {
	Audio          []byte  // The audio file content
	Filename       string  // Filename with extension (e.g., "audio.mp3", "audio.wav") - required for format detection
	Language       string  // Language code (e.g., "en", "es")
	Prompt         string  // Optional text to guide the model's style
	ResponseFormat string  // Format: "json", "text", "srt", "verbose_json", "vtt"
	Temperature    float64 // Temperature (0 to 1)
}

// STTResponse represents the speech-to-text response
type STTResponse struct {
	Text     string  // Transcribed text
	Language string  // Detected language
	Duration float64 // Duration in seconds
}

// transcribeAudioInternal transcribes audio to text using Whisper models
func (a *AzureAIFoundry) transcribeAudioInternal(ctx context.Context, modelName string, req *STTRequest) (*STTResponse, error) {
	a.mu.Lock()
	if !a.initted {
		a.mu.Unlock()
		return nil, fmt.Errorf("azureaifoundry: client not initialized")
	}
	client := a.client
	a.mu.Unlock()

	// Determine filename - use provided filename or default based on format
	filename := req.Filename
	if filename == "" {
		filename = "audio.mp3" // Default to mp3 if not specified
	}

	// Create a named reader for the file upload
	// The openai SDK expects an io.Reader, and the filename is inferred from the field name
	// We need to use a file-like reader that can provide metadata
	file := &fileReader{
		Reader: bytes.NewReader(req.Audio),
		name:   filename,
	}

	// Build transcription parameters
	params := openai.AudioTranscriptionNewParams{
		Model: openai.AudioModel(modelName),
		File:  file,
	}

	if req.Language != "" {
		params.Language = openai.String(req.Language)
	}
	if req.Prompt != "" {
		params.Prompt = openai.String(req.Prompt)
	}
	if req.ResponseFormat != "" {
		params.ResponseFormat = openai.AudioResponseFormat(req.ResponseFormat)
	}
	if req.Temperature > 0 {
		params.Temperature = openai.Float(req.Temperature)
	}

	// Transcribe audio
	resp, err := client.Audio.Transcriptions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("audio transcription failed: %w", err)
	}

	return &STTResponse{
		Text:     resp.Text,
		Language: resp.Language,
		Duration: resp.Duration,
	}, nil
}

// generateSpeech handles text-to-speech through Genkit's Generate interface
func (a *AzureAIFoundry) generateSpeech(ctx context.Context, modelName string, input *ai.ModelRequest) (*ai.ModelResponse, error) {
	// Extract text from messages
	var text string
	for _, msg := range input.Messages {
		for _, part := range msg.Content {
			if part.IsText() {
				text += part.Text
			}
		}
	}

	// Extract config if provided
	req := &TTSRequest{
		Input:          text,
		Voice:          "alloy",
		ResponseFormat: "mp3",
		Speed:          1.0,
	}

	// Apply config from input if available
	type speechConfig struct {
		Voice          string   `json:"voice,omitempty"`
		ResponseFormat string   `json:"response_format,omitempty"`
		Speed          *float64 `json:"speed,omitempty"`
	}
	if cfg, ok := decodeConfig[speechConfig](input.Config); ok {
		if cfg.Voice != "" {
			req.Voice = cfg.Voice
		}
		if cfg.ResponseFormat != "" {
			req.ResponseFormat = cfg.ResponseFormat
		}
		if cfg.Speed != nil {
			req.Speed = *cfg.Speed
		}
	}

	// Generate speech
	resp, err := a.generateSpeechInternal(ctx, modelName, req)
	if err != nil {
		return nil, err
	}

	// Return audio as base64-encoded text (following Genkit pattern)
	audioBase64 := base64.StdEncoding.EncodeToString(resp.Audio)

	return &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(audioBase64)},
		},
		FinishReason: ai.FinishReasonStop,
	}, nil
}

// transcribeAudioFromRequest handles speech-to-text through Genkit's Generate interface
func (a *AzureAIFoundry) transcribeAudioFromRequest(ctx context.Context, modelName string, input *ai.ModelRequest) (*ai.ModelResponse, error) {
	// Extract audio from media parts
	var audioData []byte
	var filename string

	for _, msg := range input.Messages {
		for _, part := range msg.Content {
			if part.IsMedia() {
				// Media part contains base64-encoded audio
				// Format: "data:audio/wav;base64,..."
				mediaText := part.Text
				if idx := strings.Index(mediaText, "base64,"); idx != -1 {
					b64Data := mediaText[idx+7:]
					var err error
					audioData, err = base64.StdEncoding.DecodeString(b64Data)
					if err != nil {
						return nil, fmt.Errorf("failed to decode audio: %w", err)
					}

					// Extract format from media type
					if strings.Contains(mediaText, "audio/mp3") || strings.Contains(mediaText, "audio/mpeg") {
						filename = "audio.mp3"
					} else if strings.Contains(mediaText, "audio/wav") {
						filename = "audio.wav"
					} else if strings.Contains(mediaText, "audio/opus") {
						filename = "audio.opus"
					} else {
						filename = "audio.mp3" // default
					}
				}
			}
		}
	}

	if len(audioData) == 0 {
		return nil, fmt.Errorf("no audio data found in request")
	}

	// Extract config if provided
	req := &STTRequest{
		Audio:          audioData,
		Filename:       filename,
		ResponseFormat: "json",
	}

	// Apply config from input if available
	type transcriptionConfig struct {
		Language       string   `json:"language,omitempty"`
		Prompt         string   `json:"prompt,omitempty"`
		ResponseFormat string   `json:"response_format,omitempty"`
		Temperature    *float64 `json:"temperature,omitempty"`
	}
	if cfg, ok := decodeConfig[transcriptionConfig](input.Config); ok {
		if cfg.Language != "" {
			req.Language = cfg.Language
		}
		if cfg.Prompt != "" {
			req.Prompt = cfg.Prompt
		}
		if cfg.ResponseFormat != "" {
			req.ResponseFormat = cfg.ResponseFormat
		}
		if cfg.Temperature != nil {
			req.Temperature = *cfg.Temperature
		}
	}

	// Transcribe audio
	resp, err := a.transcribeAudioInternal(ctx, modelName, req)
	if err != nil {
		return nil, err
	}

	return &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(resp.Text)},
		},
		FinishReason: ai.FinishReasonStop,
	}, nil
}
