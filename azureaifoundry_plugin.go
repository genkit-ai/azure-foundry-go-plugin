// Copyright 2025 Xavier Portilla Edo
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

// test

// Package azureaifoundry provides a comprehensive Azure AI Foundry plugin for Genkit Go.
// This plugin supports text generation and chat capabilities using Azure OpenAI and other models
// available through Azure AI Foundry.
package azureaifoundry

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/azure"
	"github.com/openai/openai-go/v3/option"
)

const provider = "azureaifoundry"

// fileReader wraps a bytes.Reader to provide a filename for multipart uploads
type fileReader struct {
	*bytes.Reader
	name string
}

// Name returns the filename for multipart form uploads
func (f *fileReader) Name() string {
	return f.name
}

// AzureAIFoundry provides configuration options for the Azure AI Foundry plugin.
type AzureAIFoundry struct {
	Endpoint   string                 // Azure AI Foundry endpoint URL (required)
	APIKey     string                 // API key for authentication (required if not using DefaultAzureCredential)
	APIVersion string                 // Azure OpenAI API version (e.g., "2024-12-01-preview", "2024-02-01"). Defaults to "2024-12-01-preview" if not specified
	Credential azcore.TokenCredential // Optional: Use Azure DefaultAzureCredential instead of API key

	mu      sync.Mutex // Mutex to control access
	client  openai.Client
	initted bool // Whether the plugin has been initialized
}

// ModelDefinition represents a model with its name and type.
type ModelDefinition struct {
	Name          string // Model deployment name in Azure AI Foundry
	Type          string // Type: "chat", "text"
	MaxTokens     int32  // Maximum tokens the model can handle (optional)
	SupportsMedia bool   // Whether the model supports media (images, audio) (optional)
}

// Name returns the provider name.
func (a *AzureAIFoundry) Name() string {
	return provider
}

// Init initializes the Azure AI Foundry plugin.
func (a *AzureAIFoundry) Init(ctx context.Context) []api.Action {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.initted {
		panic("azureaifoundry: Init already called")
	}

	// Validate required configuration
	if a.Endpoint == "" {
		panic("azureaifoundry: Endpoint is required")
	}

	// Set default API version if not specified
	apiVersion := a.APIVersion
	if apiVersion == "" {
		apiVersion = "2025-03-01-preview"
	}

	// Create client options using Azure-specific configuration
	var opts []option.RequestOption

	// Use azure.WithEndpoint which properly handles Azure OpenAI deployment-based URLs
	opts = append(opts, azure.WithEndpoint(a.Endpoint, apiVersion))

	if a.APIKey != "" {
		// Use API key authentication
		opts = append(opts, azure.WithAPIKey(a.APIKey))
	} else if a.Credential != nil {
		// Use token credential
		opts = append(opts, azure.WithTokenCredential(a.Credential))
	} else {
		// Try default Azure credential
		cred, err := azidentity.NewDefaultAzureCredential(nil)
		if err != nil {
			panic(fmt.Sprintf("azureaifoundry: failed to create default credential: %v", err))
		}
		opts = append(opts, azure.WithTokenCredential(cred))
	}

	a.client = openai.NewClient(opts...)
	a.initted = true

	return []api.Action{}
}

// DefineModel defines a model in the registry.
func (a *AzureAIFoundry) DefineModel(g *genkit.Genkit, model ModelDefinition, info *ai.ModelInfo) ai.Model {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.initted {
		panic("azureaifoundry: Init not called")
	}

	// Auto-detect model capabilities if not provided
	if info == nil {
		info = a.inferModelCapabilities(model.Name, model.SupportsMedia)
	}

	// Create model metadata
	meta := &ai.ModelOptions{
		Label:    provider + "-" + model.Name,
		Supports: info.Supports,
		Versions: info.Versions,
	}

	// Create the model function
	return genkit.DefineModel(g, api.NewName(provider, model.Name), meta, func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		return a.generateText(ctx, model.Name, input, cb)
	})
}

// DefineEmbedder defines an embedder in the registry.
func (a *AzureAIFoundry) DefineEmbedder(g *genkit.Genkit, modelName string) ai.Embedder {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.initted {
		panic("azureaifoundry: Init not called")
	}

	return genkit.DefineEmbedder(g, api.NewName(provider, modelName), nil, func(
		ctx context.Context,
		req *ai.EmbedRequest,
	) (*ai.EmbedResponse, error) {
		return a.embed(ctx, modelName, req)
	})
}

// ImageGenerationRequest represents a request to generate images
type ImageGenerationRequest struct {
	Prompt         string // The text prompt to generate images from
	N              int    // Number of images to generate (1-10)
	Size           string // Size: "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
	Quality        string // Quality: "standard" or "hd" (DALL-E 3 only)
	Style          string // Style: "vivid" or "natural" (DALL-E 3 only)
	ResponseFormat string // Format: "url" or "b64_json"
}

// ImageGenerationResponse represents the response from image generation
type ImageGenerationResponse struct {
	Images        []GeneratedImage // Generated images
	RevisedPrompt string           // The revised prompt used (DALL-E 3)
}

// GeneratedImage represents a generated image
type GeneratedImage struct {
	URL           string // URL of the generated image (if response_format=url)
	B64JSON       string // Base64-encoded image data (if response_format=b64_json)
	RevisedPrompt string // The revised prompt used for this image
}

// generateImagesInternal generates images using DALL-E models
func (a *AzureAIFoundry) generateImagesInternal(ctx context.Context, modelName string, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	a.mu.Lock()
	if !a.initted {
		a.mu.Unlock()
		return nil, fmt.Errorf("azureaifoundry: client not initialized")
	}
	client := a.client
	a.mu.Unlock()

	// Build image generation parameters
	params := openai.ImageGenerateParams{
		Prompt: req.Prompt,
		Model:  openai.ImageModel(modelName),
	}

	if req.N > 0 {
		params.N = openai.Int(int64(req.N))
	}
	if req.Size != "" {
		params.Size = openai.ImageGenerateParamsSize(req.Size)
	}
	if req.Quality != "" {
		params.Quality = openai.ImageGenerateParamsQuality(req.Quality)
	}
	if req.Style != "" {
		params.Style = openai.ImageGenerateParamsStyle(req.Style)
	}
	if req.ResponseFormat != "" {
		params.ResponseFormat = openai.ImageGenerateParamsResponseFormat(req.ResponseFormat)
	}

	// Generate images
	resp, err := client.Images.Generate(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("image generation failed: %w", err)
	}

	// Convert response
	var images []GeneratedImage
	for _, img := range resp.Data {
		images = append(images, GeneratedImage{
			URL:           img.URL,
			B64JSON:       img.B64JSON,
			RevisedPrompt: img.RevisedPrompt,
		})
	}

	return &ImageGenerationResponse{
		Images: images,
	}, nil
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
		Voice: openai.AudioSpeechNewParamsVoice(req.Voice),
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

// inferModelCapabilities infers model capabilities based on model info.
func (a *AzureAIFoundry) inferModelCapabilities(modelName string, supportsMedia bool) *ai.ModelInfo {
	// Detect tool support based on model name
	supportsTools := strings.Contains(strings.ToLower(modelName), "gpt")
	return &ai.ModelInfo{
		Label: modelName,
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      supportsTools,
			SystemRole: true,
			Media:      supportsMedia,
		},
	}
}

// generateText handles text generation using Azure OpenAI
func (a *AzureAIFoundry) generateText(ctx context.Context, modelName string, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	modelLower := strings.ToLower(modelName)

	// Handle image generation models (DALL-E)
	if strings.Contains(modelLower, "dall-e") || strings.Contains(modelLower, "gpt-image") {
		return a.generateImages(ctx, modelName, input)
	}

	// Handle text-to-speech models
	if strings.Contains(modelLower, "tts-") || strings.Contains(modelLower, "tts") {
		return a.generateSpeech(ctx, modelName, input)
	}

	// Handle speech-to-text models (Whisper, transcribe)
	if strings.Contains(modelLower, "whisper") || strings.Contains(modelLower, "transcribe") {
		return a.transcribeAudioFromRequest(ctx, modelName, input)
	}

	// Default: standard chat completion
	// Build chat completion parameters
	params := a.buildChatCompletionParams(input, modelName)

	// Handle streaming vs non-streaming
	if cb != nil {
		return a.generateTextStream(ctx, params, input, cb)
	}
	return a.generateTextSync(ctx, params, input)
}

// generateImages handles image generation through Genkit's Generate interface
func (a *AzureAIFoundry) generateImages(ctx context.Context, modelName string, input *ai.ModelRequest) (*ai.ModelResponse, error) {
	// Extract prompt from messages
	var prompt string
	for _, msg := range input.Messages {
		for _, part := range msg.Content {
			if part.IsText() {
				prompt += part.Text
			}
		}
	}

	// Extract config if provided
	req := &ImageGenerationRequest{
		Prompt:         prompt,
		N:              1,
		Size:           "1024x1024",
		Quality:        "standard",
		Style:          "vivid",
		ResponseFormat: "url",
	}

	// Apply config from input if available
	if input.Config != nil {
		if configMap, ok := input.Config.(map[string]interface{}); ok {
			if n, ok := configMap["n"].(int); ok {
				req.N = n
			}
			if size, ok := configMap["size"].(string); ok {
				req.Size = size
			}
			if quality, ok := configMap["quality"].(string); ok {
				req.Quality = quality
			}
			if style, ok := configMap["style"].(string); ok {
				req.Style = style
			}
			if format, ok := configMap["response_format"].(string); ok {
				req.ResponseFormat = format
			}
		}
	}

	// Generate images
	resp, err := a.generateImagesInternal(ctx, modelName, req)
	if err != nil {
		return nil, err
	}

	// Convert to ModelResponse
	var content []*ai.Part
	for _, img := range resp.Images {
		if img.URL != "" {
			content = append(content, ai.NewTextPart(img.URL))
		} else if img.B64JSON != "" {
			content = append(content, ai.NewTextPart(img.B64JSON))
		}
	}

	return &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: content,
		},
		FinishReason: ai.FinishReasonStop,
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
	if input.Config != nil {
		if configMap, ok := input.Config.(map[string]interface{}); ok {
			if voice, ok := configMap["voice"].(string); ok {
				req.Voice = voice
			}
			if format, ok := configMap["response_format"].(string); ok {
				req.ResponseFormat = format
			}
			if speed, ok := configMap["speed"].(float64); ok {
				req.Speed = speed
			}
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
	if input.Config != nil {
		if configMap, ok := input.Config.(map[string]interface{}); ok {
			if lang, ok := configMap["language"].(string); ok {
				req.Language = lang
			}
			if prompt, ok := configMap["prompt"].(string); ok {
				req.Prompt = prompt
			}
			if format, ok := configMap["response_format"].(string); ok {
				req.ResponseFormat = format
			}
			if temp, ok := configMap["temperature"].(float64); ok {
				req.Temperature = temp
			}
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

// hasMultimodalContent checks if a message contains multimodal content (text + images)
func (a *AzureAIFoundry) hasMultimodalContent(msg *ai.Message) bool {
	hasText := false
	hasMedia := false

	for _, part := range msg.Content {
		if part.IsText() {
			hasText = true
		}
		if part.IsMedia() {
			hasMedia = true
		}
	}

	// Return true if it has media, or if it has multiple parts (regardless of media)
	return hasMedia || (hasText && len(msg.Content) > 1)
}

// convertMessagesToOpenAI converts Genkit messages to OpenAI message format
func (a *AzureAIFoundry) convertMessagesToOpenAI(messages []*ai.Message) []openai.ChatCompletionMessageParamUnion {
	var openAIMessages []openai.ChatCompletionMessageParamUnion

	for _, msg := range messages {
		if len(msg.Content) == 0 {
			continue // Skip messages with no content
		}

		switch msg.Role {
		case ai.RoleSystem:
			openAIMessages = append(openAIMessages, openai.ChatCompletionMessageParamUnion{
				OfSystem: &openai.ChatCompletionSystemMessageParam{
					Content: openai.ChatCompletionSystemMessageParamContentUnion{
						OfString: openai.String(msg.Content[0].Text),
					},
				},
			})
		case ai.RoleUser:
			// Check if message contains multimodal content (text + images)
			if a.hasMultimodalContent(msg) {
				// Handle multimodal content with array of content parts
				var contentParts []openai.ChatCompletionContentPartUnionParam

				for _, part := range msg.Content {
					if part.IsText() {
						contentParts = append(contentParts, openai.ChatCompletionContentPartUnionParam{
							OfText: &openai.ChatCompletionContentPartTextParam{
								Text: part.Text,
							},
						})
					} else if part.IsMedia() {
						// Handle image/media content
						// Media parts store the URL in the Text field
						contentParts = append(contentParts, openai.ChatCompletionContentPartUnionParam{
							OfImageURL: &openai.ChatCompletionContentPartImageParam{
								ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
									URL: part.Text,
								},
							},
						})
					}
				}

				openAIMessages = append(openAIMessages, openai.ChatCompletionMessageParamUnion{
					OfUser: &openai.ChatCompletionUserMessageParam{
						Content: openai.ChatCompletionUserMessageParamContentUnion{
							OfArrayOfContentParts: contentParts,
						},
					},
				})
			} else {
				// Simple text-only message
				openAIMessages = append(openAIMessages, openai.ChatCompletionMessageParamUnion{
					OfUser: &openai.ChatCompletionUserMessageParam{
						Content: openai.ChatCompletionUserMessageParamContentUnion{
							OfString: openai.String(msg.Content[0].Text),
						},
					},
				})
			}
		case ai.RoleModel:
			// Extract all content parts and tool requests
			var textContent string
			var toolCalls []openai.ChatCompletionMessageToolCallUnionParam

			for _, part := range msg.Content {
				if part.IsText() {
					textContent += part.Text
				} else if part.IsToolRequest() {
					toolReq := part.ToolRequest
					// Marshal the input to JSON string
					argsJSON, err := json.Marshal(toolReq.Input)
					if err != nil {
						continue
					}
					toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
						OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
							ID:   fmt.Sprintf("call_%s", toolReq.Name),
							Type: "function",
							Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
								Name:      toolReq.Name,
								Arguments: string(argsJSON),
							},
						},
					})
				}
			}

			assistantMsg := &openai.ChatCompletionAssistantMessageParam{
				Content: openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(textContent),
				},
			}

			if len(toolCalls) > 0 {
				assistantMsg.ToolCalls = toolCalls
			}

			openAIMessages = append(openAIMessages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: assistantMsg,
			})
		case ai.RoleTool:
			// Handle tool response messages
			for _, part := range msg.Content {
				if part.IsToolResponse() {
					toolResp := part.ToolResponse
					// Marshal the output to JSON string for content
					outputJSON, err := json.Marshal(toolResp.Output)
					if err != nil {
						continue
					}
					openAIMessages = append(openAIMessages, openai.ChatCompletionMessageParamUnion{
						OfTool: &openai.ChatCompletionToolMessageParam{
							Content: openai.ChatCompletionToolMessageParamContentUnion{
								OfString: openai.String(string(outputJSON)),
							},
							ToolCallID: fmt.Sprintf("call_%s", toolResp.Name),
						},
					})
				}
			}
		}
	}

	return openAIMessages
}

// extractConfig extracts and validates configuration values from a ModelRequest
type modelConfig struct {
	maxTokens   *int64
	temperature *float64
	topP        *float64
	toolChoice  string
}

// extractConfigFromRequest safely extracts configuration values from request
func (a *AzureAIFoundry) extractConfigFromRequest(input *ai.ModelRequest) *modelConfig {
	config := &modelConfig{}

	if input.Config == nil {
		return config
	}

	configMap, ok := input.Config.(map[string]interface{})
	if !ok {
		return config
	}

	if maxTokens, ok := configMap["maxOutputTokens"].(int); ok {
		val := int64(maxTokens)
		config.maxTokens = &val
	}
	if temp, ok := configMap["temperature"].(float64); ok {
		config.temperature = &temp
	}
	if topP, ok := configMap["topP"].(float64); ok {
		config.topP = &topP
	}
	if toolChoice, ok := configMap["toolChoice"].(string); ok {
		config.toolChoice = toolChoice
	}

	return config
}

// buildChatCompletionParams builds OpenAI chat completion parameters from Genkit request
func (a *AzureAIFoundry) buildChatCompletionParams(input *ai.ModelRequest, modelName string) openai.ChatCompletionNewParams {
	messages := a.convertMessagesToOpenAI(input.Messages)

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(modelName),
		Messages: messages,
	}

	// Apply configuration if provided
	config := a.extractConfigFromRequest(input)
	if config.maxTokens != nil {
		params.MaxTokens = openai.Int(*config.maxTokens)
	}
	if config.temperature != nil {
		params.Temperature = openai.Float(*config.temperature)
	}
	if config.topP != nil {
		params.TopP = openai.Float(*config.topP)
	}

	// Handle tools
	if len(input.Tools) > 0 {
		var tools []openai.ChatCompletionToolUnionParam
		for _, tool := range input.Tools {
			// Convert Genkit tool definition to OpenAI function tool format
			funcDef := openai.FunctionDefinitionParam{
				Name: tool.Name,
			}
			if tool.Description != "" {
				funcDef.Description = openai.String(tool.Description)
			}
			if tool.InputSchema != nil {
				funcDef.Parameters = tool.InputSchema
			}
			tools = append(tools, openai.ChatCompletionFunctionTool(funcDef))
		}
		params.Tools = tools

		// Set tool choice if specified in config
		switch config.toolChoice {
		case "auto":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String(string(openai.ChatCompletionToolChoiceOptionAutoAuto)),
			}
		case "required":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String(string(openai.ChatCompletionToolChoiceOptionAutoRequired)),
			}
		case "none":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String(string(openai.ChatCompletionToolChoiceOptionAutoNone)),
			}
		}
	}

	return params
}

// generateTextSync handles synchronous text generation
func (a *AzureAIFoundry) generateTextSync(ctx context.Context, params openai.ChatCompletionNewParams, originalInput *ai.ModelRequest) (*ai.ModelResponse, error) {
	resp, err := a.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("chat completion failed for model '%s': %w", params.Model, err)
	}

	return a.convertResponse(resp, originalInput), nil
}

// toolCallAccumulator holds tool call information during streaming
type toolCallAccumulator struct {
	id        string
	name      string
	arguments strings.Builder
}

// generateTextStream handles streaming text generation
func (a *AzureAIFoundry) generateTextStream(ctx context.Context, params openai.ChatCompletionNewParams, originalInput *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Note: Stream parameter is automatically set by NewStreaming
	stream := a.client.Chat.Completions.NewStreaming(ctx, params)
	defer func() {
		if err := stream.Close(); err != nil {
			// Log stream close error but don't override the main error
			_ = err
		}
	}()

	var fullText strings.Builder
	toolCallsMap := make(map[int]*toolCallAccumulator)

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta

			// Handle content streaming
			if delta.Content != "" {
				fullText.WriteString(delta.Content)

				if cb != nil {
					chunkResponse := &ai.ModelResponseChunk{
						Content: []*ai.Part{
							ai.NewTextPart(delta.Content),
						},
					}
					if err := cb(ctx, chunkResponse); err != nil {
						return nil, fmt.Errorf("streaming callback error: %w", err)
					}
				}
			}

			// Handle tool call deltas
			for _, toolCallDelta := range delta.ToolCalls {
				idx := int(toolCallDelta.Index)

				if toolCallsMap[idx] == nil {
					toolCallsMap[idx] = &toolCallAccumulator{
						id: toolCallDelta.ID,
					}
				}

				// Accumulate function name and arguments
				if toolCallDelta.Function.Name != "" {
					toolCallsMap[idx].name = toolCallDelta.Function.Name
				}
				if toolCallDelta.Function.Arguments != "" {
					toolCallsMap[idx].arguments.WriteString(toolCallDelta.Function.Arguments)
				}
			}
		}
	}

	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("stream error: %w", err)
	}

	// Build final message content
	var content []*ai.Part
	if fullText.Len() > 0 {
		content = append(content, ai.NewTextPart(fullText.String()))
	}

	// Add tool calls to content
	toolParts, err := a.convertToolCallsToParts(toolCallsMap)
	if err != nil {
		return nil, fmt.Errorf("failed to convert tool calls: %w", err)
	}
	content = append(content, toolParts...)

	return &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: content,
		},
		FinishReason: ai.FinishReasonStop,
	}, nil
}

// convertToolCallsToParts converts accumulated tool calls to AI parts
func (a *AzureAIFoundry) convertToolCallsToParts(toolCallsMap map[int]*toolCallAccumulator) ([]*ai.Part, error) {
	var parts []*ai.Part

	for _, toolCall := range toolCallsMap {
		if toolCall.name == "" {
			continue
		}

		var args map[string]interface{}
		if toolCall.arguments.Len() > 0 {
			if err := json.Unmarshal([]byte(toolCall.arguments.String()), &args); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tool arguments for '%s': %w", toolCall.name, err)
			}
		}

		parts = append(parts, ai.NewToolRequestPart(&ai.ToolRequest{
			Name:  toolCall.name,
			Input: args,
		}))
	}

	return parts, nil
}

// convertResponse converts OpenAI response to Genkit format
func (a *AzureAIFoundry) convertResponse(resp *openai.ChatCompletion, originalInput *ai.ModelRequest) *ai.ModelResponse {
	if len(resp.Choices) == 0 {
		return &ai.ModelResponse{
			Message: &ai.Message{
				Role:    ai.RoleModel,
				Content: []*ai.Part{},
			},
			FinishReason: ai.FinishReasonUnknown,
		}
	}

	choice := resp.Choices[0]
	var content []*ai.Part

	if choice.Message.Content != "" {
		content = append(content, ai.NewTextPart(choice.Message.Content))
	}

	// Handle tool calls
	if len(choice.Message.ToolCalls) > 0 {
		for _, toolCall := range choice.Message.ToolCalls {
			// Handle function tool calls (most common case)
			if functionToolCall := toolCall.AsFunction(); functionToolCall.ID != "" {
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(functionToolCall.Function.Arguments), &args); err != nil {
					// If we can't parse arguments, skip this tool call
					continue
				}
				content = append(content, ai.NewToolRequestPart(&ai.ToolRequest{
					Name:  functionToolCall.Function.Name,
					Input: args,
				}))
			}
		}
	}

	finishReason := a.convertFinishReason(choice.FinishReason)

	usage := &ai.GenerationUsage{}
	if resp.Usage.PromptTokens > 0 {
		usage.InputTokens = int(resp.Usage.PromptTokens)
		usage.OutputTokens = int(resp.Usage.CompletionTokens)
		usage.TotalTokens = int(resp.Usage.TotalTokens)
	}

	return &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: content,
		},
		FinishReason: finishReason,
		Usage:        usage,
	}
}

// convertFinishReason converts OpenAI finish reason to Genkit format
func (a *AzureAIFoundry) convertFinishReason(reason string) ai.FinishReason {
	switch reason {
	case "stop":
		return ai.FinishReasonStop
	case "length":
		return ai.FinishReasonLength
	case "content_filter":
		return ai.FinishReasonBlocked
	case "tool_calls", "function_call":
		return ai.FinishReasonStop
	default:
		return ai.FinishReasonOther
	}
}

// embed handles embedding generation using Azure OpenAI
func (a *AzureAIFoundry) embed(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	var embeddings []*ai.Embedding

	// Process each document
	for _, doc := range req.Input {
		var inputText string
		// Extract text from document parts
		for _, part := range doc.Content {
			if part.IsText() {
				inputText += part.Text
			}
		}

		if inputText == "" {
			continue // Skip empty documents
		}

		// Call Azure OpenAI embeddings API
		resp, err := a.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
			Model: openai.EmbeddingModel(modelName),
			Input: openai.EmbeddingNewParamsInputUnion{
				OfString: openai.String(inputText),
			},
		})
		if err != nil {
			return nil, fmt.Errorf("embedding generation failed for model '%s': %w", modelName, err)
		}

		// Extract embeddings from response
		if len(resp.Data) > 0 {
			// Convert []float64 to []float32
			embedding := make([]float32, len(resp.Data[0].Embedding))
			for i, val := range resp.Data[0].Embedding {
				embedding[i] = float32(val)
			}

			embeddings = append(embeddings, &ai.Embedding{
				Embedding: embedding,
			})
		}
	}

	return &ai.EmbedResponse{
		Embeddings: embeddings,
	}, nil
}

// DefineCommonModels is a helper to define commonly used Azure OpenAI models
func DefineCommonModels(a *AzureAIFoundry, g *genkit.Genkit) map[string]ai.Model {
	models := make(map[string]ai.Model)
	//GPT-5 models
	models["gpt-5"] = a.DefineModel(g, ModelDefinition{
		Name:          "gpt-5",
		Type:          "chat",
		SupportsMedia: true,
	}, nil)

	// GPT-5 Mini models
	models["gpt-5-mini"] = a.DefineModel(g, ModelDefinition{
		Name:          "gpt-5-mini",
		Type:          "chat",
		SupportsMedia: true,
	}, nil)

	// GPT-4o models
	models["gpt-4o"] = a.DefineModel(g, ModelDefinition{
		Name:          "gpt-4o",
		Type:          "chat",
		SupportsMedia: true,
	}, nil)

	models["gpt-4o-mini"] = a.DefineModel(g, ModelDefinition{
		Name:          "gpt-4o-mini",
		Type:          "chat",
		SupportsMedia: true,
	}, nil)

	// GPT-4 Turbo models
	models["gpt-4-turbo"] = a.DefineModel(g, ModelDefinition{
		Name:          "gpt-4-turbo",
		Type:          "chat",
		SupportsMedia: true,
	}, nil)

	// GPT-4 models
	models["gpt-4"] = a.DefineModel(g, ModelDefinition{
		Name: "gpt-4",
		Type: "chat",
	}, nil)

	// GPT-3.5 Turbo models
	models["gpt-35-turbo"] = a.DefineModel(g, ModelDefinition{
		Name: "gpt-35-turbo",
		Type: "chat",
	}, nil)

	return models
}

// DefineCommonEmbedders is a helper to define commonly used Azure OpenAI embedding models
func DefineCommonEmbedders(a *AzureAIFoundry, g *genkit.Genkit) map[string]ai.Embedder {
	embedders := make(map[string]ai.Embedder)

	// text-embedding-ada-002
	embedders["text-embedding-ada-002"] = a.DefineEmbedder(g, "text-embedding-ada-002")

	// text-embedding-3-small
	embedders["text-embedding-3-small"] = a.DefineEmbedder(g, "text-embedding-3-small")

	// text-embedding-3-large
	embedders["text-embedding-3-large"] = a.DefineEmbedder(g, "text-embedding-3-large")

	return embedders
}

// Common model names for image generation
const (
	ModelDallE2       = "dall-e-2"
	ModelDallE3       = "dall-e-3"
	ModelGPTImageBeta = "gpt-image-1"
)

// Common model names for text-to-speech
const (
	ModelTTS1         = "tts-1"
	ModelTTS1HD       = "tts-1-hd"
	ModelGPT4oMiniTTS = "gpt-4o-mini-tts"
)

// Common model names for speech-to-text
const (
	ModelWhisper1               = "whisper-1"
	ModelGPT4oMiniTranscribe    = "gpt-4o-mini-transcribe"
	ModelGPT4oTranscribe        = "gpt-4o-transcribe"
	ModelGPT4oTranscribeDiarize = "gpt-4o-transcribe-diarize"
)

// Model returns the Model with the given name.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}

// IsDefinedModel reports whether a model is defined.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, api.NewName(provider, name)) != nil
}

// Embedder returns the Embedder with the given name.
func Embedder(g *genkit.Genkit, name string) ai.Embedder {
	return genkit.LookupEmbedder(g, api.NewName(provider, name))
}

// IsDefinedEmbedder reports whether an embedder is defined.
func IsDefinedEmbedder(g *genkit.Genkit, name string) bool {
	return genkit.LookupEmbedder(g, api.NewName(provider, name)) != nil
}
