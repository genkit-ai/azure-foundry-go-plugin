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
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go/v3"
)

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
	type imageConfig struct {
		N              *int64 `json:"n,omitempty"`
		Size           string `json:"size,omitempty"`
		Quality        string `json:"quality,omitempty"`
		Style          string `json:"style,omitempty"`
		ResponseFormat string `json:"response_format,omitempty"`
	}
	if cfg, ok := decodeConfig[imageConfig](input.Config); ok {
		if cfg.N != nil {
			req.N = int(*cfg.N)
		}
		if cfg.Size != "" {
			req.Size = cfg.Size
		}
		if cfg.Quality != "" {
			req.Quality = cfg.Quality
		}
		if cfg.Style != "" {
			req.Style = cfg.Style
		}
		if cfg.ResponseFormat != "" {
			req.ResponseFormat = cfg.ResponseFormat
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
