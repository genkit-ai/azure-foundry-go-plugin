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
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// inferModelCapabilities infers model capabilities based on model info.
func (a *AzureAIFoundry) inferModelCapabilities(modelName string, supportsMedia bool) *ai.ModelInfo {
	// Detect tool support based on model name
	supportsTools := supportsToolCalling(modelName)
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

func supportsToolCalling(modelName string) bool {
	modelLower := strings.ToLower(modelName)
	if strings.Contains(modelLower, "tts") ||
		strings.Contains(modelLower, "transcribe") ||
		strings.Contains(modelLower, "image") {
		return false
	}

	return strings.Contains(modelLower, "gpt") ||
		strings.Contains(modelLower, "kimi")
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
