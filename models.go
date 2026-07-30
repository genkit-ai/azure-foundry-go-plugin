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

type modelCapability struct {
	families   []string
	media      bool
	tools      bool
	toolChoice bool
	output     []string
}

// modelCapabilities contains metadata for well-known Azure OpenAI deployment
// families. Entries are ordered from most specific to least specific so broad
// families such as gpt-4 do not capture gpt-4o or gpt-4-turbo deployments.
var modelCapabilities = []modelCapability{
	{
		families:   []string{"gpt-5", "gpt-5-mini", "gpt-5-nano"},
		media:      true,
		tools:      true,
		toolChoice: true,
		output:     []string{"text", "json"},
	},
	{
		families: []string{
			"gpt-5.1",
			"gpt-5.2",
			"gpt-5.4",
			"gpt-5.4-mini",
			"gpt-5.4-nano",
			"gpt-5.5",
		},
		media:      true,
		tools:      true,
		toolChoice: true,
		output:     []string{"text", "json"},
	},
	{
		families: []string{
			"gpt-5.1-chat",
			"gpt-5.2-chat",
			"gpt-5.3-chat",
			"gpt-chat-latest",
		},
		tools:      true,
		toolChoice: true,
		output:     []string{"text", "json"},
	},
	{
		families:   []string{"gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"},
		media:      true,
		tools:      true,
		toolChoice: true,
		output:     []string{"text", "json"},
	},
	{
		families:   []string{"gpt-4o", "gpt-4o-mini"},
		media:      true,
		tools:      true,
		toolChoice: true,
		output:     []string{"text", "json"},
	},
	{
		families:   []string{"gpt-4-turbo", "gpt-4-turbo-preview"},
		tools:      true,
		toolChoice: true,
		output:     []string{"text", "json"},
	},
	{
		families: []string{"gpt-4-vision", "gpt-4-vision-preview", "gpt-4-1106-vision-preview"},
		media:    true,
		output:   []string{"text"},
	},
	{
		families:   []string{"gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-1106", "gpt-4-0125"},
		tools:      true,
		toolChoice: true,
		output:     []string{"text", "json"},
	},
	{
		families:   []string{"gpt-4", "gpt-4-32k"},
		tools:      true,
		toolChoice: false,
		output:     []string{"text"},
	},
	{
		families:   []string{"gpt-35-turbo-1106", "gpt-35-turbo-0125"},
		tools:      true,
		toolChoice: true,
		output:     []string{"text", "json"},
	},
	{
		families:   []string{"gpt-35-turbo", "gpt-35-turbo-16k"},
		tools:      true,
		toolChoice: false,
		output:     []string{"text"},
	},
}

type embedderCapability struct {
	family     string
	dimensions int
}

var embedderCapabilities = []embedderCapability{
	{family: "text-embedding-ada-002", dimensions: 1536},
	{family: "text-embedding-3-small", dimensions: 1536},
	{family: "text-embedding-3-large", dimensions: 3072},
}

// inferModelCapabilities infers model capabilities based on model info.
func (a *AzureAIFoundry) inferModelCapabilities(modelName string, supportsMedia bool) *ai.ModelInfo {
	if !isNonChatModelName(modelName) {
		if capability, ok := lookupModelCapability(modelName); ok {
			return &ai.ModelInfo{
				Label: modelName,
				Supports: &ai.ModelSupports{
					Multiturn:  true,
					Tools:      capability.tools,
					ToolChoice: capability.toolChoice,
					SystemRole: true,
					Media:      capability.media || supportsMedia,
					Output:     append([]string(nil), capability.output...),
				},
			}
		}
	}

	// Preserve the previous substring-based behavior for unknown and custom
	// deployment names.
	supportsTools := supportsToolCallingFallback(modelName)
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

func lookupModelCapability(modelName string) (modelCapability, bool) {
	normalized := strings.ToLower(modelName)
	for _, capability := range modelCapabilities {
		for _, family := range capability.families {
			if matchesModelFamily(normalized, family) {
				return capability, true
			}
		}
	}
	return modelCapability{}, false
}

func matchesModelFamily(modelName, family string) bool {
	if modelName == family {
		return true
	}
	suffix := strings.TrimPrefix(modelName, family+"-")
	if suffix == modelName {
		return false
	}
	return isDateVersion(suffix) || isLegacyVersion(suffix)
}

func isDateVersion(version string) bool {
	if len(version) != len("2006-01-02") || version[4] != '-' || version[7] != '-' {
		return false
	}
	for i, char := range version {
		if i == 4 || i == 7 {
			continue
		}
		if char < '0' || char > '9' {
			return false
		}
	}
	return true
}

func isLegacyVersion(version string) bool {
	switch version {
	case "0301", "0314", "0613", "1106", "0125":
		return true
	default:
		return false
	}
}

func supportsToolCallingFallback(modelName string) bool {
	if isNonChatModelName(modelName) {
		return false
	}

	modelLower := strings.ToLower(modelName)
	return strings.Contains(modelLower, "gpt") ||
		strings.Contains(modelLower, "kimi")
}

func isNonChatModelName(modelName string) bool {
	modelLower := strings.ToLower(modelName)
	return strings.Contains(modelLower, "tts") ||
		strings.Contains(modelLower, "transcribe") ||
		strings.Contains(modelLower, "whisper") ||
		strings.Contains(modelLower, "dall-e") ||
		strings.Contains(modelLower, "image")
}

func inferEmbedderOptions(modelName string) *ai.EmbedderOptions {
	normalized := strings.ToLower(modelName)
	for _, capability := range embedderCapabilities {
		if normalized == capability.family {
			return &ai.EmbedderOptions{
				Label: provider + "-" + modelName,
				Supports: &ai.EmbedderSupports{
					Input: []string{"text"},
				},
				Dimensions: capability.dimensions,
			}
		}
	}
	return nil
}

// DefineCommonModels is a helper to define commonly used Azure OpenAI models
func DefineCommonModels(a *AzureAIFoundry, g *genkit.Genkit) map[string]ai.Model {
	models := make(map[string]ai.Model)
	//GPT-5 models
	models["gpt-5"] = a.DefineModel(g, ModelDefinition{
		Name:          "gpt-5",
		Type:          ModelTypeChat,
		SupportsMedia: true,
	}, nil)

	// GPT-5 Mini models
	models["gpt-5-mini"] = a.DefineModel(g, ModelDefinition{
		Name:          "gpt-5-mini",
		Type:          ModelTypeChat,
		SupportsMedia: true,
	}, nil)

	// GPT-4o models
	models["gpt-4o"] = a.DefineModel(g, ModelDefinition{
		Name:          "gpt-4o",
		Type:          ModelTypeChat,
		SupportsMedia: true,
	}, nil)

	models["gpt-4o-mini"] = a.DefineModel(g, ModelDefinition{
		Name:          "gpt-4o-mini",
		Type:          ModelTypeChat,
		SupportsMedia: true,
	}, nil)

	// GPT-4 Turbo models
	models["gpt-4-turbo"] = a.DefineModel(g, ModelDefinition{
		Name:          "gpt-4-turbo",
		Type:          ModelTypeChat,
		SupportsMedia: true,
	}, nil)

	// GPT-4 models
	models["gpt-4"] = a.DefineModel(g, ModelDefinition{
		Name: "gpt-4",
		Type: ModelTypeChat,
	}, nil)

	// GPT-3.5 Turbo models
	models["gpt-35-turbo"] = a.DefineModel(g, ModelDefinition{
		Name: "gpt-35-turbo",
		Type: ModelTypeChat,
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
