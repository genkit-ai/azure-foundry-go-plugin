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

import "encoding/json"

// Supported model types. An empty model type preserves name-based routing for
// backwards compatibility.
const (
	ModelTypeChat         = "chat"
	ModelTypeText         = "text"
	ModelTypeImage        = "image"
	ModelTypeTextToSpeech = "text-to-speech"
	ModelTypeSpeechToText = "speech-to-text"
)

// ModelDefinition represents a model deployment and its defaults.
type ModelDefinition struct {
	Name          string // Model deployment name in Azure AI Foundry
	Type          string // One of the ModelType constants; empty infers the type from Name
	MaxTokens     int32  // Default maximum output tokens; a per-call value takes precedence
	SupportsMedia bool   // Whether the model supports media (images, audio) (optional)
}

// extractConfig extracts and validates configuration values from a ModelRequest
type modelConfig struct {
	maxTokens       *int64
	temperature     *float64
	topP            *float64
	toolChoice      string
	reasoningEffort *string // "none", "minimal", "low", "medium", "high", "xhigh"
}

// decodeConfig converts a request's Config into T via a JSON round-trip. Genkit delivers
// Config as an untyped value that is usually a map[string]any decoded from JSON — so every
// number is a float64 — but may also be a typed struct. Marshaling and unmarshaling into T
// lets encoding/json coerce the numeric types, instead of relying on exact type assertions
// that only match one of the possible representations. This mirrors how Genkit's own
// plugins read config (internal/base.MapToStruct). It reports false when there is no config
// or the config does not fit T.
func decodeConfig[T any](cfg any) (T, bool) {
	var out T
	if cfg == nil {
		return out, false
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		return out, false
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return out, false
	}
	return out, true
}
