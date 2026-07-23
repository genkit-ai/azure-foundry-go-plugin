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
	"encoding/json"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestInferModelCapabilitiesDetectsToolCallingModels(t *testing.T) {
	plugin := &AzureAIFoundry{}

	tests := []struct {
		name      string
		modelName string
		wantTools bool
		wantMedia bool
	}{
		{
			name:      "gpt model supports tools",
			modelName: "gpt-5",
			wantTools: true,
		},
		{
			name:      "kimi model supports tools",
			modelName: "Kimi-K2.6",
			wantTools: true,
		},
		{
			name:      "non tool model does not support tools",
			modelName: "dall-e-3",
		},
		{
			name:      "gpt tts model does not support tools",
			modelName: "gpt-4o-mini-tts",
		},
		{
			name:      "gpt transcribe model does not support tools",
			modelName: "gpt-4o-transcribe",
		},
		{
			name:      "gpt image model does not support tools",
			modelName: "gpt-image-1",
		},
		{
			name:      "media flag is preserved",
			modelName: "gpt-4o",
			wantTools: true,
			wantMedia: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := plugin.inferModelCapabilities(tt.modelName, tt.wantMedia)
			if info.Supports.Tools != tt.wantTools {
				t.Fatalf("Tools = %v, want %v", info.Supports.Tools, tt.wantTools)
			}
			if info.Supports.Media != tt.wantMedia {
				t.Fatalf("Media = %v, want %v", info.Supports.Media, tt.wantMedia)
			}
		})
	}
}

func TestExtractConfigFromRequestAcceptsJSONNumberTypes(t *testing.T) {
	plugin := &AzureAIFoundry{}

	config := plugin.extractConfigFromRequest(&ai.ModelRequest{
		Config: map[string]interface{}{
			"maxOutputTokens": float64(128),
			"temperature":     1,
			"topP":            float32(0.75),
		},
	})

	if config.MaxOutputTokens == nil || *config.MaxOutputTokens != 128 {
		t.Fatalf("MaxOutputTokens = %v, want 128", config.MaxOutputTokens)
	}
	if config.Temperature == nil || *config.Temperature != 1 {
		t.Fatalf("Temperature = %v, want 1", config.Temperature)
	}
	if config.TopP == nil || *config.TopP != 0.75 {
		t.Fatalf("TopP = %v, want 0.75", config.TopP)
	}
}

func TestBuildChatCompletionParamsMapsRichConfig(t *testing.T) {
	plugin := &AzureAIFoundry{}
	params, err := plugin.buildChatCompletionParams(&ai.ModelRequest{
		Config: map[string]any{
			"stopSequences":     []string{"STOP", "END"},
			"seed":              7,
			"presencePenalty":   0.25,
			"frequencyPenalty":  -0.5,
			"parallelToolCalls": false,
		},
	}, ModelDefinition{Name: "gpt-deployment"})
	if err != nil {
		t.Fatalf("buildChatCompletionParams() error = %v", err)
	}

	if got := params.Stop.OfStringArray; len(got) != 2 || got[0] != "STOP" || got[1] != "END" {
		t.Fatalf("Stop = %v, want [STOP END]", got)
	}
	if !params.Seed.Valid() || params.Seed.Value != 7 {
		t.Fatalf("Seed = %v, want 7", params.Seed)
	}
	if !params.PresencePenalty.Valid() || params.PresencePenalty.Value != 0.25 {
		t.Fatalf("PresencePenalty = %v, want 0.25", params.PresencePenalty)
	}
	if !params.FrequencyPenalty.Valid() || params.FrequencyPenalty.Value != -0.5 {
		t.Fatalf("FrequencyPenalty = %v, want -0.5", params.FrequencyPenalty)
	}
	if !params.ParallelToolCalls.Valid() || params.ParallelToolCalls.Value {
		t.Fatalf("ParallelToolCalls = %v, want explicit false", params.ParallelToolCalls)
	}
}

func TestBuildChatCompletionParamsRejectsTopK(t *testing.T) {
	plugin := &AzureAIFoundry{}
	_, err := plugin.buildChatCompletionParams(&ai.ModelRequest{
		Config: map[string]any{"topK": 40},
	}, ModelDefinition{Name: "gpt-deployment"})
	if err == nil {
		t.Fatal("buildChatCompletionParams() error = nil, want unsupported topK error")
	}
	if got, want := err.Error(), "azureaifoundry: topK is not supported by Azure OpenAI chat completions"; got != want {
		t.Fatalf("buildChatCompletionParams() error = %q, want %q", got, want)
	}
}

func TestExtractConfigFromRequestAcceptsGenerationCommonConfig(t *testing.T) {
	plugin := &AzureAIFoundry{}
	config := plugin.extractConfigFromRequest(&ai.ModelRequest{
		Config: ai.GenerationCommonConfig{
			StopSequences: []string{"STOP"},
			TopK:          20,
		},
	})

	if len(config.StopSequences) != 1 || config.StopSequences[0] != "STOP" {
		t.Fatalf("StopSequences = %v, want [STOP]", config.StopSequences)
	}
	if config.TopK == nil || *config.TopK != 20 {
		t.Fatalf("TopK = %v, want 20", config.TopK)
	}
}

func TestExtractConfigFromRequestAcceptsTypedConfigWithZeroValues(t *testing.T) {
	plugin := &AzureAIFoundry{}
	zeroInt := int64(0)
	zeroFloat := 0.0
	falseValue := false

	config := plugin.extractConfigFromRequest(&ai.ModelRequest{
		Config: GenerationConfig{
			Seed:              &zeroInt,
			Temperature:       &zeroFloat,
			PresencePenalty:   &zeroFloat,
			FrequencyPenalty:  &zeroFloat,
			ParallelToolCalls: &falseValue,
		},
	})

	if config.Seed == nil || *config.Seed != 0 {
		t.Fatalf("Seed = %v, want explicit 0", config.Seed)
	}
	if config.Temperature == nil || *config.Temperature != 0 {
		t.Fatalf("Temperature = %v, want explicit 0", config.Temperature)
	}
	if config.PresencePenalty == nil || *config.PresencePenalty != 0 {
		t.Fatalf("PresencePenalty = %v, want explicit 0", config.PresencePenalty)
	}
	if config.FrequencyPenalty == nil || *config.FrequencyPenalty != 0 {
		t.Fatalf("FrequencyPenalty = %v, want explicit 0", config.FrequencyPenalty)
	}
	if config.ParallelToolCalls == nil || *config.ParallelToolCalls {
		t.Fatalf("ParallelToolCalls = %v, want explicit false", config.ParallelToolCalls)
	}
}

func TestBuildChatCompletionParamsSerializesRichConfig(t *testing.T) {
	plugin := &AzureAIFoundry{}
	params, err := plugin.buildChatCompletionParams(&ai.ModelRequest{
		Config: map[string]any{
			"stopSequences":     []string{"STOP", "END"},
			"seed":              7,
			"presencePenalty":   0.25,
			"frequencyPenalty":  -0.5,
			"parallelToolCalls": false,
		},
		Output: &ai.ModelOutputConfig{
			Format:      ai.OutputFormatJSON,
			Constrained: true,
			Schema: map[string]any{
				"type": "object",
			},
		},
	}, ModelDefinition{Name: "gpt-deployment"})
	if err != nil {
		t.Fatalf("buildChatCompletionParams() error = %v", err)
	}

	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("json.Marshal(params) error = %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("json.Unmarshal(payload) error = %v", err)
	}

	stop, ok := payload["stop"].([]any)
	if !ok || len(stop) != 2 || stop[0] != "STOP" || stop[1] != "END" {
		t.Fatalf("stop = %#v, want [STOP END]", payload["stop"])
	}
	if payload["seed"] != float64(7) {
		t.Fatalf("seed = %#v, want 7", payload["seed"])
	}
	if payload["presence_penalty"] != 0.25 {
		t.Fatalf("presence_penalty = %#v, want 0.25", payload["presence_penalty"])
	}
	if payload["frequency_penalty"] != -0.5 {
		t.Fatalf("frequency_penalty = %#v, want -0.5", payload["frequency_penalty"])
	}
	if parallel, ok := payload["parallel_tool_calls"].(bool); !ok || parallel {
		t.Fatalf("parallel_tool_calls = %#v, want explicit false", payload["parallel_tool_calls"])
	}

	responseFormat, ok := payload["response_format"].(map[string]any)
	if !ok || responseFormat["type"] != "json_schema" {
		t.Fatalf("response_format = %#v, want json_schema", payload["response_format"])
	}
	jsonSchema, ok := responseFormat["json_schema"].(map[string]any)
	if !ok || jsonSchema["name"] != "output" || jsonSchema["strict"] != true {
		t.Fatalf("response_format.json_schema = %#v, want named strict schema", responseFormat["json_schema"])
	}
	schema, ok := jsonSchema["schema"].(map[string]any)
	if !ok || schema["type"] != "object" {
		t.Fatalf("response_format.json_schema.schema = %#v, want object schema", jsonSchema["schema"])
	}
}

func TestResponseFormatForOutput(t *testing.T) {
	t.Run("structured JSON", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"answer": map[string]any{"type": "string"},
			},
		}
		format := responseFormatForOutput(&ai.ModelOutputConfig{
			Format:      ai.OutputFormatJSON,
			Schema:      schema,
			Constrained: true,
		})

		if format.OfJSONSchema == nil {
			t.Fatal("OfJSONSchema = nil, want structured output config")
		}
		if format.OfJSONSchema.JSONSchema.Name != "output" {
			t.Fatalf("schema name = %q, want output", format.OfJSONSchema.JSONSchema.Name)
		}
		if !format.OfJSONSchema.JSONSchema.Strict.Valid() || !format.OfJSONSchema.JSONSchema.Strict.Value {
			t.Fatalf("schema strict = %v, want true", format.OfJSONSchema.JSONSchema.Strict)
		}
		if got, ok := format.OfJSONSchema.JSONSchema.Schema.(map[string]any); !ok || got["type"] != "object" {
			t.Fatalf("schema = %#v, want object schema", format.OfJSONSchema.JSONSchema.Schema)
		}
	})

	t.Run("JSON object", func(t *testing.T) {
		format := responseFormatForOutput(&ai.ModelOutputConfig{Format: ai.OutputFormatJSON})
		if format.OfJSONObject == nil {
			t.Fatal("OfJSONObject = nil, want JSON object config")
		}
	})

	t.Run("schema without native constraint support", func(t *testing.T) {
		format := responseFormatForOutput(&ai.ModelOutputConfig{
			Format: ai.OutputFormatJSON,
			Schema: map[string]any{"type": "object"},
		})
		if format.OfJSONObject == nil {
			t.Fatal("OfJSONObject = nil, want JSON object fallback")
		}
		if format.OfJSONSchema != nil {
			t.Fatal("OfJSONSchema is set, want no native JSON Schema without constrained support")
		}
	})

	t.Run("text", func(t *testing.T) {
		format := responseFormatForOutput(&ai.ModelOutputConfig{Format: ai.OutputFormatText})
		if format.OfText == nil {
			t.Fatal("OfText = nil, want text config")
		}
	})
}

func TestResolveModelType(t *testing.T) {
	tests := []struct {
		name  string
		model ModelDefinition
		want  string
	}{
		{
			name:  "explicit chat type overrides a misleading deployment name",
			model: ModelDefinition{Name: "product-tts-analyzer", Type: ModelTypeChat},
			want:  ModelTypeChat,
		},
		{
			name:  "explicit image type does not depend on deployment name",
			model: ModelDefinition{Name: "creative-deployment", Type: ModelTypeImage},
			want:  ModelTypeImage,
		},
		{
			name:  "empty type retains image name inference",
			model: ModelDefinition{Name: "dall-e-3"},
			want:  ModelTypeImage,
		},
		{
			name:  "empty type retains text to speech name inference",
			model: ModelDefinition{Name: "gpt-4o-mini-tts"},
			want:  ModelTypeTextToSpeech,
		},
		{
			name:  "empty type retains speech to text name inference",
			model: ModelDefinition{Name: "whisper-1"},
			want:  ModelTypeSpeechToText,
		},
		{
			name:  "empty type defaults to chat",
			model: ModelDefinition{Name: "custom-deployment"},
			want:  ModelTypeChat,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resolveModelType(tt.model)
			if err != nil {
				t.Fatalf("resolveModelType() error = %v", err)
			}
			if got != tt.want {
				t.Fatalf("resolveModelType() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestResolveModelTypeRejectsUnsupportedType(t *testing.T) {
	_, err := resolveModelType(ModelDefinition{Name: "deployment", Type: "video"})
	if err == nil {
		t.Fatal("resolveModelType() error = nil, want unsupported type error")
	}
}

func TestBuildChatCompletionParamsUsesModelMaxTokensAsDefault(t *testing.T) {
	plugin := &AzureAIFoundry{}
	model := ModelDefinition{Name: "gpt-deployment", MaxTokens: 512}

	params, err := plugin.buildChatCompletionParams(&ai.ModelRequest{}, model)
	if err != nil {
		t.Fatalf("buildChatCompletionParams() error = %v", err)
	}
	if !params.MaxTokens.Valid() || params.MaxTokens.Value != 512 {
		t.Fatalf("MaxTokens = %v, want model default 512", params.MaxTokens)
	}

	params, err = plugin.buildChatCompletionParams(&ai.ModelRequest{
		Config: map[string]any{"maxOutputTokens": 128},
	}, model)
	if err != nil {
		t.Fatalf("buildChatCompletionParams() error = %v", err)
	}
	if !params.MaxTokens.Valid() || params.MaxTokens.Value != 128 {
		t.Fatalf("MaxTokens = %v, want per-call override 128", params.MaxTokens)
	}

	params, err = plugin.buildChatCompletionParams(&ai.ModelRequest{}, ModelDefinition{Name: "gpt-deployment"})
	if err != nil {
		t.Fatalf("buildChatCompletionParams() error = %v", err)
	}
	if params.MaxTokens.Valid() {
		t.Fatalf("MaxTokens = %v, want nil without a model default", params.MaxTokens)
	}
}

func TestConvertMessagesToOpenAIPreservesToolRefs(t *testing.T) {
	plugin := &AzureAIFoundry{}
	messages := []*ai.Message{
		{
			Role: ai.RoleModel,
			Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
				Name:  "lookup",
				Ref:   "call_123",
				Input: map[string]interface{}{"q": "azure"},
			})},
		},
		{
			Role: ai.RoleTool,
			Content: []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{
				Name:   "lookup",
				Ref:    "call_123",
				Output: map[string]interface{}{"ok": true},
			})},
		},
	}

	converted := plugin.convertMessagesToOpenAI(messages)
	if got := converted[0].OfAssistant.ToolCalls[0].OfFunction.ID; got != "call_123" {
		t.Fatalf("assistant tool call ID = %q, want call_123", got)
	}
	if got := converted[1].OfTool.ToolCallID; got != "call_123" {
		t.Fatalf("tool response ID = %q, want call_123", got)
	}
}
