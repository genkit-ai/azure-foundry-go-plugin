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
	"slices"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

func TestInferModelCapabilitiesUsesRegistryAndFallback(t *testing.T) {
	plugin := &AzureAIFoundry{}

	tests := []struct {
		name           string
		modelName      string
		supportsMedia  bool
		wantTools      bool
		wantToolChoice bool
		wantMedia      bool
		wantOutput     []string
	}{
		{
			name:           "gpt 5 family",
			modelName:      "gpt-5-mini",
			wantTools:      true,
			wantToolChoice: true,
			wantMedia:      true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 5 dated version",
			modelName:      "GPT-5-2025-08-07",
			wantTools:      true,
			wantToolChoice: true,
			wantMedia:      true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 5 dot version",
			modelName:      "gpt-5.2-2025-12-11",
			wantTools:      true,
			wantToolChoice: true,
			wantMedia:      true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 5.1 family",
			modelName:      "gpt-5.1",
			wantTools:      true,
			wantToolChoice: true,
			wantMedia:      true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "latest gpt 5 mini variant",
			modelName:      "gpt-5.4-mini-2026-03-17",
			wantTools:      true,
			wantToolChoice: true,
			wantMedia:      true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 5.5 family",
			modelName:      "gpt-5.5-2026-04-24",
			wantTools:      true,
			wantToolChoice: true,
			wantMedia:      true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 5 chat variant",
			modelName:      "gpt-5.3-chat-2026-03-03",
			wantTools:      true,
			wantToolChoice: true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 4.1 family",
			modelName:      "gpt-4.1-mini-2025-04-14",
			wantTools:      true,
			wantToolChoice: true,
			wantMedia:      true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 4o family",
			modelName:      "GPT-4O-2024-11-20",
			wantTools:      true,
			wantToolChoice: true,
			wantMedia:      true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 4 turbo explicit media support",
			modelName:      "gpt-4-turbo-2024-04-09",
			supportsMedia:  true,
			wantTools:      true,
			wantToolChoice: true,
			wantMedia:      true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 4 turbo defaults to text for deployment compatibility",
			modelName:      "gpt-4-turbo-2024-04-09",
			wantTools:      true,
			wantToolChoice: true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 4 turbo text preview",
			modelName:      "gpt-4-1106-preview",
			wantTools:      true,
			wantToolChoice: true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 4 turbo 1106 version without preview suffix",
			modelName:      "gpt-4-1106",
			wantTools:      true,
			wantToolChoice: true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:           "gpt 4 turbo 0125 version without preview suffix",
			modelName:      "gpt-4-0125",
			wantTools:      true,
			wantToolChoice: true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:       "gpt 4 vision preview",
			modelName:  "gpt-4-vision-preview",
			wantMedia:  true,
			wantOutput: []string{"text"},
		},
		{
			name:       "legacy gpt 4 family",
			modelName:  "gpt-4-0613",
			wantTools:  true,
			wantOutput: []string{"text"},
		},
		{
			name:           "modern gpt 35 turbo family",
			modelName:      "gpt-35-turbo-0125",
			wantTools:      true,
			wantToolChoice: true,
			wantOutput:     []string{"text", "json"},
		},
		{
			name:       "legacy gpt 35 turbo family",
			modelName:  "gpt-35-turbo-0613",
			wantTools:  true,
			wantOutput: []string{"text"},
		},
		{
			name:      "unknown kimi model keeps tool fallback",
			modelName: "Kimi-K2.6",
			wantTools: true,
		},
		{
			name:          "custom gpt deployment keeps fallback and media flag",
			modelName:     "production-gpt-deployment",
			supportsMedia: true,
			wantTools:     true,
			wantMedia:     true,
		},
		{
			name:      "numeric custom suffix uses fallback",
			modelName: "gpt-4-123-prod",
			wantTools: true,
		},
		{
			name:      "unknown model keeps conservative fallback",
			modelName: "custom-deployment",
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
			name:      "gpt whisper deployment does not support tools",
			modelName: "gpt-whisper-1",
		},
		{
			name:      "gpt dall-e deployment does not support tools",
			modelName: "gpt-dall-e-3",
		},
		{
			name:          "media flag augments known text model",
			modelName:     "gpt-4",
			supportsMedia: true,
			wantTools:     true,
			wantMedia:     true,
			wantOutput:    []string{"text"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := plugin.inferModelCapabilities(tt.modelName, tt.supportsMedia)
			if info.Supports.Tools != tt.wantTools {
				t.Fatalf("Tools = %v, want %v", info.Supports.Tools, tt.wantTools)
			}
			if info.Supports.ToolChoice != tt.wantToolChoice {
				t.Fatalf("ToolChoice = %v, want %v", info.Supports.ToolChoice, tt.wantToolChoice)
			}
			if info.Supports.Media != tt.wantMedia {
				t.Fatalf("Media = %v, want %v", info.Supports.Media, tt.wantMedia)
			}
			if !slices.Equal(info.Supports.Output, tt.wantOutput) {
				t.Fatalf("Output = %v, want %v", info.Supports.Output, tt.wantOutput)
			}
			if !info.Supports.Multiturn {
				t.Fatal("Multiturn = false, want true")
			}
			if !info.Supports.SystemRole {
				t.Fatal("SystemRole = false, want true")
			}
		})
	}
}

func TestDefineModelExplicitInfoOverridesRegistry(t *testing.T) {
	plugin := &AzureAIFoundry{initted: true}
	g := genkit.Init(context.Background())

	model := plugin.DefineModel(g, ModelDefinition{Name: "gpt-5"}, &ai.ModelInfo{
		Supports: &ai.ModelSupports{
			Tools:      false,
			ToolChoice: false,
			Media:      false,
			Output:     []string{"text"},
		},
	})

	metadata := model.(api.Action).Desc().Metadata["model"].(map[string]any)
	supports := metadata["supports"].(map[string]any)
	if supports["tools"] != false || supports["toolChoice"] != false || supports["media"] != false {
		t.Fatalf("explicit supports metadata was not preserved: %#v", supports)
	}
	if output, _ := supports["output"].([]string); !slices.Equal(output, []string{"text"}) {
		t.Fatalf("Output = %v, want [text]", output)
	}
}

func TestInferEmbedderOptions(t *testing.T) {
	tests := []struct {
		modelName      string
		wantDimensions int
		wantKnown      bool
	}{
		{modelName: "text-embedding-ada-002", wantDimensions: 1536, wantKnown: true},
		{modelName: "TEXT-EMBEDDING-3-SMALL", wantDimensions: 1536, wantKnown: true},
		{modelName: "text-embedding-3-large", wantDimensions: 3072, wantKnown: true},
		{modelName: "custom-embedding-deployment"},
	}

	for _, tt := range tests {
		t.Run(tt.modelName, func(t *testing.T) {
			opts := inferEmbedderOptions(tt.modelName)
			if !tt.wantKnown {
				if opts != nil {
					t.Fatalf("inferEmbedderOptions() = %#v, want nil", opts)
				}
				return
			}
			if opts == nil {
				t.Fatal("inferEmbedderOptions() = nil, want options")
			}
			if opts.Dimensions != tt.wantDimensions {
				t.Fatalf("Dimensions = %d, want %d", opts.Dimensions, tt.wantDimensions)
			}
			if opts.Supports == nil || !slices.Equal(opts.Supports.Input, []string{"text"}) {
				t.Fatalf("Supports.Input = %v, want [text]", opts.Supports)
			}
		})
	}
}

func TestDefineEmbedderPublishesKnownMetadata(t *testing.T) {
	plugin := &AzureAIFoundry{initted: true}
	g := genkit.Init(context.Background())

	embedder := plugin.DefineEmbedder(g, "text-embedding-3-large")
	info := embedder.(api.Action).Desc().Metadata["info"].(map[string]any)
	if info["dimensions"] != 3072 {
		t.Fatalf("dimensions = %v, want 3072", info["dimensions"])
	}
	supports := info["supports"].(map[string]any)
	if input, _ := supports["input"].([]string); !slices.Equal(input, []string{"text"}) {
		t.Fatalf("supports.input = %v, want [text]", input)
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

func TestBuildChatCompletionParamsToolChoiceModes(t *testing.T) {
	plugin := &AzureAIFoundry{}
	tools := []*ai.ToolDefinition{{Name: "lookup"}}

	for _, mode := range []string{"auto", "required", "none"} {
		t.Run("config "+mode, func(t *testing.T) {
			params, err := plugin.buildChatCompletionParams(&ai.ModelRequest{
				Config: map[string]any{"toolChoice": mode},
				Tools:  tools,
			}, ModelDefinition{Name: "gpt-deployment"})
			if err != nil {
				t.Fatalf("buildChatCompletionParams() error = %v", err)
			}
			if !params.ToolChoice.OfAuto.Valid() || params.ToolChoice.OfAuto.Value != mode {
				t.Fatalf("ToolChoice = %v, want %q mode", params.ToolChoice, mode)
			}
		})

		t.Run("model request "+mode, func(t *testing.T) {
			params, err := plugin.buildChatCompletionParams(&ai.ModelRequest{
				ToolChoice: ai.ToolChoice(mode),
				Tools:      tools,
			}, ModelDefinition{Name: "gpt-deployment"})
			if err != nil {
				t.Fatalf("buildChatCompletionParams() error = %v", err)
			}
			if !params.ToolChoice.OfAuto.Valid() || params.ToolChoice.OfAuto.Value != mode {
				t.Fatalf("ToolChoice = %v, want %q mode", params.ToolChoice, mode)
			}
		})
	}

	t.Run("config takes precedence", func(t *testing.T) {
		params, err := plugin.buildChatCompletionParams(&ai.ModelRequest{
			Config:     map[string]any{"toolChoice": "required"},
			ToolChoice: ai.ToolChoiceNone,
			Tools:      tools,
		}, ModelDefinition{Name: "gpt-deployment"})
		if err != nil {
			t.Fatalf("buildChatCompletionParams() error = %v", err)
		}
		if !params.ToolChoice.OfAuto.Valid() || params.ToolChoice.OfAuto.Value != "required" {
			t.Fatalf("ToolChoice = %v, want config value %q", params.ToolChoice, "required")
		}
	})

	t.Run("unset", func(t *testing.T) {
		params, err := plugin.buildChatCompletionParams(&ai.ModelRequest{
			Tools: tools,
		}, ModelDefinition{Name: "gpt-deployment"})
		if err != nil {
			t.Fatalf("buildChatCompletionParams() error = %v", err)
		}
		if params.ToolChoice.OfAuto.Valid() || params.ToolChoice.OfFunctionToolChoice != nil {
			t.Fatalf("ToolChoice = %v, want omitted choice", params.ToolChoice)
		}
	})
}

func TestBuildChatCompletionParamsForcesNamedTool(t *testing.T) {
	plugin := &AzureAIFoundry{}
	tools := []*ai.ToolDefinition{
		{Name: "lookup"},
		{Name: "search"},
	}
	configs := map[string]any{
		"map config": map[string]any{
			"toolChoice": "lookup",
		},
		"typed config": GenerationConfig{
			ToolChoice: "lookup",
		},
	}

	for name, config := range configs {
		t.Run(name, func(t *testing.T) {
			params, err := plugin.buildChatCompletionParams(&ai.ModelRequest{
				Config: config,
				Tools:  tools,
			}, ModelDefinition{Name: "gpt-deployment"})
			if err != nil {
				t.Fatalf("buildChatCompletionParams() error = %v", err)
			}

			function := params.ToolChoice.GetFunction()
			if function == nil || function.Name != "lookup" {
				t.Fatalf("ToolChoice function = %#v, want lookup", function)
			}

			data, err := json.Marshal(params)
			if err != nil {
				t.Fatalf("json.Marshal(params) error = %v", err)
			}
			var payload map[string]any
			if err := json.Unmarshal(data, &payload); err != nil {
				t.Fatalf("json.Unmarshal(payload) error = %v", err)
			}
			choice, ok := payload["tool_choice"].(map[string]any)
			if !ok || choice["type"] != "function" {
				t.Fatalf("tool_choice = %#v, want function choice", payload["tool_choice"])
			}
			selected, ok := choice["function"].(map[string]any)
			if !ok || selected["name"] != "lookup" {
				t.Fatalf("tool_choice.function = %#v, want lookup", choice["function"])
			}
		})
	}
}

func TestBuildChatCompletionParamsRejectsUnknownNamedTool(t *testing.T) {
	plugin := &AzureAIFoundry{}

	t.Run("no tools", func(t *testing.T) {
		_, err := plugin.buildChatCompletionParams(&ai.ModelRequest{
			Config: map[string]any{"toolChoice": "lookup"},
		}, ModelDefinition{Name: "gpt-deployment"})
		if err == nil {
			t.Fatal("buildChatCompletionParams() error = nil, want missing tool error")
		}
		if got, want := err.Error(), `azureaifoundry: toolChoice "lookup" requires a matching tool`; got != want {
			t.Fatalf("buildChatCompletionParams() error = %q, want %q", got, want)
		}
	})

	t.Run("tool not found", func(t *testing.T) {
		_, err := plugin.buildChatCompletionParams(&ai.ModelRequest{
			Config: map[string]any{"toolChoice": "lookup"},
			Tools:  []*ai.ToolDefinition{{Name: "search"}},
		}, ModelDefinition{Name: "gpt-deployment"})
		if err == nil {
			t.Fatal("buildChatCompletionParams() error = nil, want unknown tool error")
		}
		if got, want := err.Error(), `azureaifoundry: toolChoice "lookup" does not match any provided tool`; got != want {
			t.Fatalf("buildChatCompletionParams() error = %q, want %q", got, want)
		}
	})
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
