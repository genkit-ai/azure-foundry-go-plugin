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
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/respjson"
	"github.com/openai/openai-go/v3/shared"
)

// generateText routes a request according to the explicitly configured model
// type, falling back to name-based inference when Type is empty.
func (a *AzureAIFoundry) generateText(ctx context.Context, model ModelDefinition, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	modelType, err := resolveModelType(model)
	if err != nil {
		return nil, err
	}

	switch modelType {
	case ModelTypeImage:
		return a.generateImages(ctx, model.Name, input)
	case ModelTypeTextToSpeech:
		return a.generateSpeech(ctx, model.Name, input)
	case ModelTypeSpeechToText:
		return a.transcribeAudioFromRequest(ctx, model.Name, input)
	}

	params, err := a.buildChatCompletionParams(input, model)
	if err != nil {
		return nil, err
	}

	// Handle streaming vs non-streaming
	if cb != nil {
		return a.generateTextStream(ctx, params, input, cb)
	}
	return a.generateTextSync(ctx, params, input)
}

func resolveModelType(model ModelDefinition) (string, error) {
	if model.Type == "" {
		return inferModelType(model.Name), nil
	}

	switch strings.ToLower(strings.TrimSpace(model.Type)) {
	case ModelTypeChat:
		return ModelTypeChat, nil
	case ModelTypeText:
		return ModelTypeText, nil
	case ModelTypeImage:
		return ModelTypeImage, nil
	case ModelTypeTextToSpeech, "tts":
		return ModelTypeTextToSpeech, nil
	case ModelTypeSpeechToText, "stt", "transcription":
		return ModelTypeSpeechToText, nil
	default:
		return "", fmt.Errorf("azureaifoundry: unsupported model type %q", model.Type)
	}
}

func inferModelType(modelName string) string {
	modelLower := strings.ToLower(modelName)
	if strings.Contains(modelLower, "dall-e") || strings.Contains(modelLower, "gpt-image") {
		return ModelTypeImage
	}
	if strings.Contains(modelLower, "tts") {
		return ModelTypeTextToSpeech
	}
	if strings.Contains(modelLower, "whisper") || strings.Contains(modelLower, "transcribe") {
		return ModelTypeSpeechToText
	}
	return ModelTypeChat
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
			var reasoningContent string
			var toolCalls []openai.ChatCompletionMessageToolCallUnionParam

			for _, part := range msg.Content {
				if part.IsReasoning() {
					reasoningContent += part.Text
				} else if part.IsText() {
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
							ID:   toolCallID(toolReq.Ref, toolReq.Name),
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

			// Round-trip reasoning back to the provider as the non-standard
			// reasoning_content field, mirroring how it is read from responses.
			// openai-go does not model this field, so set it as an extra field.
			if reasoningContent != "" {
				assistantMsg.SetExtraFields(map[string]any{
					"reasoning_content": reasoningContent,
				})
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
							ToolCallID: toolCallID(toolResp.Ref, toolResp.Name),
						},
					})
				}
			}
		}
	}

	return openAIMessages
}

// extractConfigFromRequest safely extracts chat configuration values from a request.
func (a *AzureAIFoundry) extractConfigFromRequest(input *ai.ModelRequest) *GenerationConfig {
	config := &GenerationConfig{}
	parsed, ok := decodeConfig[GenerationConfig](input.Config)
	if !ok {
		return config
	}
	return &parsed
}

// buildChatCompletionParams builds OpenAI chat completion parameters from Genkit request
func (a *AzureAIFoundry) buildChatCompletionParams(input *ai.ModelRequest, model ModelDefinition) (openai.ChatCompletionNewParams, error) {
	messages := a.convertMessagesToOpenAI(input.Messages)

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(model.Name),
		Messages: messages,
	}

	// Apply configuration if provided
	config := a.extractConfigFromRequest(input)
	if config.TopK != nil {
		return openai.ChatCompletionNewParams{}, fmt.Errorf("azureaifoundry: topK is not supported by Azure OpenAI chat completions")
	}
	if config.MaxOutputTokens != nil {
		params.MaxTokens = openai.Int(*config.MaxOutputTokens)
	} else if model.MaxTokens > 0 {
		params.MaxTokens = openai.Int(int64(model.MaxTokens))
	}
	if len(config.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: config.StopSequences,
		}
	}
	if config.Temperature != nil {
		params.Temperature = openai.Float(*config.Temperature)
	}
	if config.TopP != nil {
		params.TopP = openai.Float(*config.TopP)
	}
	if config.Seed != nil {
		params.Seed = openai.Int(*config.Seed)
	}
	if config.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(*config.PresencePenalty)
	}
	if config.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(*config.FrequencyPenalty)
	}
	if config.ParallelToolCalls != nil {
		params.ParallelToolCalls = openai.Bool(*config.ParallelToolCalls)
	}
	params.ResponseFormat = responseFormatForOutput(input.Output)
	if config.ReasoningEffort != nil {
		// https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning?view=foundry-classic&tabs=REST%2Cgpt-5
		reasoningEffortMap := map[string]openai.ReasoningEffort{
			"low":     openai.ReasoningEffortLow,
			"medium":  openai.ReasoningEffortMedium,
			"high":    openai.ReasoningEffortHigh,
			"none":    openai.ReasoningEffortNone,
			"minimal": openai.ReasoningEffortMinimal,
			"xhigh":   openai.ReasoningEffortXhigh,
		}
		if effort, ok := reasoningEffortMap[*config.ReasoningEffort]; ok {
			params.ReasoningEffort = effort
		}
		// Invalid values are ignored, maintaining the default behavior.
	}
	// Handle tools
	toolChoice := config.ToolChoice
	if toolChoice == "" {
		toolChoice = string(input.ToolChoice)
	}
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
		switch toolChoice {
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
		default:
			if toolChoice != "" {
				if !hasToolNamed(input.Tools, toolChoice) {
					return openai.ChatCompletionNewParams{}, fmt.Errorf(
						"azureaifoundry: toolChoice %q does not match any provided tool",
						toolChoice,
					)
				}
				params.ToolChoice = openai.ToolChoiceOptionFunctionToolChoice(
					openai.ChatCompletionNamedToolChoiceFunctionParam{
						Name: toolChoice,
					},
				)
			}
		}
	} else if toolChoice != "" && !isToolChoiceMode(toolChoice) {
		return openai.ChatCompletionNewParams{}, fmt.Errorf(
			"azureaifoundry: toolChoice %q requires a matching tool",
			toolChoice,
		)
	}

	return params, nil
}

func isToolChoiceMode(choice string) bool {
	switch choice {
	case "auto", "required", "none":
		return true
	default:
		return false
	}
}

func hasToolNamed(tools []*ai.ToolDefinition, name string) bool {
	for _, tool := range tools {
		if tool != nil && tool.Name == name {
			return true
		}
	}
	return false
}

func responseFormatForOutput(output *ai.ModelOutputConfig) openai.ChatCompletionNewParamsResponseFormatUnion {
	var format openai.ChatCompletionNewParamsResponseFormatUnion
	if output == nil {
		return format
	}

	switch output.Format {
	case ai.OutputFormatJSON:
		if output.Schema == nil || !output.Constrained {
			jsonObject := shared.NewResponseFormatJSONObjectParam()
			format.OfJSONObject = &jsonObject
			return format
		}
		format.OfJSONSchema = &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:   "output",
				Schema: output.Schema,
				Strict: openai.Bool(output.Constrained),
			},
		}
	case ai.OutputFormatText:
		text := shared.NewResponseFormatTextParam()
		format.OfText = &text
	}

	return format
}

// generateTextSync handles synchronous text generation
func (a *AzureAIFoundry) generateTextSync(ctx context.Context, params openai.ChatCompletionNewParams, originalInput *ai.ModelRequest) (*ai.ModelResponse, error) {
	resp, err := a.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("chat completion failed for model '%s': %w", params.Model, err)
	}

	return a.convertResponse(resp, originalInput), nil
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

	// Reasoning first. reasoning_content is a non-standard field (Kimi, DeepSeek-R1,
	// ...) not exposed by the typed message, so read it from the raw JSON extra fields.
	if reasoning := extractReasoningContent(choice.Message.JSON.ExtraFields); reasoning != "" {
		content = append(content, ai.NewReasoningPart(reasoning, nil))
	}

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
					Ref:   functionToolCall.ID,
				}))
			}
		}
	}

	return &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: content,
		},
		FinishReason: a.convertFinishReason(choice.FinishReason),
		Usage:        convertUsage(resp.Usage),
	}
}

// convertFinishReason converts OpenAI finish reason to Genkit format
func (a *AzureAIFoundry) convertFinishReason(reason string) ai.FinishReason {
	switch reason {
	case "stop", "":
		// An empty reason means the stream ended without a per-choice finish reason;
		// treat it as a clean stop.
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

// convertUsage converts an OpenAI usage payload to Genkit's usage model, including
// reasoning ("thoughts") tokens which reasoning models report separately.
func convertUsage(u openai.CompletionUsage) *ai.GenerationUsage {
	usage := &ai.GenerationUsage{
		InputTokens:  int(u.PromptTokens),
		OutputTokens: int(u.CompletionTokens),
		TotalTokens:  int(u.TotalTokens),
	}
	if reasoning := u.CompletionTokensDetails.ReasoningTokens; reasoning > 0 {
		usage.ThoughtsTokens = int(reasoning)
	}
	return usage
}

// extractReasoningContent reads the non-standard `reasoning_content` field from a
// response's raw JSON extra fields. openai-go v3 does not model it on the typed
// message/delta structs, so reasoning models surface it here. Note that Field.Valid()
// is always false for extra fields (they are never type-checked), so presence is
// determined from Raw() instead. Returns "" when the field is absent or cannot be decoded.
func extractReasoningContent(fields map[string]respjson.Field) string {
	field, ok := fields["reasoning_content"]
	if !ok {
		return ""
	}
	// Raw() is "" when the field was omitted and the literal "null" for a JSON null;
	// otherwise it is the raw JSON value (a quoted string), which we decode.
	raw := field.Raw()
	if raw == "" || raw == "null" {
		return ""
	}
	var content string
	if err := json.Unmarshal([]byte(raw), &content); err != nil {
		return ""
	}
	return content
}

// toolCallID returns the provider-issued tool call reference when available, falling
// back to a name-derived id for transcripts that predate real-id round-tripping.
// Using the real ref keeps assistant tool_calls and their tool results matched even
// when the same tool is called more than once in a single turn.
func toolCallID(ref, name string) string {
	if ref != "" {
		return ref
	}
	return fmt.Sprintf("call_%s", name)
}
