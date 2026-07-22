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
)

// toolCallAccumulator holds tool call information during streaming
type toolCallAccumulator struct {
	id        string
	name      string
	arguments strings.Builder
}

// generateTextStream handles streaming text generation
func (a *AzureAIFoundry) generateTextStream(ctx context.Context, params openai.ChatCompletionNewParams, originalInput *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Request per-request token usage on the terminal chunk (choices will be empty
	// on that chunk). Without this, streaming responses carry no usage at all.
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}

	// Note: Stream parameter is automatically set by NewStreaming
	stream := a.client.Chat.Completions.NewStreaming(ctx, params)
	defer func() {
		if err := stream.Close(); err != nil {
			// Log stream close error but don't override the main error
			_ = err
		}
	}()

	var fullText strings.Builder
	var reasoningText strings.Builder
	var finishReason string
	var usage openai.CompletionUsage
	var haveUsage bool
	toolCallsMap := make(map[int]*toolCallAccumulator)

	for stream.Next() {
		chunk := stream.Current()

		// The terminal usage chunk (requested via StreamOptions) carries usage but
		// an empty Choices slice, so read it independently of the choices below.
		if chunk.Usage.TotalTokens > 0 {
			usage = chunk.Usage
			haveUsage = true
		}

		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta

			// Capture the real finish reason (e.g. content_filter, length, tool_calls)
			// so Azure content filtering / truncation is distinguishable from a clean stop.
			if chunk.Choices[0].FinishReason != "" {
				finishReason = chunk.Choices[0].FinishReason
			}

			// Handle reasoning streaming. reasoning_content is a non-standard field
			// (Kimi, DeepSeek-R1, ...) not exposed by the typed delta, so read it from
			// the raw JSON extra fields.
			if reasoningDelta := extractReasoningContent(delta.JSON.ExtraFields); reasoningDelta != "" {
				reasoningText.WriteString(reasoningDelta)

				if cb != nil {
					chunkResponse := &ai.ModelResponseChunk{
						Content: []*ai.Part{
							ai.NewReasoningPart(reasoningDelta, nil),
						},
					}
					if err := cb(ctx, chunkResponse); err != nil {
						return nil, fmt.Errorf("streaming callback error: %w", err)
					}
				}
			}

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
				} else if toolCallsMap[idx].id == "" && toolCallDelta.ID != "" {
					toolCallsMap[idx].id = toolCallDelta.ID
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

	// Build final message content: reasoning first, then text, then tool calls.
	var content []*ai.Part
	if reasoningText.Len() > 0 {
		content = append(content, ai.NewReasoningPart(reasoningText.String(), nil))
	}
	if fullText.Len() > 0 {
		content = append(content, ai.NewTextPart(fullText.String()))
	}

	// Add tool calls to content
	toolParts, err := a.convertToolCallsToParts(toolCallsMap)
	if err != nil {
		return nil, fmt.Errorf("failed to convert tool calls: %w", err)
	}
	content = append(content, toolParts...)

	response := &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: content,
		},
		FinishReason: a.convertFinishReason(finishReason),
	}
	if haveUsage {
		response.Usage = convertUsage(usage)
	}
	return response, nil
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
			Ref:   toolCall.id,
		}))
	}

	return parts, nil
}
