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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func TestConvertFinishReason(t *testing.T) {
	a := &AzureAIFoundry{}
	tests := []struct {
		name   string
		reason string
		want   ai.FinishReason
	}{
		{"stop", "stop", ai.FinishReasonStop},
		{"empty defaults to stop", "", ai.FinishReasonStop},
		{"length", "length", ai.FinishReasonLength},
		{"content_filter", "content_filter", ai.FinishReasonBlocked},
		{"tool_calls", "tool_calls", ai.FinishReasonStop},
		{"function_call", "function_call", ai.FinishReasonStop},
		{"unknown", "something_else", ai.FinishReasonOther},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := a.convertFinishReason(tt.reason); got != tt.want {
				t.Fatalf("convertFinishReason(%q) = %v, want %v", tt.reason, got, tt.want)
			}
		})
	}
}

// TestConvertResponseMapsReasoningUsageAndToolRef verifies the non-streaming path
// surfaces reasoning_content as a reasoning part, maps reasoning_tokens to
// ThoughtsTokens, reports the real finish reason, and preserves the provider tool id.
func TestConvertResponseMapsReasoningUsageAndToolRef(t *testing.T) {
	body := `{
		"id": "x",
		"object": "chat.completion",
		"created": 1,
		"model": "kimi",
		"choices": [{
			"index": 0,
			"finish_reason": "content_filter",
			"message": {
				"role": "assistant",
				"content": "Final answer",
				"reasoning_content": "my hidden reasoning",
				"tool_calls": [{
					"id": "call_abc123",
					"type": "function",
					"function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
				}]
			}
		}],
		"usage": {
			"prompt_tokens": 20,
			"completion_tokens": 8,
			"total_tokens": 28,
			"completion_tokens_details": {"reasoning_tokens": 6}
		}
	}`

	var resp openai.ChatCompletion
	if err := json.Unmarshal([]byte(body), &resp); err != nil {
		t.Fatalf("failed to unmarshal completion: %v", err)
	}

	a := &AzureAIFoundry{}
	got := a.convertResponse(&resp, &ai.ModelRequest{})

	if got.FinishReason != ai.FinishReasonBlocked {
		t.Fatalf("FinishReason = %v, want %v", got.FinishReason, ai.FinishReasonBlocked)
	}

	parts := got.Message.Content
	if len(parts) != 3 {
		t.Fatalf("expected 3 content parts (reasoning, text, tool), got %d", len(parts))
	}
	if !parts[0].IsReasoning() || parts[0].Text != "my hidden reasoning" {
		t.Fatalf("part[0] = %+v, want reasoning %q", parts[0], "my hidden reasoning")
	}
	if !parts[1].IsText() || parts[1].Text != "Final answer" {
		t.Fatalf("part[1] = %+v, want text %q", parts[1], "Final answer")
	}
	if !parts[2].IsToolRequest() {
		t.Fatalf("part[2] = %+v, want tool request", parts[2])
	}
	if parts[2].ToolRequest.Name != "get_weather" || parts[2].ToolRequest.Ref != "call_abc123" {
		t.Fatalf("tool request = %+v, want name get_weather ref call_abc123", parts[2].ToolRequest)
	}

	if got.Usage == nil {
		t.Fatal("expected usage, got nil")
	}
	if got.Usage.InputTokens != 20 || got.Usage.OutputTokens != 8 || got.Usage.TotalTokens != 28 {
		t.Fatalf("usage tokens = %+v, want in=20 out=8 total=28", got.Usage)
	}
	if got.Usage.ThoughtsTokens != 6 {
		t.Fatalf("ThoughtsTokens = %d, want 6", got.Usage.ThoughtsTokens)
	}
}

// TestGenerateTextStreamReasoningFinishUsage drives the streaming path end-to-end
// against a mock SSE endpoint and asserts reasoning is both forwarded as chunks and
// accumulated, the real finish reason is used, and usage (incl. ThoughtsTokens) is set.
func TestGenerateTextStreamReasoningFinishUsage(t *testing.T) {
	sse := "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"kimi\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"Thinking. \"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"kimi\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Still thinking.\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"kimi\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"kimi\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n" +
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"kimi\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15,\"completion_tokens_details\":{\"reasoning_tokens\":4}}}\n\n" +
		"data: [DONE]\n\n"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sse))
	}))
	defer server.Close()

	a := &AzureAIFoundry{}
	a.client = openai.NewClient(
		option.WithBaseURL(server.URL+"/"),
		option.WithAPIKey("test"),
	)

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel("kimi"),
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	}

	var streamedReasoning, streamedText string
	cb := func(_ context.Context, chunk *ai.ModelResponseChunk) error {
		for _, p := range chunk.Content {
			switch {
			case p.IsReasoning():
				streamedReasoning += p.Text
			case p.IsText():
				streamedText += p.Text
			}
		}
		return nil
	}

	resp, err := a.generateTextStream(context.Background(), params, &ai.ModelRequest{}, cb)
	if err != nil {
		t.Fatalf("generateTextStream error: %v", err)
	}

	if streamedReasoning != "Thinking. Still thinking." {
		t.Fatalf("streamed reasoning = %q, want %q", streamedReasoning, "Thinking. Still thinking.")
	}
	if streamedText != "Hello" {
		t.Fatalf("streamed text = %q, want %q", streamedText, "Hello")
	}

	if resp.FinishReason != ai.FinishReasonLength {
		t.Fatalf("FinishReason = %v, want %v", resp.FinishReason, ai.FinishReasonLength)
	}

	parts := resp.Message.Content
	if len(parts) != 2 {
		t.Fatalf("expected 2 content parts (reasoning, text), got %d: %+v", len(parts), parts)
	}
	if !parts[0].IsReasoning() || parts[0].Text != "Thinking. Still thinking." {
		t.Fatalf("part[0] = %+v, want accumulated reasoning", parts[0])
	}
	if !parts[1].IsText() || parts[1].Text != "Hello" {
		t.Fatalf("part[1] = %+v, want text Hello", parts[1])
	}

	if resp.Usage == nil {
		t.Fatal("expected usage, got nil")
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 || resp.Usage.TotalTokens != 15 {
		t.Fatalf("usage tokens = %+v, want in=10 out=5 total=15", resp.Usage)
	}
	if resp.Usage.ThoughtsTokens != 4 {
		t.Fatalf("ThoughtsTokens = %d, want 4", resp.Usage.ThoughtsTokens)
	}
}

// TestConvertMessagesToOpenAIToolCallIDRoundTrip verifies that two calls to the same
// tool in one turn keep distinct provider-issued ids (no collision), and that a
// request without a Ref falls back to the name-derived id.
func TestConvertMessagesToOpenAIToolCallIDRoundTrip(t *testing.T) {
	a := &AzureAIFoundry{}
	messages := []*ai.Message{
		{
			Role: ai.RoleModel,
			Content: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "search", Input: map[string]any{"q": "a"}, Ref: "call_1"}),
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "search", Input: map[string]any{"q": "b"}, Ref: "call_2"}),
			},
		},
		{
			Role:    ai.RoleTool,
			Content: []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{Name: "search", Output: "resA", Ref: "call_1"})},
		},
		{
			Role:    ai.RoleTool,
			Content: []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{Name: "search", Output: "resB", Ref: "call_2"})},
		},
	}

	got := a.convertMessagesToOpenAI(messages)
	if len(got) != 3 {
		t.Fatalf("expected 3 openai messages, got %d", len(got))
	}

	assistant := got[0].OfAssistant
	if assistant == nil || len(assistant.ToolCalls) != 2 {
		t.Fatalf("expected assistant message with 2 tool calls, got %+v", got[0])
	}
	id0 := assistant.ToolCalls[0].OfFunction.ID
	id1 := assistant.ToolCalls[1].OfFunction.ID
	if id0 != "call_1" || id1 != "call_2" {
		t.Fatalf("tool call ids = %q, %q, want call_1, call_2", id0, id1)
	}
	if id0 == id1 {
		t.Fatalf("tool call ids collided: %q", id0)
	}

	if got[1].OfTool == nil || got[1].OfTool.ToolCallID != "call_1" {
		t.Fatalf("tool result 0 id = %+v, want call_1", got[1].OfTool)
	}
	if got[2].OfTool == nil || got[2].OfTool.ToolCallID != "call_2" {
		t.Fatalf("tool result 1 id = %+v, want call_2", got[2].OfTool)
	}
}

// TestConvertMessagesToOpenAIToolCallIDFallback verifies the name-derived fallback for
// transcripts that predate real-id round-tripping (no Ref set).
func TestConvertMessagesToOpenAIToolCallIDFallback(t *testing.T) {
	a := &AzureAIFoundry{}
	messages := []*ai.Message{
		{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{Name: "search", Input: map[string]any{"q": "a"}})},
		},
		{
			Role:    ai.RoleTool,
			Content: []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{Name: "search", Output: "resA"})},
		},
	}

	got := a.convertMessagesToOpenAI(messages)
	if len(got) != 2 {
		t.Fatalf("expected 2 openai messages, got %d", len(got))
	}
	if id := got[0].OfAssistant.ToolCalls[0].OfFunction.ID; id != "call_search" {
		t.Fatalf("tool call id = %q, want call_search", id)
	}
	if id := got[1].OfTool.ToolCallID; id != "call_search" {
		t.Fatalf("tool result id = %q, want call_search", id)
	}
}
