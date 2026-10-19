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

package azureaifoundry_test

// Live tests call real Azure deployments. They are skipped unless
// AZURE_LIVE_TEST=1, and each test is additionally gated on the environment
// variable naming the deployment it exercises:
//
//	AZURE_OPENAI_ENDPOINT             Azure OpenAI resource endpoint (Endpoint mode)
//	AZURE_OPENAI_API_KEY              optional; DefaultAzureCredential is used when unset
//	AZURE_OPENAI_API_VERSION          optional API version
//	AZURE_LIVE_CHAT_DEPLOYMENT        chat sync, streaming, tool calls, structured output
//	                                  (AZURE_OPENAI_DEPLOYMENT_NAME is accepted as a fallback)
//	AZURE_LIVE_EMBEDDING_DEPLOYMENT   embeddings
//	AZURE_LIVE_IMAGE_DEPLOYMENT       image generation
//	AZURE_LIVE_TTS_DEPLOYMENT         text-to-speech
//	AZURE_LIVE_STT_DEPLOYMENT         speech-to-text
//
//	AZURE_FOUNDRY_ENDPOINT            Microsoft Foundry resource endpoint (FoundryEndpoint mode)
//	AZURE_FOUNDRY_API_KEY             optional; DefaultAzureCredential is used when unset
//	AZURE_LIVE_FOUNDRY_DEPLOYMENT     chat deployment served through the OpenAI v1 route
//
// Example:
//
//	AZURE_LIVE_TEST=1 AZURE_LIVE_CHAT_DEPLOYMENT=gpt-4o go test -run TestLive -v .

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

const liveTimeout = 2 * time.Minute

func livePtr[T any](value T) *T {
	return &value
}

// liveEnv skips the test unless live tests are enabled and every named
// variable is set, returning the variables' values in order.
func liveEnv(t *testing.T, names ...string) []string {
	t.Helper()
	if os.Getenv("AZURE_LIVE_TEST") != "1" {
		t.Skip("set AZURE_LIVE_TEST=1 to run live Azure tests")
	}
	values := make([]string, len(names))
	for i, name := range names {
		values[i] = os.Getenv(name)
		if values[i] == "" {
			t.Skipf("set %s to run this live test", name)
		}
	}
	return values
}

func liveChatDeployment(t *testing.T) string {
	t.Helper()
	if deployment := os.Getenv("AZURE_LIVE_CHAT_DEPLOYMENT"); deployment != "" {
		return deployment
	}
	if deployment := os.Getenv("AZURE_OPENAI_DEPLOYMENT_NAME"); deployment != "" {
		return deployment
	}
	return liveEnv(t, "AZURE_LIVE_CHAT_DEPLOYMENT")[0]
}

// liveAzureOpenAI initializes Genkit with an Endpoint-mode plugin.
func liveAzureOpenAI(t *testing.T) (context.Context, *genkit.Genkit, *azureaifoundry.AzureAIFoundry) {
	t.Helper()
	endpoint := liveEnv(t, "AZURE_OPENAI_ENDPOINT")[0]
	return liveGenkit(t, &azureaifoundry.AzureAIFoundry{
		Endpoint:   endpoint,
		APIKey:     os.Getenv("AZURE_OPENAI_API_KEY"),
		APIVersion: os.Getenv("AZURE_OPENAI_API_VERSION"),
	})
}

func liveGenkit(t *testing.T, plugin *azureaifoundry.AzureAIFoundry) (context.Context, *genkit.Genkit, *azureaifoundry.AzureAIFoundry) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), liveTimeout)
	t.Cleanup(cancel)
	return ctx, genkit.Init(ctx, genkit.WithPlugins(plugin)), plugin
}

func liveChatModel(g *genkit.Genkit, plugin *azureaifoundry.AzureAIFoundry, deployment string) ai.Model {
	return plugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: deployment,
		Type: azureaifoundry.ModelTypeChat,
	}, &ai.ModelInfo{
		Label: deployment,
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			SystemRole:  true,
			Tools:       true,
			ToolChoice:  true,
			Constrained: ai.ConstrainedSupportAll,
		},
	})
}

func TestLiveChatSync(t *testing.T) {
	deployment := liveChatDeployment(t)
	ctx, g, plugin := liveAzureOpenAI(t)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(liveChatModel(g, plugin, deployment)),
		ai.WithPrompt("Reply with the single word: pong"),
		ai.WithConfig(map[string]any{"maxOutputTokens": 50}),
	)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Text()), "pong") {
		t.Errorf("Text() = %q, want it to contain %q", resp.Text(), "pong")
	}
	if resp.Usage == nil || resp.Usage.TotalTokens == 0 {
		t.Errorf("Usage = %+v, want token usage", resp.Usage)
	}
}

func TestLiveChatStreaming(t *testing.T) {
	deployment := liveChatDeployment(t)
	ctx, g, plugin := liveAzureOpenAI(t)

	var chunks int
	var streamed strings.Builder
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(liveChatModel(g, plugin, deployment)),
		ai.WithPrompt("Count from one to five in words, separated by spaces."),
		ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
			chunks++
			streamed.WriteString(chunk.Text())
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}
	if chunks == 0 {
		t.Fatal("no streaming chunks received")
	}
	if streamed.String() != resp.Text() {
		t.Errorf("streamed text %q does not match final text %q", streamed.String(), resp.Text())
	}
	if resp.Usage == nil || resp.Usage.TotalTokens == 0 {
		t.Errorf("Usage = %+v, want streaming token usage", resp.Usage)
	}
}

func TestLiveChatToolCalls(t *testing.T) {
	deployment := liveChatDeployment(t)
	ctx, g, plugin := liveAzureOpenAI(t)

	var called bool
	weather := genkit.DefineTool(g, "get_weather", "Returns the current weather for a city.",
		func(_ *ai.ToolContext, input struct {
			City string `json:"city"`
		}) (string, error) {
			called = true
			return "It is sunny and 21C in " + input.City + ".", nil
		},
	)

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(liveChatModel(g, plugin, deployment)),
		ai.WithTools(weather),
		ai.WithPrompt("What is the weather in Lisbon? Use the tool."),
		ai.WithMaxTurns(2),
	)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}
	if !called {
		t.Fatal("tool was not called")
	}
	if !strings.Contains(strings.ToLower(resp.Text()), "sunny") {
		t.Errorf("Text() = %q, want the tool result to be used", resp.Text())
	}
}

func TestLiveRichGenerationConfig(t *testing.T) {
	deployment := liveChatDeployment(t)
	ctx, g, plugin := liveAzureOpenAI(t)

	type output struct {
		Answer string `json:"answer"`
	}

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(liveChatModel(g, plugin, deployment)),
		ai.WithPrompt("Reply with a short greeting."),
		ai.WithConfig(azureaifoundry.GenerationConfig{
			MaxOutputTokens:   livePtr[int64](100),
			Temperature:       livePtr(0.0),
			Seed:              livePtr[int64](7),
			PresencePenalty:   livePtr(0.0),
			FrequencyPenalty:  livePtr(0.0),
			ParallelToolCalls: livePtr(false),
		}),
		ai.WithOutputType(output{}),
	)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}

	var got output
	if err := json.Unmarshal([]byte(resp.Text()), &got); err != nil {
		t.Fatalf("response is not valid structured JSON: %v\nresponse: %s", err, resp.Text())
	}
	if got.Answer == "" {
		t.Fatal("structured response contains an empty answer")
	}
}

func TestLiveEmbeddings(t *testing.T) {
	deployment := liveEnv(t, "AZURE_LIVE_EMBEDDING_DEPLOYMENT")[0]
	ctx, g, plugin := liveAzureOpenAI(t)

	resp, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(plugin.DefineEmbedder(g, deployment)),
		ai.WithTextDocs("first document", "second document", "third document"),
	)
	if err != nil {
		t.Fatalf("Embed() error: %v", err)
	}
	if len(resp.Embeddings) != 3 {
		t.Fatalf("embeddings = %d, want one per document", len(resp.Embeddings))
	}
	for i, embedding := range resp.Embeddings {
		if len(embedding.Embedding) == 0 {
			t.Errorf("embedding %d is empty", i)
		}
	}
}

func TestLiveImageGeneration(t *testing.T) {
	deployment := liveEnv(t, "AZURE_LIVE_IMAGE_DEPLOYMENT")[0]
	ctx, g, plugin := liveAzureOpenAI(t)

	model := plugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: deployment,
		Type: azureaifoundry.ModelTypeImage,
	}, nil)
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithPrompt("A simple flat illustration of a lighthouse."),
	)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}
	if len(resp.Message.Content) == 0 || !resp.Message.Content[0].IsImage() {
		t.Fatalf("content = %#v, want an image media part", resp.Message.Content)
	}
}

func TestLiveTextToSpeech(t *testing.T) {
	deployment := liveEnv(t, "AZURE_LIVE_TTS_DEPLOYMENT")[0]
	ctx, g, plugin := liveAzureOpenAI(t)

	model := plugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: deployment,
		Type: azureaifoundry.ModelTypeTextToSpeech,
	}, nil)
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithPrompt("Hello from the live test suite."),
	)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}
	if len(resp.Message.Content) == 0 || !resp.Message.Content[0].IsAudio() {
		t.Fatalf("content = %#v, want an audio media part", resp.Message.Content)
	}
	_, encoded, ok := strings.Cut(resp.Media(), ",")
	if !ok {
		t.Fatalf("media %q is not a data URL", resp.Media())
	}
	audio, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatalf("audio payload is not valid base64: %v", err)
	}
	if len(audio) == 0 {
		t.Error("decoded audio payload is empty")
	}
}

func TestLiveSpeechToText(t *testing.T) {
	deployment := liveEnv(t, "AZURE_LIVE_STT_DEPLOYMENT")[0]
	ctx, g, plugin := liveAzureOpenAI(t)

	audio, err := os.ReadFile("examples/speech_to_text/output_alloy.mp3")
	if err != nil {
		t.Fatalf("read sample audio: %v", err)
	}
	model := plugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name:          deployment,
		Type:          azureaifoundry.ModelTypeSpeechToText,
		SupportsMedia: true,
	}, nil)
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewMediaPart("audio/mpeg", "data:audio/mpeg;base64,"+base64.StdEncoding.EncodeToString(audio)),
		)),
	)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}
	if strings.TrimSpace(resp.Text()) == "" {
		t.Fatal("transcription is empty")
	}
	t.Logf("transcription: %s", resp.Text())
}

func TestLiveFoundryModelsChat(t *testing.T) {
	values := liveEnv(t, "AZURE_FOUNDRY_ENDPOINT", "AZURE_LIVE_FOUNDRY_DEPLOYMENT")
	ctx, g, plugin := liveGenkit(t, &azureaifoundry.AzureAIFoundry{
		FoundryEndpoint: values[0],
		APIKey:          os.Getenv("AZURE_FOUNDRY_API_KEY"),
	})

	model := plugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: values[1],
		Type: azureaifoundry.ModelTypeChat,
	}, &ai.ModelInfo{
		Label:    values[1],
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true},
	})
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithPrompt("Reply with the single word: pong"),
	)
	if err != nil {
		t.Fatalf("Generate() error: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Text()), "pong") {
		t.Errorf("Text() = %q, want it to contain %q", resp.Text(), "pong")
	}
}
