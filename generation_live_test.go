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

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

func livePtr[T any](value T) *T {
	return &value
}

func TestLiveRichGenerationConfig(t *testing.T) {
	if os.Getenv("AZURE_LIVE_TEST") != "1" {
		t.Skip("set AZURE_LIVE_TEST=1 to run live Azure tests")
	}

	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	deployment := os.Getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
	if endpoint == "" || apiKey == "" || deployment == "" {
		t.Fatal("AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME are required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	plugin := &azureaifoundry.AzureAIFoundry{
		Endpoint:   endpoint,
		APIKey:     apiKey,
		APIVersion: os.Getenv("AZURE_OPENAI_API_VERSION"),
	}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	model := plugin.DefineModel(
		g,
		azureaifoundry.ModelDefinition{
			Name: deployment,
			Type: azureaifoundry.ModelTypeChat,
		},
		&ai.ModelInfo{
			Label: deployment,
			Supports: &ai.ModelSupports{
				Multiturn:   true,
				SystemRole:  true,
				Constrained: ai.ConstrainedSupportAll,
			},
		},
	)

	type output struct {
		Answer string `json:"answer"`
	}

	response, err := genkit.Generate(
		ctx,
		g,
		ai.WithModel(model),
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
	if err := json.Unmarshal([]byte(response.Text()), &got); err != nil {
		t.Fatalf("response is not valid structured JSON: %v\nresponse: %s", err, response.Text())
	}
	if got.Answer == "" {
		t.Fatal("structured response contains an empty answer")
	}

	t.Logf("Azure response: %s", got.Answer)
}
