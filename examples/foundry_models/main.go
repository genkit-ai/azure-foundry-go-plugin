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
// Package main demonstrates Microsoft Foundry Models through the OpenAI v1 endpoint
package main

import (
	"context"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

func main() {
	ctx := context.Background()

	endpoint := os.Getenv("AZURE_FOUNDRY_ENDPOINT") // e.g. https://<resource>.services.ai.azure.com
	if endpoint == "" {
		log.Fatal("AZURE_FOUNDRY_ENDPOINT environment variable must be set")
	}
	deployment := os.Getenv("AZURE_FOUNDRY_DEPLOYMENT") // e.g. DeepSeek-V3.1
	if deployment == "" {
		log.Fatal("AZURE_FOUNDRY_DEPLOYMENT environment variable must be set")
	}

	// FoundryEndpoint sends requests to the resource's /openai/v1/ route, which
	// serves Foundry Models sold by Azure as well as Azure OpenAI models.
	// Without an API key, DefaultAzureCredential is used.
	azurePlugin := &azureaifoundry.AzureAIFoundry{
		FoundryEndpoint: endpoint,
		APIKey:          os.Getenv("AZURE_FOUNDRY_API_KEY"),
	}
	g := genkit.Init(ctx, genkit.WithPlugins(azurePlugin))

	// Capability inference only recognizes OpenAI model families, so describe
	// what this deployment supports explicitly.
	model := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: deployment,
		Type: azureaifoundry.ModelTypeChat,
	}, &ai.ModelInfo{
		Label: deployment,
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			SystemRole: true,
		},
	})

	response, err := genkit.Generate(ctx, g,
		ai.WithModel(model),
		ai.WithPrompt("In two sentences, what is Microsoft Foundry?"),
	)
	if err != nil {
		log.Fatalf("Error generating text: %v", err)
	}

	// Reasoning models such as DeepSeek-R1 also return their reasoning.
	if reasoning := response.Reasoning(); reasoning != "" {
		log.Printf("Reasoning: %s", reasoning)
	}
	log.Printf("Response: %s", response.Text())
}
