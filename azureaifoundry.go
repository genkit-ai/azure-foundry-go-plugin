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

// Package azureaifoundry provides a comprehensive Azure AI Foundry plugin for Genkit Go.
// This plugin supports text generation and chat capabilities using Azure OpenAI and other models
// available through Azure AI Foundry.
package azureaifoundry

import (
	"context"
	"fmt"
	"sync"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/azure"
	"github.com/openai/openai-go/v3/option"
)

const provider = "azureaifoundry"

// AzureAIFoundry provides configuration options for the Azure AI Foundry plugin.
type AzureAIFoundry struct {
	Endpoint   string                 // Azure AI Foundry endpoint URL (required)
	APIKey     string                 // API key for authentication (required if not using DefaultAzureCredential)
	APIVersion string                 // Azure OpenAI API version (e.g., "2024-12-01-preview", "2024-02-01"). Defaults to "2024-12-01-preview" if not specified
	Credential azcore.TokenCredential // Optional: Use Azure DefaultAzureCredential instead of API key

	mu      sync.Mutex // Mutex to control access
	client  openai.Client
	initted bool // Whether the plugin has been initialized
}

// Name returns the provider name.
func (a *AzureAIFoundry) Name() string {
	return provider
}

// Init initializes the Azure AI Foundry plugin.
func (a *AzureAIFoundry) Init(ctx context.Context) []api.Action {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.initted {
		panic("azureaifoundry: Init already called")
	}

	// Validate required configuration
	if a.Endpoint == "" {
		panic("azureaifoundry: Endpoint is required")
	}

	// Set default API version if not specified
	apiVersion := a.APIVersion
	if apiVersion == "" {
		apiVersion = "2025-03-01-preview"
	}

	// Create client options using Azure-specific configuration
	var opts []option.RequestOption

	// Use azure.WithEndpoint which properly handles Azure OpenAI deployment-based URLs
	opts = append(opts, azure.WithEndpoint(a.Endpoint, apiVersion))

	if a.APIKey != "" {
		// Use API key authentication
		opts = append(opts, azure.WithAPIKey(a.APIKey))
	} else if a.Credential != nil {
		// Use token credential
		opts = append(opts, azure.WithTokenCredential(a.Credential))
	} else {
		// Try default Azure credential
		cred, err := azidentity.NewDefaultAzureCredential(nil)
		if err != nil {
			panic(fmt.Sprintf("azureaifoundry: failed to create default credential: %v", err))
		}
		opts = append(opts, azure.WithTokenCredential(cred))
	}

	a.client = openai.NewClient(opts...)
	a.initted = true

	return []api.Action{}
}

// DefineModel defines a model in the registry.
func (a *AzureAIFoundry) DefineModel(g *genkit.Genkit, model ModelDefinition, info *ai.ModelInfo) ai.Model {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.initted {
		panic("azureaifoundry: Init not called")
	}

	// Auto-detect model capabilities if not provided
	if info == nil {
		info = a.inferModelCapabilities(model.Name, model.SupportsMedia)
	}

	// Create model metadata
	meta := &ai.ModelOptions{
		Label:    provider + "-" + model.Name,
		Supports: info.Supports,
		Versions: info.Versions,
	}

	// Create the model function
	return genkit.DefineModel(g, api.NewName(provider, model.Name), meta, func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		return a.generateText(ctx, model, input, cb)
	})
}

// DefineEmbedder defines an embedder in the registry.
func (a *AzureAIFoundry) DefineEmbedder(g *genkit.Genkit, modelName string) ai.Embedder {
	a.mu.Lock()
	defer a.mu.Unlock()

	if !a.initted {
		panic("azureaifoundry: Init not called")
	}

	return genkit.DefineEmbedder(g, api.NewName(provider, modelName), inferEmbedderOptions(modelName), func(
		ctx context.Context,
		req *ai.EmbedRequest,
	) (*ai.EmbedResponse, error) {
		return a.embed(ctx, modelName, req)
	})
}

// Model returns the Model with the given name.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}

// IsDefinedModel reports whether a model is defined.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, api.NewName(provider, name)) != nil
}

// Embedder returns the Embedder with the given name.
func Embedder(g *genkit.Genkit, name string) ai.Embedder {
	return genkit.LookupEmbedder(g, api.NewName(provider, name))
}

// IsDefinedEmbedder reports whether an embedder is defined.
func IsDefinedEmbedder(g *genkit.Genkit, name string) bool {
	return genkit.LookupEmbedder(g, api.NewName(provider, name)) != nil
}
