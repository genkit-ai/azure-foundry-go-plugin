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

// Package common provides shared utilities for Azure AI Foundry examples
package common

import (
	"context"
	"fmt"
	"os"

	"github.com/firebase/genkit/go/genkit"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

// Config holds Azure AI Foundry configuration
type Config struct {
	Endpoint string
	APIKey   string
}

// LoadConfig loads configuration from environment variables
func LoadConfig() (*Config, error) {
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")

	if endpoint == "" {
		return nil, fmt.Errorf("AZURE_OPENAI_ENDPOINT environment variable must be set")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("AZURE_OPENAI_API_KEY environment variable must be set")
	}

	return &Config{
		Endpoint: endpoint,
		APIKey:   apiKey,
	}, nil
}

// SetupGenkit initializes Genkit with Azure AI Foundry plugin
func SetupGenkit(ctx context.Context, config *Config) (*genkit.Genkit, *azureaifoundry.AzureAIFoundry, error) {
	if config == nil {
		var err error
		config, err = LoadConfig()
		if err != nil {
			return nil, nil, err
		}
	}

	// Initialize Azure AI Foundry plugin
	azurePlugin := &azureaifoundry.AzureAIFoundry{
		Endpoint: config.Endpoint,
		APIKey:   config.APIKey,
	}

	// Initialize Genkit
	g := genkit.Init(ctx, genkit.WithPlugins(azurePlugin))

	return g, azurePlugin, nil
}
