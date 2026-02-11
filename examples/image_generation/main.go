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

// Package main demonstrates image generation using genkit.Generate()
package main

import (
	"context"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
	"github.com/xavidop/genkit-azure-foundry-go/examples/common"
)

func main() {
	ctx := context.Background()

	// Setup Genkit with Azure AI Foundry
	g, azurePlugin, err := common.SetupGenkit(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to setup Genkit: %v", err)
	}

	// Define DALL-E model
	dallE3 := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: azureaifoundry.ModelDallE3,
		Type: "chat",
	}, nil)

	log.Println("Starting image generation with genkit.Generate()...")

	// Example 1: Generate image with standard quality
	log.Println("\n=== Example 1: Standard quality image ===")
	resp1, err := genkit.Generate(ctx, g,
		ai.WithModel(dallE3),
		ai.WithPrompt("A serene landscape with mountains and a lake at sunset"),
		ai.WithConfig(map[string]interface{}{
			"n":               1,
			"size":            "1024x1024",
			"quality":         "standard",
			"style":           "vivid",
			"response_format": "url",
		}),
	)
	if err != nil {
		log.Fatalf("Failed to generate image: %v", err)
	}
	log.Printf("Generated image URL: %s", resp1.Text())

	// Example 2: Generate HD quality image
	log.Println("\n=== Example 2: HD quality image ===")
	resp2, err := genkit.Generate(ctx, g,
		ai.WithModel(dallE3),
		ai.WithPrompt("A futuristic cityscape with flying cars, cyberpunk style"),
		ai.WithConfig(map[string]interface{}{
			"n":               1,
			"size":            "1792x1024",
			"quality":         "hd",
			"style":           "vivid",
			"response_format": "url",
		}),
	)
	if err != nil {
		log.Fatalf("Failed to generate image: %v", err)
	}
	log.Printf("Generated HD image URL: %s", resp2.Text())

	log.Println("\n✅ Image generation with genkit.Generate() completed successfully!")
}
