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

// Package main demonstrates streaming text generation with Azure AI Foundry
package main

import (
	"context"
	"fmt"
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

	log.Println("Genkit initialized")
	log.Println("Starting streaming text generation example...")

	// Define GPT-5 model (use your deployment name)
	gpt5Model := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: "gpt-5", // Replace with your actual deployment name
		Type: "chat",
	}, nil)

	// Streaming callback to handle response chunks
	streamCallback := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		if len(chunk.Content) > 0 {
			for _, part := range chunk.Content {
				if part.IsText() {
					fmt.Print(part.Text)
				}
			}
		}
		return nil
	}

	// Generate streaming response
	log.Println("Generating streaming response...")
	response, err := genkit.Generate(ctx, g,
		ai.WithModel(gpt5Model),
		ai.WithPrompt("Write a short story about a robot learning to paint. Keep it under 200 words."),
		ai.WithStreaming(streamCallback),
	)
	fmt.Println() // New line after streaming
	if err != nil {
		log.Printf("Error generating streaming text: %v", err)
	} else {
		log.Printf("\n\nComplete response received. Total tokens: %d", response.Usage.TotalTokens)
	}

	log.Println("Streaming example completed")
}
