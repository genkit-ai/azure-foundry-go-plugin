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

// Package main demonstrates multi-turn chat conversation with Azure AI Foundry
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

	log.Println("Starting multi-turn chat conversation example...")

	// Define GPT-5 model (use your deployment name)
	gpt5Model := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: "gpt-5", // Replace with your actual deployment name
		Type: "chat",
	}, nil)

	// System message to set context
	systemMessage := ai.NewSystemMessage(
		ai.NewTextPart("You are a helpful assistant that specializes in explaining cloud computing concepts in simple terms."),
	)

	// First turn: Ask about cloud computing
	response1, err := genkit.Generate(ctx, g,
		ai.WithModel(gpt5Model),
		ai.WithMessages(
			systemMessage,
			ai.NewUserTextMessage("What is Azure AI Foundry?"),
		),
		ai.WithConfig(map[string]interface{}{
			"temperature":     0.7,
			"maxOutputTokens": 500,
		}),
	)
	if err != nil {
		log.Fatalf("Error in first turn: %v", err)
	}
	log.Printf("Assistant: %s\n", response1.Text())

	// Second turn: Follow-up question (maintaining conversation context)
	response2, err := genkit.Generate(ctx, g,
		ai.WithModel(gpt5Model),
		ai.WithMessages(
			systemMessage,
			ai.NewUserTextMessage("What is Azure AI Foundry?"),
			response1.Message, // Include previous assistant response
			ai.NewUserTextMessage("How does it compare to AWS Bedrock?"),
		),
		ai.WithConfig(map[string]interface{}{
			"temperature":     0.7,
			"maxOutputTokens": 500,
		}),
	)
	if err != nil {
		log.Fatalf("Error in second turn: %v", err)
	}
	log.Printf("Assistant: %s\n", response2.Text())

	// Third turn: Another follow-up
	response3, err := genkit.Generate(ctx, g,
		ai.WithModel(gpt5Model),
		ai.WithMessages(
			systemMessage,
			ai.NewUserTextMessage("What is Azure AI Foundry?"),
			response1.Message,
			ai.NewUserTextMessage("How does it compare to AWS Bedrock?"),
			response2.Message,
			ai.NewUserTextMessage("What are the key advantages of using Azure AI Foundry?"),
		),
		ai.WithConfig(map[string]interface{}{
			"temperature":     0.7,
			"maxOutputTokens": 500,
		}),
	)
	if err != nil {
		log.Fatalf("Error in third turn: %v", err)
	}
	log.Printf("Assistant: %s\n", response3.Text())

	log.Println("Multi-turn chat conversation completed")
}
