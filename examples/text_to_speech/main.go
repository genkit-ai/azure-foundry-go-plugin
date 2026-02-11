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

// Package main demonstrates text-to-speech using genkit.Generate()
package main

import (
	"context"
	"encoding/base64"
	"log"
	"os"

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

	// Define TTS model
	ttsModel := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: azureaifoundry.ModelTTS1,
		Type: "chat",
	}, nil)

	log.Println("Starting text-to-speech with genkit.Generate()...")

	// Example 1: Generate speech with Alloy voice
	log.Println("\n=== Example 1: Alloy voice ===")
	resp1, err := genkit.Generate(ctx, g,
		ai.WithModel(ttsModel),
		ai.WithPrompt("Hello! Welcome to Azure AI Foundry text-to-speech with Genkit."),
		ai.WithConfig(map[string]interface{}{
			"voice":           "alloy",
			"response_format": "mp3",
			"speed":           1.0,
		}),
	)
	if err != nil {
		log.Fatalf("Failed to generate speech: %v", err)
	}

	// Decode base64 audio and save to file
	audioData, err := base64.StdEncoding.DecodeString(resp1.Text())
	if err != nil {
		log.Fatalf("Failed to decode audio: %v", err)
	}

	outputFile1 := "output_alloy.mp3"
	if err := os.WriteFile(outputFile1, audioData, 0644); err != nil {
		log.Fatalf("Failed to save audio: %v", err)
	}
	log.Printf("Audio saved to: %s (size: %d bytes)", outputFile1, len(audioData))

	// Example 2: Generate speech with Nova voice at faster speed
	log.Println("\n=== Example 2: Nova voice (faster speed) ===")
	resp2, err := genkit.Generate(ctx, g,
		ai.WithModel(ttsModel),
		ai.WithPrompt("This is an example of faster speech synthesis using Azure OpenAI."),
		ai.WithConfig(map[string]interface{}{
			"voice":           "nova",
			"response_format": "mp3",
			"speed":           1.5,
		}),
	)
	if err != nil {
		log.Fatalf("Failed to generate speech: %v", err)
	}

	audioData2, err := base64.StdEncoding.DecodeString(resp2.Text())
	if err != nil {
		log.Fatalf("Failed to decode audio: %v", err)
	}

	outputFile2 := "output_nova_fast.mp3"
	if err := os.WriteFile(outputFile2, audioData2, 0644); err != nil {
		log.Fatalf("Failed to save audio: %v", err)
	}
	log.Printf("Audio saved to: %s (size: %d bytes)", outputFile2, len(audioData2))

	// Example 3: HD quality with Echo voice
	log.Println("\n=== Example 3: Echo voice (HD quality) ===")
	ttsHDModel := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: azureaifoundry.ModelTTS1HD,
		Type: "chat",
	}, nil)

	resp3, err := genkit.Generate(ctx, g,
		ai.WithModel(ttsHDModel),
		ai.WithPrompt("High definition audio quality demonstration."),
		ai.WithConfig(map[string]interface{}{
			"voice":           "echo",
			"response_format": "mp3",
			"speed":           1.0,
		}),
	)
	if err != nil {
		log.Fatalf("Failed to generate speech: %v", err)
	}

	audioData3, err := base64.StdEncoding.DecodeString(resp3.Text())
	if err != nil {
		log.Fatalf("Failed to decode audio: %v", err)
	}

	outputFile3 := "output_echo_hd.mp3"
	if err := os.WriteFile(outputFile3, audioData3, 0644); err != nil {
		log.Fatalf("Failed to save audio: %v", err)
	}
	log.Printf("Audio saved to: %s (size: %d bytes)", outputFile3, len(audioData3))

	log.Println("\n✅ Text-to-speech with genkit.Generate() completed successfully!")
}
