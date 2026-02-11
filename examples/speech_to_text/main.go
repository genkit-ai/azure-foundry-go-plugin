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

// Package main demonstrates speech-to-text using genkit.Generate()
package main

import (
	"context"
	"encoding/base64"
	"io"
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

	// Define Whisper model with media support (required for audio input)
	whisperModel := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name:          azureaifoundry.ModelWhisper1,
		Type:          "chat",
		SupportsMedia: true, // Required for media parts (audio)
	}, nil)

	log.Println("Starting speech-to-text with genkit.Generate()...")

	// Example 1: Basic transcription
	log.Println("\n=== Example 1: Basic transcription ===")
	audioFile := "output_alloy.mp3" // From TTS example or your own file

	audio, err := os.Open(audioFile)
	if err != nil {
		log.Printf("Warning: Could not read %s: %v", audioFile, err)
		log.Println("Please run the text_to_speech example first or provide your own audio file.")
		return
	}
	defer func() {
		if closeErr := audio.Close(); closeErr != nil {
			log.Printf("Warning: failed to close audio file: %v", closeErr)
		}
	}()

	audioBytes, err := io.ReadAll(audio)
	if err != nil {
		log.Fatalf("Failed to read audio: %v", err)
	}

	resp1, err := genkit.Generate(ctx, g,
		ai.WithModel(whisperModel),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("Transcribe this audio:"),
			ai.NewMediaPart("audio/mp3", "data:audio/mp3;base64,"+base64.StdEncoding.EncodeToString(audioBytes)),
		)),
		ai.WithConfig(map[string]interface{}{
			"response_format": "json",
		}),
	)
	if err != nil {
		log.Fatalf("Failed to transcribe audio: %v", err)
	}
	log.Printf("Transcribed text: %s", resp1.Text())

	// Example 2: Transcription with language hint
	log.Println("\n=== Example 2: Transcription with language hint ===")
	resp2, err := genkit.Generate(ctx, g,
		ai.WithModel(whisperModel),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("Transcribe this audio:"),
			ai.NewMediaPart("audio/mp3", "data:audio/mp3;base64,"+base64.StdEncoding.EncodeToString(audioBytes)),
		)),
		ai.WithConfig(map[string]interface{}{
			"language":        "en",
			"response_format": "json",
		}),
	)
	if err != nil {
		log.Fatalf("Failed to transcribe audio: %v", err)
	}
	log.Printf("Transcribed text: %s", resp2.Text())

	// Example 3: Transcription with prompt for context
	log.Println("\n=== Example 3: Transcription with prompt ===")
	resp3, err := genkit.Generate(ctx, g,
		ai.WithModel(whisperModel),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("Transcribe this audio:"),
			ai.NewMediaPart("audio/mp3", "data:audio/mp3;base64,"+base64.StdEncoding.EncodeToString(audioBytes)),
		)),
		ai.WithConfig(map[string]interface{}{
			"prompt":          "This is a demonstration of Azure AI Foundry.",
			"response_format": "json",
			"temperature":     0.2,
		}),
	)
	if err != nil {
		log.Fatalf("Failed to transcribe audio: %v", err)
	}
	log.Printf("Transcribed text: %s", resp3.Text())

	log.Println("\n✅ Speech-to-text with genkit.Generate() completed successfully!")
}
