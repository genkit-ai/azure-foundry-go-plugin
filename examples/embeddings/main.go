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

// Package main demonstrates embeddings generation with Azure AI Foundry
package main

import (
	"context"
	"fmt"
	"log"
	"math"

	"github.com/firebase/genkit/go/ai"
	"github.com/xavidop/genkit-azure-foundry-go/examples/common"
)

// cosineSimilarity calculates the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func main() {
	ctx := context.Background()

	// Setup Genkit with Azure AI Foundry
	g, azurePlugin, err := common.SetupGenkit(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to setup Genkit: %v", err)
	}

	log.Println("Starting embeddings example...")

	// Define embedding model (use your deployment name)
	embedder := azurePlugin.DefineEmbedder(g, "text-embedding-ada-002") // Replace with your actual deployment name

	// Example texts to embed
	texts := []string{
		"Azure AI Foundry is a cloud-based AI platform.",
		"Microsoft Azure provides comprehensive AI services.",
		"The weather is sunny today.",
		"Cloud computing enables scalable AI solutions.",
	}

	// Generate embeddings for all texts
	var embeddings []*ai.Embedding
	for _, text := range texts {
		log.Printf("Generating embedding for: %s", text)

		// Create embed request
		embedRequest := &ai.EmbedRequest{
			Input: []*ai.Document{
				ai.DocumentFromText(text, nil),
			},
		}

		// Call the embedder directly
		embedResponse, err := embedder.Embed(ctx, embedRequest)
		if err != nil {
			log.Fatalf("Error generating embedding: %v", err)
		}

		if len(embedResponse.Embeddings) > 0 {
			embeddings = append(embeddings, embedResponse.Embeddings[0])
			log.Printf("✓ Generated embedding with dimension: %d", len(embedResponse.Embeddings[0].Embedding))
		}
	}

	// Calculate and display similarities between texts
	log.Println("\n=== Similarity Analysis ===")
	for i := 0; i < len(texts); i++ {
		for j := i + 1; j < len(texts); j++ {
			similarity := cosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding)
			log.Printf("Similarity between text %d and %d: %.4f", i+1, j+1, similarity)
			log.Printf("  Text %d: %s", i+1, texts[i])
			log.Printf("  Text %d: %s", j+1, texts[j])
			fmt.Println()
		}
	}

	// Find most similar pair
	var maxSim float64
	var maxI, maxJ int
	for i := 0; i < len(texts); i++ {
		for j := i + 1; j < len(texts); j++ {
			similarity := cosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding)
			if similarity > maxSim {
				maxSim = similarity
				maxI = i
				maxJ = j
			}
		}
	}

	log.Printf("\n=== Most Similar Pair (Similarity: %.4f) ===", maxSim)
	log.Printf("Text %d: %s", maxI+1, texts[maxI])
	log.Printf("Text %d: %s", maxJ+1, texts[maxJ])

	log.Println("\nEmbeddings example completed")
}
