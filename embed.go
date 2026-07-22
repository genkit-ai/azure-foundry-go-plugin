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

package azureaifoundry

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go/v3"
)

// embed handles embedding generation using Azure OpenAI
func (a *AzureAIFoundry) embed(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	var embeddings []*ai.Embedding

	// Process each document
	for _, doc := range req.Input {
		var inputText string
		// Extract text from document parts
		for _, part := range doc.Content {
			if part.IsText() {
				inputText += part.Text
			}
		}

		if inputText == "" {
			continue // Skip empty documents
		}

		// Call Azure OpenAI embeddings API
		resp, err := a.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
			Model: openai.EmbeddingModel(modelName),
			Input: openai.EmbeddingNewParamsInputUnion{
				OfString: openai.String(inputText),
			},
		})
		if err != nil {
			return nil, fmt.Errorf("embedding generation failed for model '%s': %w", modelName, err)
		}

		// Extract embeddings from response
		if len(resp.Data) > 0 {
			// Convert []float64 to []float32
			embedding := make([]float32, len(resp.Data[0].Embedding))
			for i, val := range resp.Data[0].Embedding {
				embedding[i] = float32(val)
			}

			embeddings = append(embeddings, &ai.Embedding{
				Embedding: embedding,
			})
		}
	}

	return &ai.EmbedResponse{
		Embeddings: embeddings,
	}, nil
}
