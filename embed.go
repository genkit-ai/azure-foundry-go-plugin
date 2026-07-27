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

const maxEmbeddingBatchSize = 2048

// embed handles embedding generation using Azure OpenAI
func (a *AzureAIFoundry) embed(ctx context.Context, modelName string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("azureaifoundry: embedding request is nil")
	}

	config, ok := decodeConfig[EmbeddingConfig](req.Options)
	if req.Options != nil && !ok {
		return nil, fmt.Errorf("azureaifoundry: invalid embedding config")
	}
	if config.Dimensions != nil && *config.Dimensions <= 0 {
		return nil, fmt.Errorf("azureaifoundry: embedding dimensions must be greater than zero")
	}
	if config.EncodingFormat != "" && config.EncodingFormat != string(openai.EmbeddingNewParamsEncodingFormatFloat) {
		return nil, fmt.Errorf(
			"azureaifoundry: unsupported embedding encoding_format %q; only %q is supported",
			config.EncodingFormat,
			openai.EmbeddingNewParamsEncodingFormatFloat,
		)
	}

	var inputs []string
	for _, doc := range req.Input {
		if doc == nil {
			continue
		}

		var inputText string
		for _, part := range doc.Content {
			if part.IsText() {
				inputText += part.Text
			}
		}

		if inputText == "" {
			continue
		}
		inputs = append(inputs, inputText)
	}

	embeddings := make([]*ai.Embedding, 0, len(inputs))
	for start := 0; start < len(inputs); start += maxEmbeddingBatchSize {
		end := min(start+maxEmbeddingBatchSize, len(inputs))
		batch := inputs[start:end]

		params := openai.EmbeddingNewParams{
			Model: openai.EmbeddingModel(modelName),
			Input: openai.EmbeddingNewParamsInputUnion{
				OfArrayOfStrings: batch,
			},
			EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
		}
		if config.Dimensions != nil {
			params.Dimensions = openai.Int(*config.Dimensions)
		}

		resp, err := a.client.Embeddings.New(ctx, params)
		if err != nil {
			return nil, fmt.Errorf("embedding generation failed for model '%s': %w", modelName, err)
		}

		ordered := make([]*ai.Embedding, len(batch))
		for _, result := range resp.Data {
			if result.Index < 0 || result.Index >= int64(len(batch)) {
				return nil, fmt.Errorf(
					"embedding generation failed for model '%s': response index %d is out of range for batch of %d",
					modelName,
					result.Index,
					len(batch),
				)
			}
			if ordered[result.Index] != nil {
				return nil, fmt.Errorf(
					"embedding generation failed for model '%s': duplicate response index %d",
					modelName,
					result.Index,
				)
			}

			embedding := make([]float32, len(result.Embedding))
			for i, val := range result.Embedding {
				embedding[i] = float32(val)
			}
			ordered[result.Index] = &ai.Embedding{
				Embedding: embedding,
			}
		}
		for index, embedding := range ordered {
			if embedding == nil {
				return nil, fmt.Errorf(
					"embedding generation failed for model '%s': response is missing index %d",
					modelName,
					index,
				)
			}
		}
		embeddings = append(embeddings, ordered...)
	}

	return &ai.EmbedResponse{
		Embeddings: embeddings,
	}, nil
}
