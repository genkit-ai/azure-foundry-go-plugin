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

import "testing"

func TestInferModelCapabilitiesDetectsToolCallingModels(t *testing.T) {
	plugin := &AzureAIFoundry{}

	tests := []struct {
		name      string
		modelName string
		wantTools bool
		wantMedia bool
	}{
		{
			name:      "gpt model supports tools",
			modelName: "gpt-5",
			wantTools: true,
		},
		{
			name:      "kimi model supports tools",
			modelName: "Kimi-K2.6",
			wantTools: true,
		},
		{
			name:      "non tool model does not support tools",
			modelName: "dall-e-3",
		},
		{
			name:      "gpt tts model does not support tools",
			modelName: "gpt-4o-mini-tts",
		},
		{
			name:      "gpt transcribe model does not support tools",
			modelName: "gpt-4o-transcribe",
		},
		{
			name:      "gpt image model does not support tools",
			modelName: "gpt-image-1",
		},
		{
			name:      "media flag is preserved",
			modelName: "gpt-4o",
			wantTools: true,
			wantMedia: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := plugin.inferModelCapabilities(tt.modelName, tt.wantMedia)
			if info.Supports.Tools != tt.wantTools {
				t.Fatalf("Tools = %v, want %v", info.Supports.Tools, tt.wantTools)
			}
			if info.Supports.Media != tt.wantMedia {
				t.Fatalf("Media = %v, want %v", info.Supports.Media, tt.wantMedia)
			}
		})
	}
}
