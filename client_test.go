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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go/v3/azure"
	"github.com/openai/openai-go/v3/option"
)

const chatCompletionResponse = `{"id":"1","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hello"}}]}`

type capturedRequest struct {
	path          string
	query         string
	apiKey        string
	authorization string
	organization  string
	model         string
}

func newCaptureServer(t *testing.T, response string) (*httptest.Server, func() []capturedRequest) {
	return newCaptureServerWith(t, httptest.NewServer, response)
}

func newCaptureServerWith(
	t *testing.T,
	start func(http.Handler) *httptest.Server,
	response string,
) (*httptest.Server, func() []capturedRequest) {
	t.Helper()
	var mu sync.Mutex
	var requests []capturedRequest
	server := start(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var payload struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &payload)
		mu.Lock()
		requests = append(requests, capturedRequest{
			path:          r.URL.Path,
			query:         r.URL.RawQuery,
			apiKey:        r.Header.Get("Api-Key"),
			authorization: r.Header.Get("Authorization"),
			organization:  r.Header.Get("OpenAI-Organization"),
			model:         payload.Model,
		})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, response)
	}))
	t.Cleanup(server.Close)
	return server, func() []capturedRequest {
		mu.Lock()
		defer mu.Unlock()
		return append([]capturedRequest(nil), requests...)
	}
}

func generateOnce(t *testing.T, plugin *AzureAIFoundry, deployment string) {
	t.Helper()
	plugin.Init(context.Background())
	_, err := plugin.generateText(
		context.Background(),
		ModelDefinition{Name: deployment, Type: ModelTypeChat},
		&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("hi")}},
		nil,
	)
	if err != nil {
		t.Fatalf("generateText() error = %v", err)
	}
}

type fakeCredential struct {
	mu     sync.Mutex
	calls  int
	scopes []string
	expiry time.Duration
}

func (c *fakeCredential) GetToken(_ context.Context, opts policy.TokenRequestOptions) (azcore.AccessToken, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	c.scopes = opts.Scopes
	return azcore.AccessToken{Token: "entra-token", ExpiresOn: time.Now().Add(c.expiry)}, nil
}

func TestFoundryEndpointUsesV1RouteWithAPIKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "must-not-leak")
	t.Setenv("OPENAI_ORG_ID", "must-not-leak")
	server, requests := newCaptureServer(t, chatCompletionResponse)

	generateOnce(t, &AzureAIFoundry{FoundryEndpoint: server.URL, APIKey: "foundry-key"}, "DeepSeek-V3.1")

	got := requests()
	if len(got) != 1 {
		t.Fatalf("requests = %d, want 1", len(got))
	}
	req := got[0]
	if req.path != "/openai/v1/chat/completions" {
		t.Errorf("path = %q, want %q", req.path, "/openai/v1/chat/completions")
	}
	if req.query != "" {
		t.Errorf("query = %q, want no api-version on the v1 route", req.query)
	}
	if req.model != "DeepSeek-V3.1" {
		t.Errorf("model = %q, want the deployment name", req.model)
	}
	if req.authorization != "Bearer foundry-key" {
		t.Errorf("Authorization = %q, want the Foundry key", req.authorization)
	}
	if req.organization != "" {
		t.Errorf("OpenAI-Organization = %q, want it removed", req.organization)
	}
}

func TestFoundryEndpointSendsExplicitAPIVersion(t *testing.T) {
	server, requests := newCaptureServer(t, chatCompletionResponse)

	generateOnce(t, &AzureAIFoundry{
		FoundryEndpoint: server.URL + "/openai/v1/",
		APIKey:          "foundry-key",
		APIVersion:      "preview",
	}, "gpt-5")

	req := requests()[0]
	if req.path != "/openai/v1/chat/completions" {
		t.Errorf("path = %q, want the v1 route without duplication", req.path)
	}
	if req.query != "api-version=preview" {
		t.Errorf("query = %q, want %q", req.query, "api-version=preview")
	}
}

func TestFoundryEndpointUsesTokenCredential(t *testing.T) {
	server, requests := newCaptureServer(t, chatCompletionResponse)
	cred := &fakeCredential{expiry: time.Hour}
	plugin := &AzureAIFoundry{FoundryEndpoint: server.URL, Credential: cred}

	generateOnce(t, plugin, "Llama-4-Maverick")
	if _, err := plugin.generateText(
		context.Background(),
		ModelDefinition{Name: "Llama-4-Maverick", Type: ModelTypeChat},
		&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("again")}},
		nil,
	); err != nil {
		t.Fatalf("second generateText() error = %v", err)
	}

	for _, req := range requests() {
		if req.authorization != "Bearer entra-token" {
			t.Errorf("Authorization = %q, want the Entra ID token", req.authorization)
		}
		if req.apiKey != "" {
			t.Errorf("Api-Key = %q, want none", req.apiKey)
		}
	}
	if cred.calls != 1 {
		t.Errorf("GetToken calls = %d, want the token to be cached", cred.calls)
	}
	if len(cred.scopes) != 1 || cred.scopes[0] != foundryTokenScope {
		t.Errorf("scopes = %v, want [%s]", cred.scopes, foundryTokenScope)
	}
}

func TestBearerTokenSourceRefreshesNearExpiry(t *testing.T) {
	cred := &fakeCredential{expiry: time.Minute}
	source := &bearerTokenSource{credential: cred, scopes: []string{foundryTokenScope}}

	for range 2 {
		if _, err := source.get(context.Background()); err != nil {
			t.Fatalf("get() error = %v", err)
		}
	}
	if cred.calls != 2 {
		t.Errorf("GetToken calls = %d, want a refresh for a token inside the refresh margin", cred.calls)
	}
}

func TestBearerTokenSourceHonorsRefreshOn(t *testing.T) {
	source := &bearerTokenSource{token: azcore.AccessToken{
		Token:     "cached",
		ExpiresOn: time.Now().Add(time.Hour),
		RefreshOn: time.Now().Add(-time.Second),
	}}
	if !source.needsRefresh(time.Now()) {
		t.Error("needsRefresh() = false, want true once RefreshOn has passed")
	}
}

type failingCredential struct{}

func (failingCredential) GetToken(context.Context, policy.TokenRequestOptions) (azcore.AccessToken, error) {
	return azcore.AccessToken{}, errors.New("no identity")
}

func TestFoundryEndpointSurfacesTokenErrors(t *testing.T) {
	server, requests := newCaptureServer(t, chatCompletionResponse)
	plugin := &AzureAIFoundry{FoundryEndpoint: server.URL, Credential: failingCredential{}}
	plugin.Init(context.Background())

	_, err := plugin.generateText(
		context.Background(),
		ModelDefinition{Name: "gpt-5", Type: ModelTypeChat},
		&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("hi")}},
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "no identity") {
		t.Fatalf("error = %v, want the credential error", err)
	}
	if len(requests()) != 0 {
		t.Error("request was sent without credentials")
	}
}

func TestFoundryBaseURL(t *testing.T) {
	tests := []struct {
		endpoint string
		want     string
	}{
		{"https://res.services.ai.azure.com", "https://res.services.ai.azure.com/openai/v1/"},
		{"https://res.services.ai.azure.com/", "https://res.services.ai.azure.com/openai/v1/"},
		{"https://res.services.ai.azure.com/openai/v1", "https://res.services.ai.azure.com/openai/v1/"},
		{"https://res.openai.azure.com/openai/v1/", "https://res.openai.azure.com/openai/v1/"},
		{"https://apim.example.com/foundry", "https://apim.example.com/foundry/openai/v1/"},
		{"http://localhost:8080", "http://localhost:8080/openai/v1/"},
		{"http://127.0.0.1:8080", "http://127.0.0.1:8080/openai/v1/"},
	}
	for _, tt := range tests {
		got, err := foundryBaseURL(tt.endpoint)
		if err != nil {
			t.Errorf("foundryBaseURL(%q) error = %v", tt.endpoint, err)
			continue
		}
		if got.String() != tt.want {
			t.Errorf("foundryBaseURL(%q) = %q, want %q", tt.endpoint, got, tt.want)
		}
	}
}

func TestFoundryBaseURLRejectsUnsafeEndpoints(t *testing.T) {
	for _, endpoint := range []string{
		"http://res.services.ai.azure.com",
		"res.services.ai.azure.com",
		"ftp://res.services.ai.azure.com",
		"https://res.services.ai.azure.com/?api-version=1",
		"://bad",
	} {
		if _, err := foundryBaseURL(endpoint); err == nil {
			t.Errorf("foundryBaseURL(%q) error = nil, want an error", endpoint)
		}
	}
}

func TestClientOptionsValidatesEndpoints(t *testing.T) {
	tests := []struct {
		name    string
		plugin  *AzureAIFoundry
		wantErr string
	}{
		{"missing", &AzureAIFoundry{APIKey: "k"}, "Endpoint or FoundryEndpoint is required"},
		{"both", &AzureAIFoundry{Endpoint: "https://a", FoundryEndpoint: "https://b", APIKey: "k"}, "not both"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.plugin.clientOptions()
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("clientOptions() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

func TestInitPanicsOnInvalidFoundryEndpoint(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil || !strings.HasPrefix(fmt.Sprint(r), "azureaifoundry: ") {
			t.Fatalf("recover() = %v, want an azureaifoundry panic", r)
		}
	}()
	(&AzureAIFoundry{FoundryEndpoint: "http://example.com", APIKey: "k"}).Init(context.Background())
}

type countingTransport struct {
	base  http.RoundTripper
	mu    sync.Mutex
	calls int
}

func (c *countingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	c.mu.Lock()
	c.calls++
	c.mu.Unlock()
	return c.base.RoundTrip(req)
}

func TestClientOptionsKeepAzureOpenAIDeploymentRouting(t *testing.T) {
	server, requests := newCaptureServer(t, chatCompletionResponse)

	generateOnce(t, &AzureAIFoundry{
		Endpoint:      server.URL,
		APIKey:        "azure-key",
		ClientOptions: []option.RequestOption{azure.WithUnsafeAllowHTTP()},
	}, "gpt-4o")

	req := requests()[0]
	if req.path != "/openai/deployments/gpt-4o/chat/completions" {
		t.Errorf("path = %q, want the deployment-scoped route", req.path)
	}
	if req.query != "api-version="+defaultAPIVersion {
		t.Errorf("query = %q, want the default api-version", req.query)
	}
	if req.apiKey != "azure-key" {
		t.Errorf("Api-Key = %q, want %q", req.apiKey, "azure-key")
	}
}

func TestClientOptionsInjectHTTPClient(t *testing.T) {
	for _, tt := range []struct {
		name   string
		plugin func(url string) *AzureAIFoundry
	}{
		{"Endpoint", func(url string) *AzureAIFoundry {
			return &AzureAIFoundry{Endpoint: url, APIKey: "k"}
		}},
		{"FoundryEndpoint", func(url string) *AzureAIFoundry {
			return &AzureAIFoundry{FoundryEndpoint: url, APIKey: "k"}
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			// Plaintext loopback requests bypass custom transports, so serve TLS
			// and trust the test server's certificate through the injected client.
			server, _ := newCaptureServerWith(t, httptest.NewTLSServer, chatCompletionResponse)
			transport := &countingTransport{base: server.Client().Transport}
			plugin := tt.plugin(server.URL)
			plugin.ClientOptions = []option.RequestOption{
				option.WithHTTPClient(&http.Client{Transport: transport}),
			}

			generateOnce(t, plugin, "gpt-4o")

			if transport.calls != 1 {
				t.Errorf("custom transport calls = %d, want 1", transport.calls)
			}
		})
	}
}

func TestClientOptionsApplyAfterPluginOptions(t *testing.T) {
	var attempts int
	var header string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		header = r.Header.Get("X-Request-Source")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	plugin := &AzureAIFoundry{
		FoundryEndpoint: server.URL,
		APIKey:          "k",
		ClientOptions: []option.RequestOption{
			option.WithMaxRetries(0),
			option.WithHeader("X-Request-Source", "genkit"),
		},
	}
	plugin.Init(context.Background())
	_, err := plugin.generateText(
		context.Background(),
		ModelDefinition{Name: "gpt-4o", Type: ModelTypeChat},
		&ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("hi")}},
		nil,
	)
	if err == nil {
		t.Fatal("generateText() error = nil, want the server error")
	}
	if attempts != 1 {
		t.Errorf("attempts = %d, want retries disabled by ClientOptions", attempts)
	}
	if header != "genkit" {
		t.Errorf("X-Request-Source = %q, want %q", header, "genkit")
	}
}
