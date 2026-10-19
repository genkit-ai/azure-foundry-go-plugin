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
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/openai/openai-go/v3/azure"
	"github.com/openai/openai-go/v3/option"
)

const (
	defaultAPIVersion = "2025-03-01-preview"

	// foundryV1Path is the OpenAI-compatible v1 route of a Microsoft Foundry
	// resource. It serves Azure OpenAI models and Foundry Models sold by Azure
	// (DeepSeek, Llama, Mistral, Grok, ...) with the deployment name passed in
	// the request's model field.
	foundryV1Path = "/openai/v1"

	// foundryTokenScope is the Microsoft Entra ID scope for Foundry Models
	// inference through the v1 route.
	foundryTokenScope = "https://ai.azure.com/.default"

	// tokenRefreshMargin refreshes cached tokens shortly before they expire so
	// in-flight requests do not race the expiry.
	tokenRefreshMargin = 5 * time.Minute
)

// clientOptions builds the openai-go request options for the configured
// connection mode.
func (a *AzureAIFoundry) clientOptions() ([]option.RequestOption, error) {
	switch {
	case a.Endpoint != "" && a.FoundryEndpoint != "":
		return nil, errors.New("set either Endpoint or FoundryEndpoint, not both")
	case a.FoundryEndpoint != "":
		return a.foundryClientOptions()
	case a.Endpoint != "":
		return a.azureOpenAIClientOptions()
	default:
		return nil, errors.New("an Endpoint or FoundryEndpoint is required")
	}
}

// azureOpenAIClientOptions targets an Azure OpenAI resource using
// deployment-scoped URLs ({endpoint}/openai/deployments/{deployment}/...).
func (a *AzureAIFoundry) azureOpenAIClientOptions() ([]option.RequestOption, error) {
	apiVersion := a.APIVersion
	if apiVersion == "" {
		apiVersion = defaultAPIVersion
	}

	// Use azure.WithEndpoint which properly handles Azure OpenAI deployment-based URLs
	opts := []option.RequestOption{azure.WithEndpoint(a.Endpoint, apiVersion)}

	if a.APIKey != "" {
		opts = append(opts, azure.WithAPIKey(a.APIKey))
		return opts, nil
	}

	cred, err := a.tokenCredential()
	if err != nil {
		return nil, err
	}
	return append(opts, azure.WithTokenCredential(cred)), nil
}

// foundryClientOptions targets the OpenAI v1 route of a Microsoft Foundry
// resource ({endpoint}/openai/v1/...). The v1 route uses implicit versioning,
// so api-version is only sent when APIVersion is set explicitly (for example,
// "preview").
//
// azure.WithEndpoint cannot be used here because it rewrites every request onto
// deployment-scoped paths, and the azure authentication options require it, so
// authentication is applied directly.
func (a *AzureAIFoundry) foundryClientOptions() ([]option.RequestOption, error) {
	baseURL, err := foundryBaseURL(a.FoundryEndpoint)
	if err != nil {
		return nil, err
	}

	opts := []option.RequestOption{
		option.WithBaseURL(baseURL.String()),
		// Do not forward OpenAI organization or project headers picked up from
		// OPENAI_ORG_ID / OPENAI_PROJECT_ID to Azure.
		option.WithHeaderDel("OpenAI-Organization"),
		option.WithHeaderDel("OpenAI-Project"),
	}
	if a.APIVersion != "" {
		opts = append(opts, option.WithQuery("api-version", a.APIVersion))
	}
	if baseURL.Scheme == "http" {
		// foundryBaseURL only allows plain HTTP for loopback hosts, which the SDK
		// additionally enforces for each request.
		opts = append(opts, option.WithUnsafeAllowHTTP())
	}

	if a.APIKey != "" {
		// The v1 route accepts the resource key as a bearer token. Setting it
		// explicitly also prevents OPENAI_API_KEY from being sent to Azure.
		return append(opts, option.WithAPIKey(a.APIKey)), nil
	}

	cred, err := a.tokenCredential()
	if err != nil {
		return nil, err
	}
	tokens := &bearerTokenSource{credential: cred, scopes: []string{foundryTokenScope}}
	return append(opts, option.WithMiddleware(tokens.middleware)), nil
}

// tokenCredential returns the configured credential, falling back to
// DefaultAzureCredential.
func (a *AzureAIFoundry) tokenCredential() (azcore.TokenCredential, error) {
	if a.Credential != nil {
		return a.Credential, nil
	}
	cred, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create default credential: %w", err)
	}
	return cred, nil
}

// foundryBaseURL normalizes a Foundry resource endpoint to its v1 route. Both
// the resource root (https://<resource>.services.ai.azure.com) and the full
// route (https://<resource>.services.ai.azure.com/openai/v1) are accepted.
func foundryBaseURL(endpoint string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(endpoint))
	if err != nil {
		return nil, fmt.Errorf("invalid FoundryEndpoint: %w", err)
	}
	if u.Host == "" {
		return nil, fmt.Errorf("invalid FoundryEndpoint %q: an absolute URL is required", endpoint)
	}
	switch u.Scheme {
	case "https":
	case "http":
		// Plain HTTP would expose credentials, so it is only allowed for local
		// development and test servers.
		if !isLoopbackHost(u.Hostname()) {
			return nil, fmt.Errorf("invalid FoundryEndpoint %q: https is required", endpoint)
		}
	default:
		return nil, fmt.Errorf("invalid FoundryEndpoint %q: https is required", endpoint)
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return nil, fmt.Errorf("invalid FoundryEndpoint %q: query parameters and fragments are not supported", endpoint)
	}

	path := strings.TrimRight(u.Path, "/")
	if !strings.HasSuffix(path, foundryV1Path) {
		path += foundryV1Path
	}
	u.Path = path + "/"
	u.RawPath = ""
	return u, nil
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// bearerTokenSource authenticates requests with Microsoft Entra ID tokens,
// caching each token until it is close to expiry.
type bearerTokenSource struct {
	credential azcore.TokenCredential
	scopes     []string

	mu    sync.Mutex
	token azcore.AccessToken
}

func (s *bearerTokenSource) middleware(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	token, err := s.get(req.Context())
	if err != nil {
		return nil, fmt.Errorf("azureaifoundry: failed to acquire token: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	return next(req)
}

func (s *bearerTokenSource) get(ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.token.Token != "" && !s.needsRefresh(time.Now()) {
		return s.token.Token, nil
	}
	token, err := s.credential.GetToken(ctx, policy.TokenRequestOptions{Scopes: s.scopes})
	if err != nil {
		return "", err
	}
	s.token = token
	return token.Token, nil
}

func (s *bearerTokenSource) needsRefresh(now time.Time) bool {
	if !s.token.RefreshOn.IsZero() && !now.Before(s.token.RefreshOn) {
		return true
	}
	return !now.Add(tokenRefreshMargin).Before(s.token.ExpiresOn)
}
