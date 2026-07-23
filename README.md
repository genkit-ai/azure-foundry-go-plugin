# Azure AI Foundry Plugin for Genkit Go

A comprehensive Azure AI Foundry plugin for Genkit Go that provides text generation and chat capabilities using Azure OpenAI and other models available through Azure AI Foundry.

<!-- TOC -->

- [Azure AI Foundry Plugin for Genkit Go](#azure-ai-foundry-plugin-for-genkit-go)
	- [Features](#features)
	- [Supported Models](#supported-models)
		- [Text Generation Models (with Tool Calling Support)](#text-generation-models-with-tool-calling-support)
	- [Installation](#installation)
	- [Quick Start](#quick-start)
		- [Initialize the Plugin](#initialize-the-plugin)
		- [Define Models and Generate Text](#define-models-and-generate-text)
	- [Configuration Options](#configuration-options)
		- [Available Configuration](#available-configuration)
		- [Per-call Generation Configuration](#per-call-generation-configuration)
	- [Azure Setup and Authentication](#azure-setup-and-authentication)
		- [Getting Your Endpoint and API Key](#getting-your-endpoint-and-api-key)
		- [Authentication Methods](#authentication-methods)
			- [1. API Key Authentication (Quick Start)](#1-api-key-authentication-quick-start)
			- [2. Azure Default Credential (Recommended for Production)](#2-azure-default-credential-recommended-for-production)
			- [3. Managed Identity (Azure Deployments)](#3-managed-identity-azure-deployments)
			- [4. Client Secret Credential (Service Principal)](#4-client-secret-credential-service-principal)
			- [5. Azure CLI Credential (Local Development)](#5-azure-cli-credential-local-development)
		- [Model Deployments](#model-deployments)
	- [Examples Directory](#examples-directory)
		- [Running Examples](#running-examples)
	- [Features in Detail](#features-in-detail)
		- [🔧 Tool Calling (Function Calling)](#-tool-calling-function-calling)
		- [🖼️ Multimodal Support (Vision)](#️-multimodal-support-vision)
		- [📡 Streaming](#-streaming)
		- [💬 Multi-turn Conversations](#-multi-turn-conversations)
		- [🔢 Embeddings](#-embeddings)
		- [🎨 Image Generation](#-image-generation)
		- [🗣️ Text-to-Speech](#️-text-to-speech)
		- [🎙️ Speech-to-Text](#️-speech-to-text)
	- [Troubleshooting](#troubleshooting)
		- [Common Issues](#common-issues)
	- [Contributing](#contributing)
	- [License](#license)
	- [Acknowledgments](#acknowledgments)

<!-- /TOC -->

## Features

- **Text Generation**: Support for GPT-5, GPT-5 mini, GPT-4o, GPT-4o mini, GPT-4 Turbo, GPT-4, and GPT-3.5 Turbo models
- **Embeddings**: Support for text-embedding-ada-002, text-embedding-3-small, and text-embedding-3-large models
- **Image Generation**: Support for creating images from text prompts
- **Text-to-Speech**: Convert text to natural-sounding speech with multiple voices
- **Speech-to-Text**: Transcribe audio to text using with subtitle support
- **Streaming**: Full streaming support for real-time responses
- **Tool Calling**: Complete function calling capabilities for GPT-4 and GPT-3.5-turbo models
- **Multimodal Support**: Support for text + image inputs (vision models like GPT-5, GPT-4o and GPT-4 Turbo)
- **Multi-turn Conversations**: Full support for chat history and context management
- **Type Safety**: Robust type conversion and schema validation
- **Flexible Authentication**: Support for API keys, Azure Default Credential, and custom token credentials

## Supported Models

### Text Generation Models (with Tool Calling Support)

- **GPT-5**: Latest advanced model (check Azure for availability)
- **GPT-5 mini**: Smaller, faster version of GPT-5
- **GPT-4o**: multimodal model with vision capabilities
- **GPT-4o mini**: Smaller, faster version of GPT-4o
- **GPT-4 Turbo**: High-performance GPT-4 with vision support
- **GPT-4**: Standard GPT-4 model
- **GPT-3.5 Turbo**: Fast and cost-effective model

All GPT-5, GPT-4 and GPT-3.5-turbo models support function calling (tools).

## Installation

```bash
go get github.com/xavidop/genkit-azure-foundry-go
```

## Quick Start

### Initialize the Plugin

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

func main() {
	ctx := context.Background()

	// Initialize Azure AI Foundry plugin
	azurePlugin := &azureaifoundry.AzureAIFoundry{
		Endpoint: os.Getenv("AZURE_OPENAI_ENDPOINT"),
		APIKey:   os.Getenv("AZURE_OPENAI_API_KEY"),
	}

	// Initialize Genkit
	g := genkit.Init(ctx,
		genkit.WithPlugins(azurePlugin),
		genkit.WithDefaultModel("azureaifoundry/gpt-5"),
	)

	// Optional: Define common models for easy access
	azureaifoundry.DefineCommonModels(azurePlugin, g)

	log.Println("Starting basic Azure AI Foundry example...")

	// Example: Generate text (basic usage)
	response, err := genkit.Generate(ctx, g,
		ai.WithPrompt("What are the key benefits of using Azure AI Foundry?"),
	)
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		log.Printf("Response: %s", response.Text())
	}
}
```

### Define Models and Generate Text

```go
package main

import (
	"context"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

func main() {
	ctx := context.Background()

	azurePlugin := &azureaifoundry.AzureAIFoundry{
		Endpoint: "https://your-resource.openai.azure.com/",
		APIKey:   "your-api-key",
	}

	g := genkit.Init(ctx,
		genkit.WithPlugins(azurePlugin),
	)

	// Define a GPT-5 model (use your deployment name)
	gpt5Model := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name:          "gpt-5", // Your deployment name in Azure
		Type:          azureaifoundry.ModelTypeChat,
		SupportsMedia: true,
	}, nil)

	// Generate text
	response, err := genkit.Generate(ctx, g,
		ai.WithModel(gpt4Model),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("Explain quantum computing in simple terms."),
		)),
	)

	if err != nil {
		log.Fatal(err)
	}

	log.Println(response.Text())
}
```

## Configuration Options

The plugin supports various configuration options:

```go
azurePlugin := &azureaifoundry.AzureAIFoundry{
	Endpoint:   "https://your-resource.openai.azure.com/",
	APIKey:     "your-api-key",              // Use API key
	// OR use Azure credential
	// Credential: azidentity.NewDefaultAzureCredential(),
	APIVersion: "2024-02-15-preview",        // Optional
}
```

### Available Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `Endpoint` | `string` | *required* | Azure OpenAI endpoint URL |
| `APIKey` | `string` | "" | API key for authentication |
| `Credential` | `azcore.TokenCredential` | `nil` | Azure credential (alternative to API key) |
| `APIVersion` | `string` | Latest | API version to use |

### Per-call Generation Configuration

Pass generation settings with `ai.WithConfig`. JSON-compatible maps,
`ai.GenerationCommonConfig`, and the plugin's typed `GenerationConfig` are supported.

| Field | Type | Description |
|-------|------|-------------|
| `maxOutputTokens` | number | Maximum tokens generated for this call |
| `stopSequences` | string array | Stop generation when any sequence is encountered |
| `temperature` | number | Sampling temperature |
| `topK` | number | Not supported by Azure OpenAI chat completions; supplying it returns an error |
| `topP` | number | Nucleus-sampling probability |
| `seed` | number | Best-effort deterministic sampling seed |
| `presencePenalty` | number | Penalizes tokens that have already appeared |
| `frequencyPenalty` | number | Penalizes tokens based on repetition frequency |
| `parallelToolCalls` | boolean | Enables or disables parallel tool calls |
| `toolChoice` | string | Tool-selection mode (`auto`, `required`, or `none`) or the exact name of a tool to force |
| `reasoningEffort` | string | Reasoning level: `none`, `minimal`, `low`, `medium`, `high`, or `xhigh` |

```go
response, err := genkit.Generate(ctx, g,
	ai.WithModel(gpt5Model),
	ai.WithPrompt("Summarize the release notes as JSON."),
	ai.WithConfig(map[string]any{
		"maxOutputTokens":  500,
		"stopSequences":    []string{"END"},
		"temperature":      0.2,
		"seed":             42,
		"presencePenalty":  0.1,
		"frequencyPenalty": 0.1,
	}),
	ai.WithOutputType(struct {
		Summary string `json:"summary"`
	}{}),
)
```

`ai.WithOutputFormat(ai.OutputFormatJSON)` requests JSON-object mode. `ai.WithOutputType`
or `ai.WithOutputSchema` uses native JSON Schema when Genkit marks the model as supporting
constrained output; otherwise it uses JSON-object mode with Genkit's schema instructions.

To force a specific tool, provide its exact, case-sensitive name. The named tool must also
be included with `ai.WithTools`; otherwise generation returns a local configuration error.
The values `auto`, `required`, and `none` remain reserved as selection modes.

```go
response, err := genkit.Generate(ctx, g,
	ai.WithModel(gpt5Model),
	ai.WithTools(weatherTool, stockTool),
	ai.WithPrompt("What's the weather in London?"),
	ai.WithConfig(map[string]any{
		"toolChoice": "get_current_weather",
	}),
)
```

## Azure Setup and Authentication

### Getting Your Endpoint and API Key

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Go to "Keys and Endpoint" section
4. Copy your endpoint URL and API key

### Authentication Methods

The plugin supports multiple authentication methods to suit different deployment scenarios:

#### 1. API Key Authentication (Quick Start)

Best for: Development, testing, and simple scenarios

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
```

```go
import (
	"os"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

azurePlugin := &azureaifoundry.AzureAIFoundry{
	Endpoint: os.Getenv("AZURE_OPENAI_ENDPOINT"),
	APIKey:   os.Getenv("AZURE_OPENAI_API_KEY"),
}
```

#### 2. Azure Default Credential (Recommended for Production)

Best for: Production deployments, Azure-hosted applications

`DefaultAzureCredential` automatically tries multiple authentication methods in the following order:
1. **Environment variables** (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
2. **Managed Identity** (when deployed to Azure)
3. **Azure CLI** credentials (for local development)
4. **Azure PowerShell** credentials
5. **Interactive browser** authentication

```bash
# Required environment variables
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_TENANT_ID="your-tenant-id"

# Optional: For service principal authentication
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
```

```go
import (
	"fmt"
	"os"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

func main() {
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	tenantID := os.Getenv("AZURE_TENANT_ID")

	// Create DefaultAzureCredential
	credential, err := azidentity.NewDefaultAzureCredential(&azidentity.DefaultAzureCredentialOptions{
		TenantID: tenantID,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %s\n", err)
		return
	}

	// Initialize plugin with credential
	azurePlugin := &azureaifoundry.AzureAIFoundry{
		Endpoint:   endpoint,
		Credential: credential,
	}

	// Use the plugin with Genkit...
}
```

#### 3. Managed Identity (Azure Deployments)

Best for: Applications deployed to Azure (App Service, Container Apps, VMs, AKS)

When deployed to Azure, Managed Identity provides authentication without storing credentials:

```go
import (
	"os"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

func main() {
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")

	// Use Managed Identity
	credential, err := azidentity.NewManagedIdentityCredential(nil)
	if err != nil {
		panic(err)
	}

	azurePlugin := &azureaifoundry.AzureAIFoundry{
		Endpoint:   endpoint,
		Credential: credential,
	}
}
```

#### 4. Client Secret Credential (Service Principal)

Best for: CI/CD pipelines, automated deployments

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
```

```go
import (
	"os"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

func main() {
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	tenantID := os.Getenv("AZURE_TENANT_ID")
	clientID := os.Getenv("AZURE_CLIENT_ID")
	clientSecret := os.Getenv("AZURE_CLIENT_SECRET")

	credential, err := azidentity.NewClientSecretCredential(tenantID, clientID, clientSecret, nil)
	if err != nil {
		panic(err)
	}

	azurePlugin := &azureaifoundry.AzureAIFoundry{
		Endpoint:   endpoint,
		Credential: credential,
	}
}
```

#### 5. Azure CLI Credential (Local Development)

Best for: Local development with Azure CLI installed

```bash
# Login to Azure CLI first
az login

export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

```go
import (
	"os"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

func main() {
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")

	// Use Azure CLI credentials
	credential, err := azidentity.NewAzureCLICredential(nil)
	if err != nil {
		panic(err)
	}

	azurePlugin := &azureaifoundry.AzureAIFoundry{
		Endpoint:   endpoint,
		Credential: credential,
	}
}
```

### Model Deployments

Important: The `Name` in `ModelDefinition` should match your **deployment name** in Azure, not the model name. For example:

- If you deployed `gpt-5` with deployment name `my-gpt5-deployment`, use `"my-gpt5-deployment"`
- If you deployed `gpt-4o` with deployment name `gpt-4o`, use `"gpt-4o"`

Set `Type` using `ModelTypeChat`, `ModelTypeText`, `ModelTypeImage`,
`ModelTypeTextToSpeech`, or `ModelTypeSpeechToText`. An explicit type controls request
routing, which is useful when a custom deployment name contains a word such as `tts` or
`transcribe`. Leave `Type` empty to infer the type from the deployment name for backwards
compatibility.

`MaxTokens` sets the model's default maximum output-token count. A per-call
`maxOutputTokens` configuration value takes precedence when provided.

## Examples Directory

The repository includes comprehensive examples:

- **`examples/basic/`** - Simple text generation
- **`examples/streaming/`** - Real-time streaming responses
- **`examples/chat/`** - Multi-turn conversation with context
- **`examples/embeddings/`** - Text embeddings generation
- **`examples/tool_calling/`** - Function calling with multiple tools
- **`examples/vision/`** - Multimodal image analysis
- **`examples/image_generation/`** - Generate images
- **`examples/text_to_speech/`** - Convert text to speech
- **`examples/speech_to_text/`** - Transcribe audio to text

### Running Examples

```bash
# Set environment variables
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"

# Run basic example
cd examples/basic
go run main.go

# Run streaming example
cd ../streaming
go run main.go

# Run chat example
cd ../chat
go run main.go

# Run tool calling example
cd ../tool_calling
go run main.go

# Run vision example
cd ../vision
go run main.go

# Run image generation example
cd ../image_generation
go run main.go

# Run text-to-speech example
cd ../text_to_speech
go run main.go

# Run speech-to-text example (requires audio files)
cd ../speech_to_text
go run main.go
```

## Features in Detail

### 🔧 Tool Calling (Function Calling)

```go
// Define a tool
weatherTool := genkit.DefineTool(g, "get_weather",
	"Get current weather",
	func(ctx *ai.ToolContext, input struct {
		Location string `json:"location"`
		Unit     string `json:"unit,omitempty"`
	}) (string, error) {
		return getWeather(input.Location, input.Unit)
	},
)

// Use the tool
response, err := genkit.Generate(ctx, g,
	ai.WithModel(gpt4Model),
	ai.WithTools(weatherTool),
	ai.WithPrompt("What's the weather in San Francisco?"),
)
```

### 🖼️ Multimodal Support (Vision)

GPT-5 and GPT-4o support image inputs:

```go
response, err := genkit.Generate(ctx, g,
	ai.WithModel(gpt5Model),
	ai.WithMessages(ai.NewUserMessage(
		ai.NewTextPart("What's in this image?"),
		ai.NewMediaPart("image/jpeg", imageDataURL),
	)),
)
```

### 📡 Streaming

```go
streamCallback := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
	for _, part := range chunk.Content {
		switch {
		case part.IsReasoning():
			// Reasoning models (e.g. DeepSeek-R1, Kimi) stream their chain of thought
			fmt.Print(part.Text)
		case part.IsText():
			fmt.Print(part.Text)
		}
	}
	return nil
}

response, err := genkit.Generate(ctx, g,
	ai.WithModel(gpt4Model),
	ai.WithPrompt("Tell me a story"),
	ai.WithStreaming(streamCallback),
)
```

Streaming responses surface the same data as non-streaming ones:

- **Reasoning output** from reasoning models is emitted as reasoning parts
  (`part.IsReasoning()`) during streaming and accumulated onto the final message
  (`response.Reasoning()`).
- The **real finish reason** is reported (e.g. `content_filter`, `length`), so Azure
  content filtering or truncation is distinguishable from a clean stop.
- **Token usage** is populated on the final response, including `ThoughtsTokens` for
  reasoning models (`response.Usage`).

### 💬 Multi-turn Conversations

```go
// First message
response1, _ := genkit.Generate(ctx, g,
	ai.WithModel(gpt4Model),
	ai.WithMessages(
		ai.NewSystemMessage(ai.NewTextPart("You are a helpful assistant.")),
		ai.NewUserTextMessage("What is Azure?"),
	),
)

// Follow-up message with context
response2, _ := genkit.Generate(ctx, g,
	ai.WithModel(gpt4Model),
	ai.WithMessages(
		ai.NewSystemMessage(ai.NewTextPart("You are a helpful assistant.")),
		ai.NewUserTextMessage("What is Azure?"),
		response1.Message, // Previous assistant message
		ai.NewUserTextMessage("What are its key services?"),
	),
)
```

### 🔢 Embeddings

```go
import (
	"github.com/firebase/genkit/go/ai"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
)

// Define an embedder (use your deployment name)
embedder := azurePlugin.DefineEmbedder(g, "text-embedding-3-small")

// Or use common embedders helper
embedders := azureaifoundry.DefineCommonEmbedders(azurePlugin, g)

// Generate embeddings
response, err := genkit.Embed(ctx, g,
	ai.WithEmbedder(embedder),
	ai.WithEmbedText("Azure AI Foundry provides powerful AI capabilities"),
)

if err != nil {
	log.Fatal(err)
}

// Access the embedding vector
embedding := response.Embeddings[0].Embedding // []float32
log.Printf("Embedding dimensions: %d", len(embedding))
```

### 🎨 Image Generation

Generate images with DALL-E models using the standard `genkit.Generate()` method:

```go
// Define DALL-E model
dallE3 := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
	Name: azureaifoundry.ModelDallE3,
	Type: azureaifoundry.ModelTypeImage,
}, nil)

// Generate image
response, err := genkit.Generate(ctx, g,
	ai.WithModel(dallE3),
	ai.WithPrompt("A serene landscape with mountains at sunset"),
	ai.WithConfig(map[string]interface{}{
		"quality": "hd",
		"size":    "1024x1024",
		"style":   "vivid",
	}),
)

if err != nil {
	log.Fatal(err)
}

log.Printf("Image URL: %s", response.Text())
```

### 🗣️ Text-to-Speech

Convert text to speech using the standard `genkit.Generate()` method:

```go
import "encoding/base64"

// Define TTS model
ttsModel := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
	Name: azureaifoundry.ModelTTS1HD,
	Type: azureaifoundry.ModelTypeTextToSpeech,
}, nil)

// Generate speech
response, err := genkit.Generate(ctx, g,
	ai.WithModel(ttsModel),
	ai.WithPrompt("Hello! Welcome to Azure AI Foundry."),
	ai.WithConfig(map[string]interface{}{
		"voice":           "nova",
		"response_format": "mp3",
		"speed":           1.5,
	}),
)

if err != nil {
	log.Fatal(err)
}

// Decode base64 audio and save file
audioData, _ := base64.StdEncoding.DecodeString(response.Text())
os.WriteFile("output.mp3", audioData, 0644)
```

### 🎙️ Speech-to-Text

Transcribe audio to text using the standard `genkit.Generate()` method:

```go
import "encoding/base64"

// Define Whisper model with media support (required for audio input)
whisperModel := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
	Name:          azureaifoundry.ModelWhisper1,
	Type:          azureaifoundry.ModelTypeSpeechToText,
	SupportsMedia: true, // Required for media parts (audio)
}, nil)

// Read and encode audio file
audioData, _ := os.ReadFile("audio.mp3")
base64Audio := base64.StdEncoding.EncodeToString(audioData)

// Transcribe audio
response, err := genkit.Generate(ctx, g,
	ai.WithModel(whisperModel),
	ai.WithMessages(ai.NewUserMessage(
		ai.NewMediaPart("audio/mp3", "data:audio/mp3;base64,"+base64Audio),
	)),
	ai.WithConfig(map[string]interface{}{
		"language": "en",
	}),
)

if err != nil {
	log.Fatal(err)
}

log.Printf("Transcription: %s", response.Text())
```

## Troubleshooting

### Common Issues

1. **"Endpoint is required" Error**
   - Verify `AZURE_OPENAI_ENDPOINT` is set correctly
   - Ensure the endpoint URL includes `https://` and trailing `/`

2. **"Deployment not found" Error**
   - Check that the deployment name in your code matches the actual deployment name in Azure
   - Verify the model is deployed in your Azure OpenAI resource

3. **Authentication Errors**
   - Ensure your API key is correct
   - Check that your Azure subscription is active
   - Verify network connectivity to Azure

4. **Rate Limit Errors**
   - Implement exponential backoff retry logic
   - Consider upgrading to higher rate limits
   - Distribute requests across time

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow [Conventional Commits](https://conventionalcommits.org/) format
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

Apache 2.0 - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Genkit team for the excellent Go framework
- Azure AI team for the comprehensive AI platform
- The open source community for inspiration and feedback

---

**Built with ❤️ for the Genkit Go community**
