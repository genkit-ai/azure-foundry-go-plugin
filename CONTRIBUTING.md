# Contributing to Azure AI Foundry Plugin for Genkit Go

Thank you for your interest in contributing to the Azure AI Foundry Plugin for Genkit Go! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and inclusive environment. By participating, you are expected to uphold this standard.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Go 1.26 or later
- Git
- Azure OpenAI access (for testing)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/genkit-ai/azure-foundry-go-plugin.git
   cd genkit-azure-foundry-go
   ```

2. Install dependencies:
   ```bash
   go mod download
   ```

3. Set up your Azure credentials for testing:
   ```bash
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_OPENAI_API_KEY="your-api-key"
   ```

4. Run examples to verify setup:
   ```bash
   cd examples/basic
   go run main.go
   ```

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Describe the bug in detail
- Include steps to reproduce
- Provide your environment details

### Suggesting Enhancements

- Use the GitHub issue tracker
- Clearly describe the enhancement
- Explain why it would be useful
- Provide examples if possible

### Pull Requests

1. Create a new branch for your changes
2. Make your changes following our coding standards
3. Add or update tests as needed
4. Update documentation
5. Submit a pull request

## Pull Request Process

1. **Update Documentation**: Ensure the README.md and other documentation are updated
2. **Add Tests**: Include tests that cover your changes
3. **Follow Commit Standards**: Use conventional commit format
4. **Create Pull Request**: Submit a pull request with a clear title and description
5. **Address Feedback**: Respond to review comments promptly

## Coding Standards

### Go Code Style

- Follow standard Go formatting (`go fmt`)
- Use meaningful variable and function names
- Include comments for exported functions and types
- Follow Go naming conventions
- Keep functions focused and reasonably sized

### Code Organization

- Group related functionality together
- Use clear package structure
- Separate concerns appropriately
- Follow existing patterns in the codebase

### Error Handling

- Use Go's standard error handling patterns
- Provide meaningful error messages
- Handle errors at appropriate levels
- Don't ignore errors unless explicitly justified

## Testing

### Running Tests

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run tests with race detection
go test -race ./...
```

### Live Tests

Live tests in `live_test.go` call real Azure deployments. They never run as
part of the default `go test ./...`: each one is skipped unless
`AZURE_LIVE_TEST=1` is set and the deployment it exercises is configured.

| Variable | Exercises |
|----------|-----------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource used by the tests below (required) |
| `AZURE_OPENAI_API_KEY` | Optional; `DefaultAzureCredential` is used when unset |
| `AZURE_OPENAI_API_VERSION` | Optional API version |
| `AZURE_LIVE_CHAT_DEPLOYMENT` | Chat sync, streaming, tool calls, and structured output |
| `AZURE_LIVE_EMBEDDING_DEPLOYMENT` | Embeddings |
| `AZURE_LIVE_IMAGE_DEPLOYMENT` | Image generation |
| `AZURE_LIVE_TTS_DEPLOYMENT` | Text-to-speech |
| `AZURE_LIVE_STT_DEPLOYMENT` | Speech-to-text |
| `AZURE_FOUNDRY_ENDPOINT` | Microsoft Foundry resource for the Foundry Models test |
| `AZURE_FOUNDRY_API_KEY` | Optional; `DefaultAzureCredential` is used when unset |
| `AZURE_LIVE_FOUNDRY_DEPLOYMENT` | Foundry Models chat through the OpenAI v1 route |

```bash
export AZURE_LIVE_TEST=1
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_LIVE_CHAT_DEPLOYMENT="gpt-4o"
export AZURE_LIVE_EMBEDDING_DEPLOYMENT="text-embedding-3-small"

# Run every configured live test
go test -run TestLive -v .

# Run a single live test
go test -run TestLiveChatStreaming -v .
```

### Test Requirements

- All new code should include appropriate tests
- Tests should cover both success and error cases
- Integration tests should be included for Azure OpenAI interactions
- Use table-driven tests where appropriate

## Documentation

### Code Documentation

- All exported functions and types must have comments
- Comments should explain what the code does, not how
- Include examples in comments where helpful
- Use standard Go documentation conventions

### README Updates

- Keep the README.md up to date with new features
- Include examples for new functionality
- Update supported models list when adding new models
- Maintain accurate installation and usage instructions

## Questions?

If you have questions about contributing, please:

1. Check existing issues and discussions
2. Create a new issue with the `question` label
3. Reach out to maintainers

Thank you for contributing to Azure AI Foundry Plugin for Genkit Go!
