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

// Package main demonstrates tool calling (function calling) with Azure AI Foundry
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	azureaifoundry "github.com/xavidop/genkit-azure-foundry-go"
	"github.com/xavidop/genkit-azure-foundry-go/examples/common"
)

// WeatherResult represents the weather information
type WeatherResult struct {
	Location string `json:"location"`
	Weather  string `json:"weather"`
}

// StockResult represents stock price information
type StockResult struct {
	Symbol string `json:"symbol"`
	Price  string `json:"price"`
}

// Mock function to get weather
func getCurrentWeather(location string, unit string) (*WeatherResult, error) {
	// In a real application, this would call a weather API
	// Note: unit parameter is used for demonstration; in production you'd use it to format output
	if unit == "" {
		unit = "fahrenheit"
	}
	_ = unit // unit would be used in real API calls

	weatherData := map[string]string{
		"San Francisco, CA": "72°F (22°C), Partly Cloudy",
		"New York, NY":      "65°F (18°C), Sunny",
		"Seattle, WA":       "58°F (14°C), Rainy",
		"London, UK":        "55°F (13°C), Foggy",
		"Tokyo, JP":         "70°F (21°C), Clear",
	}

	weather, exists := weatherData[location]
	if !exists {
		return &WeatherResult{
			Location: location,
			Weather:  fmt.Sprintf("Weather data not available for %s", location),
		}, nil
	}

	return &WeatherResult{
		Location: location,
		Weather:  weather,
	}, nil
}

// Mock function to get stock price
func getStockPrice(symbol string) (*StockResult, error) {
	// In a real application, this would call a stock market API
	stockData := map[string]string{
		"AAPL":  "$182.50 (+1.2%)",
		"GOOGL": "$139.75 (+0.8%)",
		"MSFT":  "$378.91 (+1.5%)",
		"AMZN":  "$145.32 (-0.3%)",
	}

	price, exists := stockData[symbol]
	if !exists {
		return &StockResult{
			Symbol: symbol,
			Price:  fmt.Sprintf("Stock price not available for %s", symbol),
		}, nil
	}

	return &StockResult{
		Symbol: symbol,
		Price:  price,
	}, nil
}

func main() {
	ctx := context.Background()

	// Setup Genkit with Azure AI Foundry
	g, azurePlugin, err := common.SetupGenkit(ctx, nil)
	if err != nil {
		log.Fatalf("Failed to setup Genkit: %v", err)
	}

	log.Println("Starting tool calling example...")

	// Define GPT-5 model (use your deployment name)
	gpt5Model := azurePlugin.DefineModel(g, azureaifoundry.ModelDefinition{
		Name: "gpt-5", // Replace with your actual deployment name
		Type: "chat",
	}, nil)

	// Define weather tool
	weatherTool := genkit.DefineTool(g, "get_current_weather",
		"Get current weather information",
		func(ctx *ai.ToolContext, input struct {
			Location string `json:"location" jsonschema:"description=The city and state, e.g. San Francisco, CA"`
			Unit     string `json:"unit,omitempty" jsonschema:"description=The temperature unit (celsius or fahrenheit),enum=celsius,enum=fahrenheit"`
		}) (*WeatherResult, error) {
			return getCurrentWeather(input.Location, input.Unit)
		},
	)

	// Define stock price tool
	stockTool := genkit.DefineTool(g, "get_stock_price",
		"Get current stock price for a given ticker symbol",
		func(ctx *ai.ToolContext, input struct {
			Symbol string `json:"symbol" jsonschema:"description=The stock ticker symbol, e.g. AAPL for Apple Inc."`
		}) (*StockResult, error) {
			return getStockPrice(input.Symbol)
		},
	)

	// Test prompts that require tool usage
	prompts := []string{
		"What's the weather like in San Francisco?",
		"Can you check the current stock price for Microsoft (MSFT)?",
		"I'm planning a trip to Tokyo. What's the weather there and what's Apple's stock price?",
	}

	for i, prompt := range prompts {
		log.Printf("\n--- Test %d ---\n", i+1)
		log.Printf("Prompt: %s\n", prompt)

		response, err := genkit.Generate(ctx, g,
			ai.WithModel(gpt5Model),
			ai.WithTools(weatherTool, stockTool),
			ai.WithMessages(ai.NewUserTextMessage(prompt)),
			ai.WithConfig(map[string]interface{}{
				"temperature":     0.1, // Lower temperature for more consistent tool usage
				"maxOutputTokens": 1000,
			}),
		)

		if err != nil {
			log.Printf("Error: %v", err)
			continue
		}

		log.Printf("Response: %s", response.Text())

		// Log if the model used any tools
		if response.Message != nil && len(response.Message.Content) > 0 {
			hasToolUse := false
			for _, part := range response.Message.Content {
				if part.IsToolRequest() {
					hasToolUse = true
					break
				}
			}
			if hasToolUse {
				log.Printf("✓ Model used tools in this response")
			} else {
				log.Printf("○ Model responded without using tools")
			}
		}
	}

	log.Println("\nTool calling example completed")
}
