# Z-Image Turbo API - Code Examples

Ready-to-use integration examples in various programming languages.

## Table of Contents

- [JavaScript/TypeScript](#javascripttypescript)
- [Python](#python)
- [cURL](#curl)
- [Ruby](#ruby)
- [Go](#go)
- [PHP](#php)

---

## JavaScript/TypeScript

### Basic Image Generation

```typescript
// TypeScript example with full type safety

interface GenerateRequest {
  prompt: string;
  negative_prompt?: string;
  style_preset?: string;
  generator?: "zimage" | "gemini";
  generator_model?: string;
  advanced?: {
    width?: number;
    height?: number;
    steps?: number;
    cfg_scale?: number;
    seed?: number | null;
    use_lora?: boolean;
    lora_id?: string;
    lora_scale?: number;
  };
  stealth?: boolean;
  recipe_id?: string;
  auto_enhance?: boolean;
}

interface GenerationRecord {
  id: string;
  image_url: string;
  prompt: string;
  created_at: string;
  width?: number;
  height?: number;
  duration_sec?: number;
}

class ZImageClient {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async generate(request: GenerateRequest): Promise<GenerationRecord> {
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: request.prompt,
        generator: request.generator || "zimage",
        style_preset: request.style_preset || "None",
        advanced: {
          width: request.advanced?.width || 768,
          height: request.advanced?.height || 768,
          steps: request.advanced?.steps || 7,
          cfg_scale: request.advanced?.cfg_scale || 0.0,
          seed: request.advanced?.seed || null,
        },
        stealth: request.stealth || false,
        recipe_id: request.recipe_id,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`HTTP ${response.status}: ${error.detail}`);
    }

    return response.json();
  }

  getImageUrl(record: GenerationRecord): string {
    return `${this.baseUrl}${record.image_url}`;
  }
}

// Usage
const client = new ZImageClient();

async function main() {
  try {
    const result = await client.generate({
      prompt: "A serene Japanese garden with cherry blossoms",
      style_preset: "Cinematic",
      advanced: {
        width: 1024,
        height: 1024,
        steps: 7,
      },
    });

    console.log(`Generated in ${result.duration_sec}s`);
    console.log(`Image: ${client.getImageUrl(result)}`);

    // Display in browser
    const img = document.createElement("img");
    img.src = client.getImageUrl(result);
    document.body.appendChild(img);

  } catch (error) {
    console.error("Generation failed:", error);
  }
}

main();
```

---

### React Hook Example

```typescript
import { useState, useCallback } from 'react';

interface UseZImageGenerateOptions {
  onSuccess?: (imageUrl: string) => void;
  onError?: (error: Error) => void;
}

export function useZImageGenerate(options: UseZImageGenerateOptions = {}) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [result, setResult] = useState<{
    imageUrl: string;
    duration: number;
  } | null>(null);

  const generate = useCallback(async (prompt: string, config = {}) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          generator: config.generator || 'zimage',
          style_preset: config.stylePreset || 'Cinematic',
          advanced: {
            width: config.width || 768,
            height: config.height || 768,
            steps: config.steps || 7,
          },
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail);
      }

      const data = await response.json();
      const imageUrl = `http://localhost:8000${data.image_url}`;

      setResult({
        imageUrl,
        duration: data.duration_sec || 0,
      });

      options.onSuccess?.(imageUrl);

    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      options.onError?.(error);
    } finally {
      setIsLoading(false);
    }
  }, [options]);

  return { generate, isLoading, error, result };
}

// Component usage
function ImageGenerator() {
  const { generate, isLoading, error, result } = useZImageGenerate({
    onSuccess: (imageUrl) => console.log('Generated!', imageUrl),
    onError: (error) => console.error('Error:', error),
  });

  return (
    <div>
      <button
        onClick={() => generate('A futuristic cityscape at sunset')}
        disabled={isLoading}
      >
        {isLoading ? 'Generating...' : 'Generate Image'}
      </button>

      {error && <p className="error">{error.message}</p>}

      {result && (
        <div>
          <img src={result.imageUrl} alt="Generated" />
          <p>Generated in {result.duration.toFixed(2)}s</p>
        </div>
      )}
    </div>
  );
}
```

---

### Node.js with Download

```javascript
const fs = require('fs');
const path = require('path');

async function generateAndDownload(prompt, outputPath) {
  try {
    // Generate the image
    const response = await fetch('http://localhost:8000/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        generator: 'zimage',
        advanced: { width: 1024, height: 1024, steps: 7 },
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail);
    }

    const result = await response.json();

    // Download the image
    const imageUrl = `http://localhost:8000${result.image_url}`;
    const imageResponse = await fetch(imageUrl);
    const buffer = await imageResponse.arrayBuffer();

    // Save to file
    fs.writeFileSync(outputPath, Buffer.from(buffer));

    console.log(`Image saved to: ${outputPath}`);
    console.log(`Generation time: ${result.duration_sec}s`);

    return result;

  } catch (error) {
    console.error('Error:', error.message);
    throw error;
  }
}

// Usage
generateAndDownload(
  'A mystical forest with glowing mushrooms',
  './mushroom-forest.png'
);
```

---

### Telegram Bot Integration

```javascript
const TelegramBot = require('node-telegram-bot-api');
const fs = require('fs');

const token = process.env.TELEGRAM_BOT_TOKEN;
const bot = new TelegramBot(token, { polling: true });

bot.onText(/\/generate (.+)/, async (msg, match) => {
  const chatId = msg.chat.id;
  const prompt = match[1];

  // Send "generating" message
  const statusMsg = await bot.sendMessage(chatId, '🎨 Generating your image...');

  try {
    // Generate image (ephemeral - no disk save)
    const response = await fetch('http://localhost:8000/api/generate_ephemeral', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        advanced: { width: 1024, height: 1024, steps: 7 },
      }),
    });

    if (!response.ok) {
      throw new Error('Generation failed');
    }

    // Get image buffer
    const buffer = await response.arrayBuffer();

    // Send photo to Telegram
    await bot.sendPhoto(chatId, Buffer.from(buffer), {
      caption: `✨ ${prompt}`,
      reply_to_message_id: msg.message_id,
    });

    // Delete status message
    await bot.deleteMessage(chatId, statusMsg.message_id);

  } catch (error) {
    await bot.sendMessage(chatId, `❌ Error: ${error.message}`);
  }
});
```

---

## Python

### Synchronous Example

```python
import requests
import time
from pathlib import Path
from typing import Optional

class ZImageClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        width: int = 768,
        height: int = 768,
        steps: int = 7,
        style_preset: str = "None",
        generator: str = "zimage",
        seed: Optional[int] = None,
    ) -> dict:
        """Generate an image from a text prompt."""

        url = f"{self.base_url}/api/generate"
        payload = {
            "prompt": prompt,
            "generator": generator,
            "style_preset": style_preset,
            "advanced": {
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
            },
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        return result

    def download_image(self, image_url: str, output_path: str) -> None:
        """Download a generated image to a file."""
        full_url = f"{self.base_url}{image_url}"
        response = requests.get(full_url)
        response.raise_for_status()

        Path(output_path).write_bytes(response.content)
        print(f"✅ Downloaded to: {output_path}")

# Usage
def main():
    client = ZImageClient()

    print("🎨 Generating image...")
    start = time.time()

    result = client.generate(
        prompt="A steampunk city with flying machines and brass gears",
        width=1024,
        height=1024,
        style_preset="Cinematic",
    )

    duration = time.time() - start
    print(f"✅ Generated in {result['duration_sec']:.2f}s")

    # Download the image
    filename = f"output_{result['id']}.png"
    client.download_image(result['image_url'], filename)

if __name__ == "__main__":
    main()
```

---

### Async with `aiohttp`

```python
import aiohttp
import asyncio
from pathlib import Path

class AsyncZImageClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def generate(self, prompt: str, **kwargs) -> dict:
        """Async image generation."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "prompt": prompt,
            "generator": kwargs.get("generator", "zimage"),
            "style_preset": kwargs.get("style_preset", "None"),
            "advanced": {
                "width": kwargs.get("width", 768),
                "height": kwargs.get("height", 768),
                "steps": kwargs.get("steps", 7),
                "seed": kwargs.get("seed"),
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()

    async def download_image(self, image_url: str, output_path: str) -> None:
        """Async image download."""
        full_url = f"{self.base_url}{image_url}"

        async with aiohttp.ClientSession() as session:
            async with session.get(full_url) as response:
                response.raise_for_status()
                content = await response.read()

        Path(output_path).write_bytes(content)
        print(f"✅ Downloaded to: {output_path}")

async def batch_generate(prompts: list[str]) -> None:
    """Generate multiple images concurrently."""
    client = AsyncZImageClient()

    tasks = []
    for i, prompt in enumerate(prompts):
        task = client.generate(prompt, width=1024, height=1024)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        filename = f"batch_{i}_{result['id']}.png"
        await client.download_image(result['image_url'], filename)
        print(f"✅ Image {i+1}: {result['duration_sec']:.2f}s")

# Usage
async def main():
    prompts = [
        "A cosmic nebula with vibrant colors",
        "An underwater city with bioluminescent creatures",
        "A medieval castle on a floating island",
    ]

    await batch_generate(prompts)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Flask Web Integration

```python
from flask import Flask, request, jsonify, send_file
import requests
import io

app = Flask(__name__)
ZIMAGE_API = "http://localhost:8000"

@app.route('/api/generate', methods=['POST'])
def proxy_generate():
    """Proxy endpoint to generate image."""
    data = request.json

    try:
        # Forward request to Z-Image
        response = requests.post(
            f"{ZIMAGE_API}/api/generate",
            json=data,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()

        # Return full image URL for frontend
        result['image_url'] = f"{ZIMAGE_API}{result['image_url']}"

        return jsonify(result)

    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 500

@app.route('/image/<image_id>')
def serve_image(image_id: str):
    """Fetch and serve image from Z-Image."""
    try:
        # Get image from Z-Image
        response = requests.get(
            f"{ZIMAGE_API}/api/download/{image_id}",
            timeout=10
        )
        response.raise_for_status()

        # Serve to client
        return send_file(
            io.BytesIO(response.content),
            mimetype='image/png',
            as_attachment=False,
            download_name=f"{image_id}.png"
        )

    except requests.RequestException as e:
        return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    app.run(port=5000, debug=True)
```

---

## cURL

### Basic Generation

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at golden hour",
    "generator": "zimage",
    "style_preset": "Cinematic",
    "advanced": {
      "width": 1024,
      "height": 1024,
      "steps": 7
    }
  }'
```

### Download Result

```bash
# Generate and save to file
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A red rose", "advanced": {"width": 512, "height": 512}}' \
  | jq -r '.image_url' \
  | xargs -I {} curl http://localhost:8000{} --output rose.png
```

### Ephemeral Generation (Direct Download)

```bash
curl -X POST http://localhost:8000/api/generate_ephemeral \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Abstract geometric art", "advanced": {"width": 512, "height": 512}}' \
  --output abstract.png
```

### Get Recipes

```bash
# Get all recipes
curl http://localhost:8000/api/recipes | jq '.recipes[] | {title, prompt}'

# Get specific group
curl "http://localhost:8000/api/recipes?group=characters" | jq '.recipes[].title'

# Search recipes
curl "http://localhost:8000/api/recipes?q=portrait" | jq '.recipes[] | {id, title}'
```

### Health Check

```bash
curl http://localhost:8000/health | jq '.'
```

---

## Ruby

### Basic Example

```ruby
require 'net/http'
require 'json'
require 'uri'

class ZImageClient
  def initialize(base_url = 'http://localhost:8000')
    @base_url = base_url
    @uri = URI(base_url)
  end

  def generate(prompt, options = {})
    url = URI.join(@base_url, '/api/generate')

    payload = {
      prompt: prompt,
      generator: options[:generator] || 'zimage',
      style_preset: options[:style_preset] || 'None',
      advanced: {
        width: options[:width] || 768,
        height: options[:height] || 768,
        steps: options[:steps] || 7,
        seed: options[:seed]
      }
    }

    http = Net::HTTP.new(url.host, url.port)
    request = Net::HTTP::Post.new(url)
    request['Content-Type'] = 'application/json'
    request.body = payload.to_json

    response = http.request(request)

    unless response.is_a?(Net::HTTPSuccess)
      error = JSON.parse(response.body)
      raise StandardError, "Error #{response.code}: #{error['detail']}"
    end

    result = JSON.parse(response.body)
    return result
  end

  def download_image(image_url, output_path)
    full_url = URI.join(@base_url, image_url)

    http = Net::HTTP.new(full_url.host, full_url.port)
    request = Net::HTTP::Get.new(full_url)

    response = http.request(request)
    File.write(output_path, response.body)

    puts "✅ Downloaded to: #{output_path}"
  end
end

# Usage
client = ZImageClient.new

puts "🎨 Generating..."
result = client.generate(
  "A mystical forest with fireflies",
  width: 1024,
  height: 1024,
  style_preset: "Fantasy Art"
)

puts "✅ Generated in #{result['duration_sec']}s"
client.download_image(result['image_url'], "forest.png")
```

---

### Rails Integration

```ruby
# app/services/z_image_service.rb
class ZImageService
  BASE_URL = ENV['ZIMAGE_URL'] || 'http://localhost:8000'

  def self.generate(prompt, options = {})
    uri = URI("#{BASE_URL}/api/generate")

    payload = {
      prompt: prompt,
      generator: options[:generator] || 'zimage',
      style_preset: options[:style_preset] || 'None',
      advanced: {
        width: options[:width] || 768,
        height: options[:height] || 768,
        steps: options[:steps] || 7
      }
    }

    response = HTTParty.post(uri, body: payload.to_json, headers: { 'Content-Type' => 'application/json' })

    unless response.success?
      error = JSON.parse(response.body)
      raise StandardError, "Z-Image error: #{error['detail']}"
    end

    JSON.parse(response.body)
  end

  def self.image_url(image_path)
    "#{BASE_URL}#{image_path}"
  end
end

# app/controllers/images_controller.rb
class ImagesController < ApplicationController
  def create
    result = ZImageService.generate(
      params[:prompt],
      width: params[:width]&.to_i,
      height: params[:height]&.to_i,
      style_preset: params[:style]
    )

    render json: {
      id: result['id'],
      url: ZImageService.image_url(result['image_url']),
      duration: result['duration_sec']
    }
  rescue => e
    render json: { error: e.message }, status: :internal_server_error
  end
end
```

---

## Go

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

const (
	BaseURL = "http://localhost:8000"
)

type GenerateRequest struct {
	Prompt      string   `json:"prompt"`
	Generator   string   `json:"generator"`
	StylePreset string   `json:"style_preset"`
	Advanced    Advanced `json:"advanced"`
}

type Advanced struct {
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Steps  int    `json:"steps"`
	Seed   *int   `json:"seed"`
}

type GenerateResponse struct {
	ID          string  `json:"id"`
	ImageURL    string  `json:"image_url"`
	Prompt      string  `json:"prompt"`
	DurationSec float64 `json:"duration_sec"`
	Width       int     `json:"width"`
	Height      int     `json:"height"`
}

func GenerateImage(prompt string) (*GenerateResponse, error) {
	req := GenerateRequest{
		Prompt:      prompt,
		Generator:   "zimage",
		StylePreset: "Cinematic",
		Advanced: Advanced{
			Width:  1024,
			Height: 1024,
			Steps:  7,
		},
	}

	payload, _ := json.Marshal(req)
	resp, err := http.Post(BaseURL+"/api/generate", "application/json", bytes.NewBuffer(payload))
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: %s", string(body))
	}

	var result GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode failed: %w", err)
	}

	return &result, nil
}

func DownloadImage(imageURL, filename string) error {
	url := BaseURL + imageURL
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %d", resp.StatusCode)
	}

	out, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func main() {
	result, err := GenerateImage("A cosmic nebula with vibrant purple and blue colors")
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		return
	}

	fmt.Printf("✅ Generated in %.2fs\n", result.DurationSec)
	fmt.Printf("   Image URL: %s\n", result.ImageURL)

	filename := fmt.Sprintf("output_%s.png", result.ID)
	if err := DownloadImage(result.ImageURL, filename); err != nil {
		fmt.Printf("❌ Download failed: %v\n", err)
		return
	}

	fmt.Printf("✅ Saved to: %s\n", filename)
}
```

---

## PHP

### Basic Example

```php
<?php

class ZImageClient {
    private string $baseUrl;

    public function __construct(string $baseUrl = 'http://localhost:8000') {
        $this->baseUrl = $baseUrl;
    }

    public function generate(string $prompt, array $options = []): array {
        $payload = [
            'prompt' => $prompt,
            'generator' => $options['generator'] ?? 'zimage',
            'style_preset' => $options['style_preset'] ?? 'None',
            'advanced' => [
                'width' => $options['width'] ?? 768,
                'height' => $options['height'] ?? 768,
                'steps' => $options['steps'] ?? 7,
                'seed' => $options['seed'] ?? null,
            ],
        ];

        $ch = curl_init($this->baseUrl . '/api/generate');
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
        curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);

        $response = curl_exec($ch);
        $statusCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($statusCode !== 200) {
            $error = json_decode($response, true);
            throw new Exception("Error {$statusCode}: {$error['detail']}");
        }

        return json_decode($response, true);
    }

    public function downloadImage(string $imageUrl, string $outputPath): void {
        $fullUrl = $this->baseUrl . $imageUrl;
        $imageData = file_get_contents($fullUrl);

        if ($imageData === false) {
            throw new Exception("Failed to download image");
        }

        file_put_contents($outputPath, $imageData);
        echo "✅ Downloaded to: {$outputPath}\n";
    }
}

// Usage
try {
    $client = new ZImageClient();

    echo "🎨 Generating...\n";
    $result = $client->generate(
        'A mystical enchanted forest with glowing mushrooms',
        [
            'width' => 1024,
            'height' => 1024,
            'style_preset' => 'Fantasy Art',
        ]
    );

    echo "✅ Generated in {$result['duration_sec']}s\n";

    $filename = "output_{$result['id']}.png";
    $client->downloadImage($result['image_url'], $filename);

} catch (Exception $e) {
    echo "❌ Error: {$e->getMessage()}\n";
}
```

---

### Laravel Integration

```php
<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;

class ZImageService
{
    protected string $baseUrl;

    public function __construct()
    {
        $this->baseUrl = config('services.zimage.url', 'http://localhost:8000');
    }

    public function generate(string $prompt, array $options = []): array
    {
        $payload = [
            'prompt' => $prompt,
            'generator' => $options['generator'] ?? 'zimage',
            'style_preset' => $options['style_preset'] ?? 'Cinematic',
            'advanced' => [
                'width' => $options['width'] ?? 768,
                'height' => $options['height'] ?? 768,
                'steps' => $options['steps'] ?? 7,
            ],
        ];

        $response = Http::timeout(30)->post("{$this->baseUrl}/api/generate", $payload);

        if (! $response->successful()) {
            $error = $response->json();
            throw new \Exception("Z-Image error: {$error['detail']}");
        }

        return $response->json();
    }

    public function downloadAndSave(string $imageUrl, string $disk = 'public'): string
    {
        $fullUrl = $this->baseUrl . $imageUrl;
        $imageContents = file_get_contents($fullUrl);

        $filename = 'zimage_' . basename($imageUrl);
        $path = 'generated/' . $filename;

        Storage::disk($disk)->put($path, $imageContents);

        return Storage::disk($disk)->url($path);
    }
}

// app/Http/Controllers/GenerateController.php

namespace App\Http\Controllers;

use App\Services\ZImageService;
use Illuminate\Http\Request;

class GenerateController extends Controller
{
    public function __invoke(Request $request, ZImageService $zimage)
    {
        $request->validate([
            'prompt' => 'required|string|max:1000',
            'width' => 'integer|between:256,1536',
            'height' => 'integer|between:256,1536',
            'style' => 'string|in:None,Cinematic,Anime,Photographic',
        ]);

        try {
            $result = $zimage->generate(
                $request->prompt,
                [
                    'width' => $request->width ?? 768,
                    'height' => $request->height ?? 768,
                    'style_preset' => $request->style ?? 'Cinematic',
                ]
            );

            $url = $zimage->downloadAndSave($result['image_url']);

            return response()->json([
                'success' => true,
                'image_url' => $url,
                'duration' => $result['duration_sec'],
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage(),
            ], 500);
        }
    }
}
```

---

## Advanced Patterns

### Batch Generation with Concurrency

```javascript
// Generate multiple images in parallel
async function batchGenerate(prompts) {
  const client = new ZImageClient();

  const promises = prompts.map(prompt =>
    client.generate({
      prompt,
      advanced: { width: 1024, height: 1024, steps: 7 }
    })
  );

  const results = await Promise.all(promises);

  return results.map((result, i) => ({
    prompt: prompts[i],
    imageUrl: client.getImageUrl(result),
    duration: result.duration_sec,
  }));
}

// Usage
const prompts = [
  "A cyberpunk street vendor",
  "A steampunk airship in flight",
  "A fantasy dragon hoard",
];

const images = await batchGenerate(prompts);
console.log(`Generated ${images.length} images in parallel`);
```

---

### Retry Logic

```javascript
async function generateWithRetry(prompt, maxRetries = 3) {
  const client = new ZImageClient();

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await client.generate({ prompt });
    } catch (error) {
      if (attempt === maxRetries) {
        throw new Error(`Failed after ${maxRetries} attempts: ${error.message}`);
      }

      console.warn(`Attempt ${attempt} failed, retrying...`);
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
    }
  }
}
```

---

### Progress Tracking

```javascript
class ZImageClientWithProgress extends ZImageClient {
  async generateWithProgress(request, onProgress) {
    const startTime = Date.now();

    // Simulate progress (actual API doesn't support progress callbacks)
    const progressInterval = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      const estimated = 5; // Average 5 seconds
      const percent = Math.min((elapsed / estimated) * 100, 90);
      onProgress(percent);
    }, 100);

    try {
      const result = await super.generate(request);
      clearInterval(progressInterval);
      onProgress(100);
      return result;
    } catch (error) {
      clearInterval(progressInterval);
      throw error;
    }
  }
}

// Usage
const client = new ZImageClientWithProgress();

await client.generateWithProgress(
  { prompt: "A beautiful landscape" },
  (percent) => {
    updateProgressBar(percent);
  }
);
```

---

## Testing

### Jest Tests

```typescript
import { ZImageClient } from './zimage-client';

describe('ZImageClient', () => {
  let client: ZImageClient;

  beforeAll(() => {
    client = new ZImageClient('http://localhost:8000');
  });

  test('should generate an image', async () => {
    const result = await client.generate({
      prompt: 'Test prompt',
      advanced: { width: 512, height: 512, steps: 4 },
    });

    expect(result).toHaveProperty('id');
    expect(result).toHaveProperty('image_url');
    expect(result.width).toBe(512);
    expect(result.height).toBe(512);
  });

  test('should throw on empty prompt', async () => {
    await expect(
      client.generate({ prompt: '' })
    ).rejects.toThrow();
  });
});
```

---

## Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  zimage:
    build: ./path/to/zimage
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./generated:/app/generated

  your-app:
    build: ./your-app
    ports:
      - "3000:3000"
    environment:
      - ZIMAGE_URL=http://zimage:8000
    depends_on:
      - zimage
```

---

## Best Practices Summary

✅ **DO:**
- Use the recipe system for consistent prompts
- Cache recipe data locally
- Implement proper error handling
- Show loading states during generation
- Use ephemeral generation for bots
- Handle network timeouts gracefully

❌ **DON'T:**
- Call the API synchronously on the main thread
- Assume the image URL is absolute (it's relative)
- Forget to handle 422 validation errors
- Expose your API keys in client-side code
- Send empty prompts
- Exceed resolution limits (max 1536px)
