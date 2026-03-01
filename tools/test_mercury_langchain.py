#!/usr/bin/env python3
"""
Test if LangChain's ChatOpenAI actually passes model_kwargs to the API.
"""
import os

INCEPTION_API_KEY = os.getenv("INCEPTION_API_KEY")
if not INCEPTION_API_KEY:
    raise ValueError("INCEPTION_API_KEY environment variable not set")

from langchain_openai import ChatOpenAI
import json

print("=" * 70)
print("Testing if ChatOpenAI passes model_kwargs to the API")
print("=" * 70)

# Create the LLM exactly as we do in youtube_summarizer.py
llm = ChatOpenAI(
    model="mercury-2",
    api_key=INCEPTION_API_KEY,
    base_url="https://api.inceptionlabs.ai/v1",
    model_kwargs={"reasoning_effort": "instant"}
)

print(f"\nChatOpenAI configuration:")
print(f"  model: {llm.model}")
print(f"  model_kwargs: {llm.model_kwargs}")

# Check if there's extra_body or other mechanism
if hasattr(llm, 'init_kwargs'):
    print(f"  init_kwargs: {llm.init_kwargs}")

# Monkey patch to intercept the actual HTTP request
import httpx
from unittest.mock import patch, MagicMock

captured_requests = []

def mock_post(self, *args, **kwargs):
    """Capture the actual request being sent"""
    url = args[0] if args else kwargs.get('url', '')

    if 'inceptionlabs' in url:
        # Extract the request body
        content = kwargs.get('content', {})
        if isinstance(content, dict) and 'json' in content:
            body = content['json']
            captured_requests.append(body)
            print("\n" + "=" * 70)
            print("CAPTURED HTTP REQUEST TO MERCURY-2 API")
            print("=" * 70)
            print(f"URL: {url}")
            print(f"Method: POST")
            print(f"\nRequest body:")
            print(json.dumps(body, indent=2))

            # Check if reasoning_effort is present
            if 'reasoning_effort' in body:
                print(f"\n✅ reasoning_effort IS present in request: {body['reasoning_effort']}")
            else:
                print(f"\n❌ reasoning_effort NOT present in request")
                print(f"   Request keys: {list(body.keys())}")

        # Return a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "reasoning_tokens": 6
            }
        }
        return mock_response

    return MagicMock(status_code=200)

# Patch httpx to capture the request
print("\nInvoking LLM to capture the actual request...")
with patch.object(httpx.Client, 'post', mock_post):
    try:
        result = llm.invoke("What is 2+2?")
    except Exception as e:
        print(f"Error during invoke: {e}")

if captured_requests:
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    body = captured_requests[0]
    if 'reasoning_effort' in body:
        print("✅ SUCCESS: reasoning_effort parameter IS being sent to the API")
        print(f"   Value: {body['reasoning_effort']}")
    else:
        print("❌ FAILURE: reasoning_effort parameter is NOT being sent")
        print(f"   Request contains: {list(body.keys())}")
        print("\n   Need to fix the implementation!")
else:
    print("⚠️  No requests captured")
