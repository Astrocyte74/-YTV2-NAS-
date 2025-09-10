#!/usr/bin/env python3
"""
LLM Configuration Manager for YTv2
Integrates with mkpy LLM Management System for centralized API keys and model selection
"""

import os
from typing import Dict, Optional, Tuple
from pathlib import Path


class LLMConfig:
    """Centralized LLM configuration with mkpy integration"""
    
    # 2025 Model Shortlists - Updated with verified working models
    SHORTLISTS = {
        "research": {
            "primary": [("openai", "gpt-5"), ("openrouter", "anthropic/claude-4-opus"), ("openrouter", "qwen/qwen-2.5-72b-instruct")],
            "fallback": [("openrouter", "openai/gpt-4o"), ("openrouter", "anthropic/claude-3-5-sonnet-20241022")]
        },
        "budget": {
            "primary": [("openai", "gpt-5-nano"), ("openrouter", "z-ai/glm-4.5"), ("ollama", "phi3:latest")],
            "fallback": [("openrouter", "openai/gpt-4o-mini"), ("openrouter", "meta-llama/llama-3.1-8b-instruct")]
        },
        "fast": {
            "primary": [("openai", "gpt-5-nano"), ("openrouter", "anthropic/claude-3-haiku-20240307"), ("ollama", "phi3:latest")],
            "fallback": [("openrouter", "openai/gpt-4o-mini"), ("openrouter", "anthropic/claude-3.7-sonnet")]
        },
        "creative": {
            "primary": [("openai", "gpt-5"), ("openrouter", "anthropic/claude-4-opus"), ("openai", "gpt-5-mini")],
            "fallback": [("openrouter", "anthropic/claude-3-5-sonnet-20241022"), ("openrouter", "openai/gpt-4o")]
        },
        "coding": {
            "primary": [("openrouter", "moonshotai/kimi-k2"), ("openrouter", "z-ai/glm-4.5"), ("openrouter", "qwen/qwen3-coder")],
            "fallback": [("openrouter", "anthropic/claude-3-5-sonnet-20241022"), ("openrouter", "openai/gpt-4o")]
        },
        "local": {
            "primary": [("ollama", "gpt-oss:20b"), ("ollama", "gemma3:12b"), ("ollama", "qwen2.5-coder:7b")],
            "fallback": [("ollama", "phi3:latest"), ("ollama", "llama3.2:3b")]
        }
    }
    
    # Default models for backward compatibility
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-sonnet-20240229",
        "openrouter": "openai/gpt-4o-mini",
        "ollama": "llama3.2"
    }
    
    def __init__(self):
        self.load_environment()
    
    def load_environment(self):
        """Load environment variables from mkpy system or fallback to project .env"""
        
        # Check if running in mkpy-integrated environment
        self.llm_profile = os.getenv('LLM_PROFILE', 'default')
        self.llm_shortlist = os.getenv('LLM_SHORTLIST', 'research')  # Default to research for summarization
        self.llm_model = os.getenv('LLM_MODEL')
        self.llm_provider = os.getenv('LLM_PROVIDER')
        
        # Load API keys
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY') 
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        
        # Ollama settings
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        
        # If no mkpy environment detected, try loading from .env
        if not self.openai_key and not self.anthropic_key:
            self._load_dotenv_fallback()
    
    def _load_dotenv_fallback(self):
        """Fallback to loading from .env file for backward compatibility"""
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                self.openai_key = os.getenv('OPENAI_API_KEY')
                self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
                self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
                print("📝 Loaded API keys from .env file")
        except ImportError:
            pass
    
    def get_model_config(self, provider: Optional[str] = None, model: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Get model configuration based on shortlist, provider preference, or explicit model
        
        Returns:
            Tuple of (provider, model, api_key)
        """
        
        # If explicit provider and model specified, use those
        if provider and model:
            api_key = self._get_api_key(provider)
            if api_key:
                return provider, model, api_key
        
        # If LLM_MODEL is set in environment, use it
        if self.llm_model:
            detected_provider = self._detect_provider_from_model(self.llm_model)
            if detected_provider:
                api_key = self._get_api_key(detected_provider)
                if api_key:
                    return detected_provider, self.llm_model, api_key
        
        # Use shortlist-based selection
        if self.llm_shortlist in self.SHORTLISTS:
            shortlist = self.SHORTLISTS[self.llm_shortlist]
            
            # Try primary models first
            for shortlist_provider, shortlist_model in shortlist["primary"]:
                api_key = self._get_api_key(shortlist_provider)
                if api_key or shortlist_provider == "ollama":  # Ollama doesn't need API key
                    print(f"🎯 Using {self.llm_shortlist} shortlist: {shortlist_provider}/{shortlist_model}")
                    return shortlist_provider, shortlist_model, api_key
            
            # Try fallback models
            for shortlist_provider, shortlist_model in shortlist["fallback"]:
                api_key = self._get_api_key(shortlist_provider)
                if api_key:
                    print(f"🔄 Using {self.llm_shortlist} fallback: {shortlist_provider}/{shortlist_model}")
                    return shortlist_provider, shortlist_model, api_key
        
        # Ultimate fallback - use any available provider
        for fallback_provider in ['openai', 'anthropic', 'openrouter']:
            api_key = self._get_api_key(fallback_provider)
            if api_key:
                fallback_model = self.DEFAULT_MODELS[fallback_provider]
                print(f"⚠️  Using fallback: {fallback_provider}/{fallback_model}")
                return fallback_provider, fallback_model, api_key
        
        # No API keys available
        raise ValueError(
            "❌ No API keys found. Please:\n"
            "   1. Run 'mkpy llm init' to set up centralized keys, or\n"
            "   2. Create .env file with OPENAI_API_KEY or ANTHROPIC_API_KEY"
        )
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider"""
        if provider == "openai":
            return self.openai_key
        elif provider == "anthropic":
            return self.anthropic_key
        elif provider == "openrouter":
            return self.openrouter_key
        elif provider == "ollama":
            return None  # Ollama doesn't need API key
        return None
    
    def _detect_provider_from_model(self, model: str) -> Optional[str]:
        """Detect provider from model name"""
        if any(prefix in model.lower() for prefix in ['gpt', 'chatgpt', 'openai']):
            return "openai"
        elif any(prefix in model.lower() for prefix in ['claude', 'anthropic']):
            return "anthropic"
        elif '/' in model:  # OpenRouter format
            return "openrouter"
        elif any(suffix in model.lower() for suffix in [':latest', ':3b', ':7b', ':9b', ':12b']):
            return "ollama"
        return None
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get status of available providers"""
        return {
            "openai": bool(self.openai_key),
            "anthropic": bool(self.anthropic_key),
            "openrouter": bool(self.openrouter_key),
            "ollama": True  # Assume Ollama is available if needed
        }
    
    def print_status(self):
        """Print current LLM configuration status"""
        print(f"🔧 LLM Configuration Status:")
        print(f"   Profile: {self.llm_profile}")
        print(f"   Shortlist: {self.llm_shortlist}")
        print(f"   Explicit Model: {self.llm_model or 'None'}")
        print(f"   Available Providers:")
        
        for provider, available in self.get_available_providers().items():
            status = "✅" if available else "❌"
            print(f"     {status} {provider}")
        
        # Show current shortlist models
        if self.llm_shortlist in self.SHORTLISTS:
            shortlist = self.SHORTLISTS[self.llm_shortlist]
            print(f"   {self.llm_shortlist.title()} Shortlist Models:")
            for i, (provider, model) in enumerate(shortlist["primary"]):
                print(f"     {i+1}. {provider}/{model}")


# Global instance
llm_config = LLMConfig()