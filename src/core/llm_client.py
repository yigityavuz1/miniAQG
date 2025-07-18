"""
LLM Client for centralized model interactions with cost tracking and retry logic.
"""

import os
import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Latest imports based on documentation
from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai

from .logger import AQGLogger


class ModelProvider(Enum):
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class TokenUsage:
    """Token usage tracking for cost calculation."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0


@dataclass
class LLMResponse:
    """Standardized response format for all LLM calls."""
    content: str
    model: str
    provider: ModelProvider
    usage: TokenUsage
    response_time: float
    metadata: Dict[str, Any] = None


class LLMClient:
    """
    Centralized LLM client with retry logic, cost tracking, and standardized interfaces.
    Updated for June 2025 with latest APIs.
    """
    
    # Updated pricing per 1M tokens (June 2025)
    PRICING = {
        "gpt-4o": {
            "prompt": 2.50,  # $2.50 per 1M input tokens
            "completion": 10.00  # $10.00 per 1M output tokens
        },
        "gemini-2.0-flash-001": {
            "prompt": 0.075,  # $0.075 per 1M input tokens
            "completion": 0.30   # $0.30 per 1M output tokens
        }
    }
    
    def __init__(self, logger: Optional[AQGLogger] = None):
        self.logger = logger or AQGLogger()
        self.total_cost = 0.0
        self.total_requests = 0
        
        # Initialize clients with latest APIs
        self._init_openai_client()
        self._init_google_client()
    
    async def initialize(self):
        """Async initialization method for API compatibility."""
        # For now, just log that initialization is complete
        self.logger.info("LLM client async initialization complete")
        return True
        
    def _init_openai_client(self):
        """Initialize OpenAI client with latest API."""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            self.async_openai_client = AsyncOpenAI(api_key=api_key)
            self.logger.info("OpenAI client initialized")
        else:
            self.openai_client = None
            self.async_openai_client = None
            self.logger.warning("OpenAI API key not found")
    
    def _init_google_client(self):
        """Initialize Google GenAI client with latest API."""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            # Configure the API key
            genai.configure(api_key=api_key)
            self.google_client = genai
            self.logger.info("Google GenAI client initialized")
        else:
            self.google_client = None
            self.logger.warning("Google API key not found")
    
    def _calculate_cost(self, model: str, usage: TokenUsage) -> float:
        """Calculate cost based on token usage and model pricing."""
        if model not in self.PRICING:
            return 0.0
            
        pricing = self.PRICING[model]
        prompt_cost = (usage.prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (usage.completion_tokens / 1_000_000) * pricing["completion"]
        return prompt_cost + completion_cost
    
    async def _retry_with_backoff(self, func, max_retries=3, base_delay=1.0):
        """Exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                    
                delay = base_delay * (2 ** attempt)
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
    
    async def call_openai_chat(
        self,
        prompt: str,
        model: str = "gpt-4o",
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> LLMResponse:
        """
        Call OpenAI using the standard Chat Completions API.
        """
        if not self.async_openai_client:
            raise ValueError("OpenAI client not initialized")
        
        start_time = time.time()
        
        async def _make_call():
            # Build messages
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            # Call the chat completions API
            response = await self.async_openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            content = response.choices[0].message.content
            
            # Clean JSON if needed
            content = self._clean_json_response(content)
            
            # Extract usage information
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            
            return content, usage
        
        content, usage = await self._retry_with_backoff(_make_call)
        
        response_time = time.time() - start_time
        cost = self._calculate_cost(model, usage)
        usage.cost_usd = cost
        
        self.total_cost += cost
        self.total_requests += 1
        
        self.logger.info(f"OpenAI call completed: {model}, tokens: {usage.total_tokens}, cost: ${cost:.4f}")
        
        return LLMResponse(
            content=content,
            model=model,
            provider=ModelProvider.OPENAI,
            usage=usage,
            response_time=response_time,
            metadata={"system_instruction": system_instruction, "temperature": temperature}
        )
    
    async def call_google_genai(
        self,
        prompt: str,
        model: str = "gemini-1.5-flash",
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        Call Google GenAI using the standard client.
        """
        if not self.google_client:
            raise ValueError("Google GenAI client not initialized")
        
        start_time = time.time()
        
        async def _make_call():
            # Use the standard generate_content API
            model_instance = self.google_client.GenerativeModel(model)
            
            # Build the prompt with system instruction if provided
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
            
            response = model_instance.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    **kwargs
                )
            )
            
            content = response.text
            
            # Extract usage information if available
            usage_info = getattr(response, 'usage_metadata', None)
            if usage_info:
                usage = TokenUsage(
                    prompt_tokens=getattr(usage_info, 'prompt_token_count', 0),
                    completion_tokens=getattr(usage_info, 'candidates_token_count', 0),
                    total_tokens=getattr(usage_info, 'total_token_count', 0)
                )
            else:
                # Estimate tokens if usage not available
                estimated_prompt_tokens = len(prompt.split()) * 1.3
                estimated_completion_tokens = len(content.split()) * 1.3
                usage = TokenUsage(
                    prompt_tokens=int(estimated_prompt_tokens),
                    completion_tokens=int(estimated_completion_tokens),
                    total_tokens=int(estimated_prompt_tokens + estimated_completion_tokens)
                )
            
            return content, usage
        
        content, usage = await self._retry_with_backoff(_make_call)
        
        response_time = time.time() - start_time
        cost = self._calculate_cost(model, usage)
        usage.cost_usd = cost
        
        self.total_cost += cost
        self.total_requests += 1
        
        self.logger.info(f"Google GenAI call completed: {model}, tokens: {usage.total_tokens}, cost: ${cost:.4f}")
        
        return LLMResponse(
            content=content,
            model=model,
            provider=ModelProvider.GOOGLE,
            usage=usage,
            response_time=response_time,
            metadata={"system_instruction": system_instruction, "temperature": temperature}
        )
    
    async def call_model(
        self,
        prompt: str,
        model: str = "gpt-4o",
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Universal model caller that routes to appropriate provider.
        For testing, we'll use gpt-4o for everything as requested.
        """
        if model.startswith("gpt-") or model in ["gpt-4o"]:
            return await self.call_openai_chat(
                prompt=prompt,
                model=model,
                system_instruction=system_instruction,
                **kwargs
            )
        elif model.startswith("gemini-"):
            return await self.call_google_genai(
                prompt=prompt,
                model=model,
                system_instruction=system_instruction,
                **kwargs
            )
        else:
            # Default to OpenAI for testing
            self.logger.warning(f"Unknown model {model}, defaulting to gpt-4o")
            return await self.call_openai_chat(
                prompt=prompt,
                model="gpt-4o",
                system_instruction=system_instruction,
                **kwargs
            )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of costs and usage."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_requests": self.total_requests,
            "average_cost_per_request": round(self.total_cost / max(1, self.total_requests), 4),
            "target_cost_per_question": 0.05,  # $0.05 target
            "cost_efficiency": "within_target" if self.total_cost <= 0.05 * self.total_requests else "over_target"
        }
    
    def reset_tracking(self):
        """Reset cost and usage tracking."""
        self.total_cost = 0.0
        self.total_requests = 0
        self.logger.info("Cost tracking reset")

    def _clean_json_response(self, content: str) -> str:
        """
        Clean JSON response by removing markdown code blocks and other formatting.
        
        Args:
            content: Raw LLM response content
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        content = content.strip()
        
        # Remove ```json and ``` wrapper
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        elif content.startswith("```"):
            content = content[3:]   # Remove ```
            
        if content.endswith("```"):
            content = content[:-3]  # Remove closing ```
            
        return content.strip()


# Global instance for easy access
llm_client = LLMClient() 