# coding=utf-8
"""Common utility for OpenAI-compatible chat completions with rate limiting and throttling."""

import os
import json
import time
import random
import threading
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List, Callable
from collections import defaultdict

from openai import OpenAI, AsyncOpenAI, APIError, RateLimitError  # type: ignore
try:
    from openai import APITimeoutError as TimeoutError  # type: ignore
except Exception:  # pragma: no cover
    TimeoutError = Exception  # fallback
try:
    from openai import InternalServerError, ServiceUnavailableError  # type: ignore
except Exception:  # pragma: no cover
    InternalServerError = ServiceUnavailableError = Exception  # fallback


@dataclass
class RateLimitConfig:
    """Rate limit configuration based on API headers."""
    requests_per_day: int = 14400
    tokens_per_minute: int = 1000000
    
    # Current remaining limits (updated from response headers)
    remaining_requests_day: int = 0
    remaining_tokens_minute: int = 0
    
    # Reset times in seconds (updated from response headers)
    reset_requests_day: float = 0.0
    reset_tokens_minute: float = 0.0
    
    
    # Advanced settings from OpenAI Cookbook
    max_tokens_conservative: bool = True  # Use conservative max_tokens estimates
    proactive_delay: bool = True  # Add proactive delays to stay under rate limits


class TokenEstimator:
    """Estimates token usage with exponential moving average."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize token estimator.
        
        Args:
            alpha: Smoothing factor for exponential moving average (0 < alpha <= 1)
        """
        self.alpha = alpha
        self.ema_ratio = 1.0  # Exponential moving average of actual_tokens / estimated_tokens
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def estimate_tokens(self, content: str) -> int:
        """
        Estimate tokens for a given content.
        
        Args:
            content: Text content to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Rough estimation: len(content) / 4
        base_estimate = max(1, len(content) // 4)
        
        with self.lock:
            # Apply exponential moving average adjustment
            adjusted_estimate = int(base_estimate * self.ema_ratio)
            return max(1, adjusted_estimate)
    
    def estimate_conservative_max_tokens(self, expected_response_length: int) -> int:
        """
        Estimate conservative max_tokens to avoid rate limit overestimation.
        
        From OpenAI Cookbook: Rate limit usage is calculated based on the greater of:
        1. max_tokens - the maximum number of tokens allowed in a response
        2. Estimated tokens in your input
        
        Args:
            expected_response_length: Expected length of response in characters
            
        Returns:
            Conservative max_tokens estimate
        """
        # Convert expected response length to tokens (rough estimate)
        base_tokens = max(10, expected_response_length // 4)
        
        with self.lock:
            # Add 20% buffer for safety, but keep it conservative
            conservative_estimate = int(base_tokens * 1.2)
            return conservative_estimate
    
    def update_with_actual(self, estimated_tokens: int, actual_tokens: int):
        """
        Update the estimator with actual token usage.
        
        Args:
            estimated_tokens: Previously estimated tokens
            actual_tokens: Actual tokens used from API response
        """
        if estimated_tokens <= 0 or actual_tokens <= 0:
            return
            
        with self.lock:
            ratio = actual_tokens / estimated_tokens
            # Update exponential moving average
            old_ratio = self.ema_ratio
            self.ema_ratio = self.alpha * ratio + (1 - self.alpha) * self.ema_ratio
            
            self.logger.debug(f"Token estimation updated: {old_ratio:.3f} -> {self.ema_ratio:.3f} "
                            f"(actual: {actual_tokens}, estimated: {estimated_tokens})")


class AsyncRateLimiter:
    """Async-aware rate limiter that respects API rate limits and throttles requests."""
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize async rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.token_estimator = TokenEstimator()
        self.lock = asyncio.Lock()
        
        # Track token usage for the current minute window
        self._minute = int(time.time() // 60)
        self._usage = 0
        
    def update_from_headers(self, headers: Dict[str, str]):
        """
        Update rate limit state from response headers.
        
        Args:
            headers: HTTP response headers
        """
        # Update remaining limits
        if 'x-ratelimit-remaining-requests-day' in headers:
            self.config.remaining_requests_day = int(headers['x-ratelimit-remaining-requests-day'])
        
        if 'x-ratelimit-remaining-tokens-minute' in headers:
            self.config.remaining_tokens_minute = int(headers['x-ratelimit-remaining-tokens-minute'])
        
        # Update reset times
        if 'x-ratelimit-reset-requests-day' in headers:
            self.config.reset_requests_day = float(headers['x-ratelimit-reset-requests-day'])
        
        if 'x-ratelimit-reset-tokens-minute' in headers:
            self.config.reset_tokens_minute = float(headers['x-ratelimit-reset-tokens-minute'])
    
    
    async def wait_if_needed(self, estimated_tokens: int) -> float:
        """
        Wait if necessary to respect rate limits (async version).
        
        Args:
            estimated_tokens: Estimated tokens for the upcoming request
            
        Returns:
            Time waited in seconds
        """
        async with self.lock:
            # If no rate-limit headers provided, skip throttling
            if self.config.remaining_requests_day <= 0 and self.config.remaining_tokens_minute <= 0:
                return 0.0
            # Reset usage counter if minute has rolled over
            now_minute = int(time.time() // 60)
            if now_minute != self._minute:
                self._minute = now_minute
                self._usage = 0
            current_time = time.time()
            current_token_usage = self._usage
            
            wait_time = 0.0
            
            # Proactive delay calculation (from OpenAI Cookbook)
            if self.config.proactive_delay:
                # Calculate delay to stay under rate limits proactively
                requests_per_second = self.config.requests_per_day / (24 * 60 * 60)
                tokens_per_second = self.config.tokens_per_minute / 60
                
                # Add small delay to prevent hitting limits
                proactive_delay = max(1.0 / requests_per_second, estimated_tokens / tokens_per_second)
                wait_time = max(wait_time, proactive_delay * 0.1)  # Use 10% of calculated delay
            
            # Check token rate limit
            if current_token_usage + estimated_tokens > self.config.remaining_tokens_minute:
                # Need to wait until next minute or reset time
                if self.config.reset_tokens_minute > 0:
                    wait_time = max(wait_time, self.config.reset_tokens_minute)
                else:
                    # Wait until next minute
                    seconds_until_next_minute = 60 - (current_time % 60)
                    wait_time = max(wait_time, seconds_until_next_minute)
            
            # Check request rate limit (simplified - assumes we're not hitting daily limit often)
            if self.config.remaining_requests_day <= 1:
                if self.config.reset_requests_day > 0:
                    wait_time = max(wait_time, self.config.reset_requests_day)
            
            if wait_time > 0:
                logging.info(f"Rate limit throttling: waiting {wait_time:.2f}s (estimated tokens: {estimated_tokens}, "
                           f"current usage: {current_token_usage}, remaining: {self.config.remaining_tokens_minute})")
                await asyncio.sleep(wait_time)
                
                # Update tracking after waiting
                current_minute = int(time.time() // 60)
                
            # Record estimated token usage in this minute
            self._usage += estimated_tokens
            
            return wait_time
    
    async def record_actual_usage(self, estimated_tokens: int, actual_tokens: int):
        """
        Record actual token usage and update estimator (async version).
        
        Args:
            estimated_tokens: Previously estimated tokens
            actual_tokens: Actual tokens used
        """
        async with self.lock:
            # Reset usage if minute rolled over
            now_minute = int(time.time() // 60)
            if now_minute != self._minute:
                self._minute = now_minute
                self._usage = 0
            # Adjust usage based on actual token count
            self._usage += (actual_tokens - estimated_tokens)
            # Update estimator EMA
            self.token_estimator.update_with_actual(estimated_tokens, actual_tokens)


class AsyncChatCompletionClient:
    """Async OpenAI-compatible chat completion client with rate limiting."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """
        Initialize async chat completion client.
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional custom base URL for OpenAI-compatible endpoints
            rate_limit_config: Rate limit configuration (uses defaults if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided via api_key parameter or OPENAI_API_KEY env variable")
        
        self.base_url = base_url
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.rate_limiter = AsyncRateLimiter(self.rate_limit_config)
        self._client = None
        
    def _build_client(self) -> AsyncOpenAI:
        """Create AsyncOpenAI client."""
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncOpenAI(**kwargs)
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the async client."""
        if self._client is None:
            self._client = self._build_client()
        return self._client
    
    async def _chat_call(self, 
                        model: str, 
                        prompt_text: str, 
                        temperature: Optional[float] = None, 
                        max_tokens: Optional[int] = None, 
                        reasoning_effort: Optional[str] = None) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Perform an async chat completion call.
        
        Args:
            model: Model name
            prompt_text: Prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            reasoning_effort: Reasoning effort hint
            
        Returns:
            Tuple of (response_text, usage_info)
        """
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
            
        resp = await self.client.chat.completions.create(**params)
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("Empty response from API, please increase max_tokens or check model config.")
        
        # Extract usage information
        usage_info = None
        if hasattr(resp, 'usage') and resp.usage:
            usage_info = {
                'prompt_tokens': getattr(resp.usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(resp.usage, 'completion_tokens', 0),
                'total_tokens': getattr(resp.usage, 'total_tokens', 0),
            }
        
        return content.strip(), usage_info
    
    async def chat_completion_with_retry(self,
                                        prompt_text: str,
                                        model: str,
                                        max_retries: int = 5,
                                        backoff_in_seconds: float = 10.0,
                                        temperature: Optional[float] = None,
                                        max_tokens: Optional[int] = None,
                                        reasoning_effort: Optional[str] = None,
                                        expected_response_length: Optional[int] = None) -> str:
        """
        Call async chat completion with retries on transient errors and rate limiting.
        
        Args:
            prompt_text: Prompt text
            model: Model name
            max_retries: Maximum number of retry attempts
            backoff_in_seconds: Base delay in seconds for exponential backoff
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            reasoning_effort: Reasoning effort hint
            expected_response_length: Expected response length for conservative max_tokens
            
        Returns:
            Response text
        """
        # Determine max_tokens conservatively
        if max_tokens is None and self.rate_limit_config.max_tokens_conservative and expected_response_length:
            max_tokens = self.rate_limiter.token_estimator.estimate_conservative_max_tokens(expected_response_length)
            logging.debug(f"Using conservative max_tokens: {max_tokens} for expected length: {expected_response_length}")
        
        # Estimate tokens and wait for rate limiting
        estimated_tokens = self.rate_limiter.token_estimator.estimate_tokens(prompt_text)
        if max_tokens:
            estimated_tokens += max_tokens  # Add expected response tokens
        
        # Wait if needed for rate limiting
        await self.rate_limiter.wait_if_needed(estimated_tokens)
        
        retry_exceptions = (RateLimitError, APIError, TimeoutError, InternalServerError, ServiceUnavailableError)
        for attempt in range(max_retries):
            try:
                response_text, usage_info = await self._chat_call(
                    model, prompt_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort
                )
                # Record actual usage
                if usage_info and 'total_tokens' in usage_info:
                    await self.rate_limiter.record_actual_usage(estimated_tokens, usage_info['total_tokens'])
                return response_text

            except retry_exceptions as e:
                if attempt == max_retries - 1:
                    raise
                # exponential backoff with jitter
                wait = random.uniform(backoff_in_seconds * 2**attempt / 2, backoff_in_seconds * 2**attempt)
                logging.warning(f"API error ({e}); retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
        # Should not reach here
        raise RuntimeError("chat_completion_with_retry failed after retries")
    
    async def close(self):
        """Close the async client."""
        if self._client:
            await self._client.close()


def create_async_chat_client(api_key: Optional[str] = None,
                            base_url: Optional[str] = None,
                            max_tokens_conservative: bool = True,
                            proactive_delay: bool = True) -> AsyncChatCompletionClient:
    """
    Convenience function to create an async chat completion client.
    
    Args:
        api_key: API key (defaults to OPENAI_API_KEY env var)
        base_url: Optional custom base URL for OpenAI-compatible endpoints
        max_tokens_conservative: Use conservative max_tokens estimates
        proactive_delay: Add proactive delays to stay under rate limits
        
    Returns:
        Configured AsyncChatCompletionClient
    """
    # initialize rate limit config; limits are read from headers
    rate_config = RateLimitConfig(
        max_tokens_conservative=max_tokens_conservative,
        proactive_delay=proactive_delay,
    )
    
    return AsyncChatCompletionClient(
        api_key=api_key,
        base_url=base_url,
        rate_limit_config=rate_config
    )


def setup_logging(level: str = "INFO"):
    """
    Setup logging for the chat completion utility.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
