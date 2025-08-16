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
    requests_per_minute: Optional[int] = 30  # default 30 RPM (None/0 disables)
    
    # Current remaining limits (updated from response headers)
    remaining_requests_day: int = 0
    remaining_tokens_minute: int = 0
    
    # Reset times in seconds (updated from response headers)
    reset_requests_day: float = 0.0
    reset_tokens_minute: float = 0.0
    
    
    # Advanced settings from OpenAI Cookbook
    max_tokens_conservative: bool = True  # Use conservative max_tokens estimates
    proactive_delay: bool = True  # Add proactive delays to stay under rate limits


@dataclass
class RetryPolicy:
    """Configuration for retry/backoff behavior."""
    max_retries: int = 10
    base_backoff_seconds: float = 10.0
    warn_after_attempt: int = 7  # log warnings only after this attempt index (1-based)
    jitter_factor: float = 0.5   # fraction for +/- jitter window around backoff


@dataclass
class StormPolicy:
    """Configuration for handling throttle storms with adaptive cooldowns.

    A "throttle" signal generally includes HTTP 429, and optionally 503 with
    explicit Retry-After or rate-limit headers indicating pressure.
    """
    min_cooldown_seconds: float = 120.0
    ema_alpha: float = 0.3
    consecutive_threshold: int = 2
    ema_threshold: float = 0.5


@dataclass
class PacingPolicy:
    """Configuration for proactive pacing to avoid limits."""
    proactive_delay: bool = True


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
        # Global cooldown timestamp (epoch seconds). When set in the future, all callers should wait.
        self._cooldown_until: float = 0.0
        # Request count buckets (shared logic via helper): per-minute and per-day
        now = time.time()
        # Per-minute bucket
        self._req_min_capacity: float = float(self.config.requests_per_minute or 0)
        self._req_min_tokens: float = self._req_min_capacity
        self._req_min_last: float = now
        # Per-day bucket (soft cap; independent of header-based remaining)
        self._req_day_capacity: float = float(self.config.requests_per_day or 0)
        self._req_day_tokens: float = self._req_day_capacity
        self._req_day_last: float = now

    def _bucket_reserve_or_wait(self,
                                now: float,
                                capacity: float,
                                period_seconds: float,
                                tokens: float,
                                last_refill: float) -> tuple[bool, float, float, float]:
        """Refill a token bucket and either reserve a token or compute wait.

        Returns: (reserved, wait_seconds, new_tokens, new_last_refill)
        """
        if capacity <= 0 or period_seconds <= 0:
            return True, 0.0, tokens, last_refill
        rate = capacity / period_seconds  # tokens per second
        elapsed = max(0.0, now - last_refill)
        tokens = min(capacity, tokens + elapsed * rate)
        last_refill = now
        if tokens >= 1.0:
            tokens -= 1.0
            return True, 0.0, tokens, last_refill
        deficit = 1.0 - tokens
        wait = deficit / rate if rate > 0 else 0.0
        return False, wait, tokens, last_refill
        
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
        # Some providers send 'retry-after' to indicate when to try again (seconds)
        # We don't set cooldown here automatically to avoid over-sleeping on single requests.
        # The retry loop will consider it and call set_cooldown if 429s are frequent.

    async def set_cooldown(self, seconds: float, reason: str = ""):  # reason is for logging context
        """Set a global cooldown for all callers to respect."""
        async with self.lock:
            now = time.time()
            new_until = now + max(0.0, seconds)
            if new_until > self._cooldown_until:
                self._cooldown_until = new_until
                logging.info(f"Global cooldown set for {seconds:.1f}s{(' - ' + reason) if reason else ''}")

    async def get_cooldown_remaining(self) -> float:
        """Get remaining cooldown seconds (0 if none)."""
        async with self.lock:
            now = time.time()
            return max(0.0, self._cooldown_until - now)
    
    
    async def wait_if_needed(self, estimated_tokens: int) -> float:
        """
        Wait if necessary to respect rate limits (async version).
        
        Args:
            estimated_tokens: Estimated tokens for the upcoming request
            
        Returns:
            Time waited in seconds
        """
        # Phase 1: compute wait under lock
        reserved_min = False
        reserved_day = False
        async with self.lock:
            # If no rate-limit headers provided, skip throttling but honor global cooldown
            now = time.time()
            wait_time = 0.0
            if self._cooldown_until > now:
                wait_time = max(wait_time, self._cooldown_until - now)

            # Reserve from request-per-minute bucket
            reserved_min, wait_min, self._req_min_tokens, self._req_min_last = self._bucket_reserve_or_wait(
                now, self._req_min_capacity, 60.0, self._req_min_tokens, self._req_min_last
            )
            wait_time = max(wait_time, wait_min)
            # Reserve from request-per-day bucket (soft cap)
            reserved_day, wait_day, self._req_day_tokens, self._req_day_last = self._bucket_reserve_or_wait(
                now, self._req_day_capacity, 24 * 60 * 60, self._req_day_tokens, self._req_day_last
            )
            wait_time = max(wait_time, wait_day)

            if self.config.remaining_requests_day > 0 or self.config.remaining_tokens_minute > 0:
                # Reset usage counter if minute has rolled over
                now_minute = int(now // 60)
                if now_minute != self._minute:
                    self._minute = now_minute
                    self._usage = 0
                current_token_usage = self._usage

                # Proactive delay calculation (from OpenAI Cookbook)
                if self.config.proactive_delay:
                    requests_per_second = self.config.requests_per_day / (24 * 60 * 60)
                    tokens_per_second = self.config.tokens_per_minute / 60
                    proactive_delay = max(1.0 / requests_per_second, estimated_tokens / tokens_per_second)
                    wait_time = max(wait_time, proactive_delay * 0.1)

                # Check token rate limit
                if current_token_usage + estimated_tokens > self.config.remaining_tokens_minute:
                    if self.config.reset_tokens_minute > 0:
                        wait_time = max(wait_time, self.config.reset_tokens_minute)
                    else:
                        seconds_until_next_minute = 60 - (now % 60)
                        wait_time = max(wait_time, seconds_until_next_minute)

                # Check request daily limit (rare)
                if self.config.remaining_requests_day <= 1 and self.config.reset_requests_day > 0:
                    wait_time = max(wait_time, self.config.reset_requests_day)

            # Capture whether we'll sleep and log context, then release lock before sleeping
            log_ctx = (estimated_tokens, self._usage, self.config.remaining_tokens_minute,
                       self._req_min_capacity, self._req_day_capacity)

        # Phase 2: sleep outside the lock to avoid blocking other callers
        if wait_time > 0:
            logging.info(
                f"Rate limit throttling: waiting {wait_time:.2f}s (estimated tokens: {log_ctx[0]}, current usage: {log_ctx[1]}, remaining/min: {log_ctx[2]}, req/min cap: {log_ctx[3]}, req/day cap: {log_ctx[4]})"
            )
            await asyncio.sleep(wait_time)

        # Phase 3: record estimated usage under lock
        async with self.lock:
            # After sleeping, if we didn't reserve earlier, try to consume now
            now2 = time.time()
            if not reserved_min:
                reserved_min, _, self._req_min_tokens, self._req_min_last = self._bucket_reserve_or_wait(
                    now2, self._req_min_capacity, 60.0, self._req_min_tokens, self._req_min_last
                )
            if not reserved_day:
                reserved_day, _, self._req_day_tokens, self._req_day_last = self._bucket_reserve_or_wait(
                    now2, self._req_day_capacity, 24 * 60 * 60, self._req_day_tokens, self._req_day_last
                )
            # Reset usage if minute rolled over during sleep
            now_minute = int(time.time() // 60)
            if now_minute != self._minute:
                self._minute = now_minute
                self._usage = 0
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
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 retry_policy: Optional[RetryPolicy] = None,
                 storm_policy: Optional[StormPolicy] = None,
                 pacing_policy: Optional[PacingPolicy] = None):
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
        # Policies
        self.retry_policy = retry_policy or RetryPolicy()
        self.storm_policy = storm_policy or StormPolicy()
        self.pacing_policy = pacing_policy or PacingPolicy(proactive_delay=self.rate_limit_config.proactive_delay)
        # Throttle storm detection without a deque: consecutive counter + EWMA
        self._consecutive_throttle: int = 0
        self._ema_throttle: float = 0.0

    def _extract_headers(self, e: Exception) -> Dict[str, str]:
        """Best-effort extraction of headers from an exception/response."""
        headers = None
        try:
            headers = getattr(e, 'response', None)
            headers = getattr(headers, 'headers', None) or getattr(e, 'headers', None)
        except Exception:
            headers = None
        if not headers:
            return {}
        try:
            return dict(headers)
        except Exception:
            return {}

    def _parse_retry_after(self, headers: Dict[str, str]) -> Optional[float]:
        """Parse Retry-After seconds if present (header may be case-insensitive)."""
        if not headers:
            return None
        ra_val = headers.get('retry-after') or headers.get('Retry-After')
        if ra_val is None:
            return None
        try:
            return float(ra_val)
        except Exception:
            return None

    def _is_throttle_signal(self, e: Exception, headers: Dict[str, str]) -> bool:
        """Classify errors that indicate throttling pressure.

        True for 429 (RateLimitError). For 503 (ServiceUnavailable), return True
        only when Retry-After is present or when rate-limit headers are present.
        """
        if isinstance(e, RateLimitError):
            return True
        if isinstance(e, ServiceUnavailableError):
            # treat as throttle only with explicit backpressure signals
            if self._parse_retry_after(headers) is not None:
                return True
            # Some providers include x-ratelimit-* even on 503s
            return any(k.startswith('x-ratelimit-') for k in headers.keys())
        return False
        
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
            
        # Try to obtain raw response to access headers for rate limit updates
        content = None
        usage_info = None
        headers: Dict[str, str] | None = None
        try:
            with_raw = getattr(self.client.chat.completions, "with_raw_response", None)
            if with_raw is not None:
                raw = await with_raw.create(**params)
                headers = dict(raw.headers or {})
                resp = raw.parse()
            else:
                resp = await self.client.chat.completions.create(**params)
                # Some clients expose headers via a 'response' attribute
                headers = dict(getattr(resp, 'headers', {}) or {})
        except Exception:
            # Fallback to normal create if with_raw_response path fails unexpectedly
            resp = await self.client.chat.completions.create(**params)
            headers = dict(getattr(resp, 'headers', {}) or {})

        # Update limiter with any rate headers we received
        if headers:
            try:
                self.rate_limiter.update_from_headers(headers)
            except Exception:
                pass

        content = resp.choices[0].message.content
        if not content:
            raise ValueError("Empty response from API, please increase max_tokens or check model config.")
        
        # Extract usage information
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
        
        rp = self.retry_policy
        retry_exceptions = (RateLimitError, APIError, TimeoutError, InternalServerError, ServiceUnavailableError)
        for attempt in range(rp.max_retries):
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
                # Success: reset throttle trackers and gently decay EMA
                self._consecutive_throttle = 0
                a = self.storm_policy.ema_alpha
                self._ema_throttle = (1 - a) * self._ema_throttle + a * 0.0
                return response_text

            except retry_exceptions as e:
                if attempt == rp.max_retries - 1:
                    raise
                # Prefer Retry-After header if present (seconds)
                headers = self._extract_headers(e)
                retry_after = self._parse_retry_after(headers)
                if headers:
                    try:
                        # Update limiter state from rate headers if available
                        self.rate_limiter.update_from_headers(dict(headers))
                    except Exception:
                        pass

                # Track throttle signals (consecutive + EWMA)
                if self._is_throttle_signal(e, headers):
                    self._consecutive_throttle += 1
                    # Update EWMA towards 1.0 on throttle
                    a = self.storm_policy.ema_alpha
                    self._ema_throttle = (1 - a) * self._ema_throttle + a * 1.0
                    # Open circuit quickly on storms
                    if (
                        self._consecutive_throttle >= self.storm_policy.consecutive_threshold
                        or self._ema_throttle >= self.storm_policy.ema_threshold
                    ):
                        # Minimal cooldown; honor larger Retry-After or reset if provided
                        cooldown = max(
                            self.rate_limit_config.reset_tokens_minute or 0.0,
                            retry_after or 0.0,
                            self.storm_policy.min_cooldown_seconds,
                        )
                        await self.rate_limiter.set_cooldown(cooldown, reason=f"storm: cons={self._consecutive_throttle}, ewma={self._ema_throttle:.2f}")
                else:
                    # On non-429 retryable errors, gently decay EWMA
                    a = self.storm_policy.ema_alpha
                    self._ema_throttle = (1 - a) * self._ema_throttle + a * 0.0

                # Exponential backoff with jitter (fallback or in addition to Retry-After)
                # Base exponential backoff
                base = rp.base_backoff_seconds * (2 ** attempt)
                # Jitter in [base*(1-j), base*(1+j)]
                jitter_span = rp.jitter_factor * base
                backoff_wait = random.uniform(max(0.0, base - jitter_span), base + jitter_span)
                backoff_wait *= (1.0 + self._ema_throttle)  # scale with error pressure
                wait = retry_after if retry_after is not None else backoff_wait
                # Log lower severity for early retries; warn after the third attempt
                if attempt < (rp.warn_after_attempt - 1):
                    logging.debug(f"API error ({type(e).__name__}): retrying in {wait:.1f}s (attempt {attempt+1}/{rp.max_retries})")
                else:
                    logging.warning(f"API error ({type(e).__name__}): retrying in {wait:.1f}s (attempt {attempt+1}/{rp.max_retries})")
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
                            proactive_delay: bool = True,
                            retry_policy: Optional[RetryPolicy] = None,
                            storm_policy: Optional[StormPolicy] = None,
                            pacing_policy: Optional[PacingPolicy] = None) -> AsyncChatCompletionClient:
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
        rate_limit_config=rate_config,
        retry_policy=retry_policy,
        storm_policy=storm_policy,
        pacing_policy=pacing_policy or PacingPolicy(proactive_delay=proactive_delay),
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
