# Chapter 12: Exception Handling and Recovery

A comprehensive pattern for managing errors, failures, and unexpected situations in AI agent systems through robust exception handling mechanisms and recovery strategies.

## Introduction

Exception handling and recovery patterns provide systematic approaches for managing errors, failures, and unexpected situations that inevitably occur in AI agent systems. These patterns ensure system resilience, graceful degradation, and automatic recovery capabilities while maintaining user experience and system reliability.

Modern AI agents operate in complex, distributed environments where failures can occur at multiple levels - from network connectivity issues and API rate limits to model inference errors and data processing failures. Without proper exception handling, these failures can cascade through the system, causing complete breakdowns and poor user experiences.

The exception handling pattern encompasses multiple strategies including error detection, classification, containment, recovery mechanisms, and learning from failures to prevent future occurrences.

## Key Concepts

### Error Classification
Understanding different types of errors and their appropriate handling strategies:

- **Transient Errors**: Temporary failures that may resolve automatically (network timeouts, rate limits)
- **Persistent Errors**: Consistent failures requiring intervention (authentication failures, missing resources)
- **Critical Errors**: System-threatening failures requiring immediate attention (memory exhaustion, security breaches)
- **Recoverable Errors**: Failures with known recovery strategies (connection drops, service unavailability)

### Recovery Strategies
Multiple approaches for recovering from different types of failures:

- **Retry Mechanisms**: Automatic retry with exponential backoff for transient failures
- **Fallback Systems**: Alternative pathways when primary systems fail
- **Circuit Breakers**: Preventing cascade failures by temporarily disabling failing services
- **Graceful Degradation**: Maintaining core functionality while non-essential features are disabled

### Error Propagation
Managing how errors flow through the system:

- **Error Isolation**: Containing failures to prevent system-wide impacts
- **Error Aggregation**: Collecting and analyzing error patterns for systematic improvements
- **User Communication**: Providing meaningful error messages and recovery guidance

## Implementation

### Basic Exception Handler Structure

```python
class AgentExceptionHandler:
    def __init__(self):
        self.error_strategies = {}
        self.retry_configs = {}
        self.circuit_breakers = {}
        self.fallback_handlers = {}

    def register_strategy(self, error_type, strategy):
        """Register handling strategy for specific error types"""
        self.error_strategies[error_type] = strategy

    def handle_exception(self, exception, context):
        """Main exception handling entry point"""
        error_type = type(exception)

        if error_type in self.error_strategies:
            return self.error_strategies[error_type](exception, context)

        return self.default_handler(exception, context)
```

### Retry Mechanism with Exponential Backoff

```python
import asyncio
import random
from typing import Callable, Any

class RetryHandler:
    def __init__(self, max_retries=3, base_delay=1.0, max_delay=60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry"""
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    raise e

                delay = min(
                    self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.max_delay
                )
                await asyncio.sleep(delay)
```

### Circuit Breaker Pattern

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except self.expected_exception as e:
            self.on_failure()
            raise e

    def on_success(self):
        """Reset circuit breaker on successful execution"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def on_failure(self):
        """Handle failure in circuit breaker"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

## Code Examples

### Comprehensive AI Agent with Exception Handling

```python
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ErrorContext:
    agent_id: str
    operation: str
    timestamp: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ResilientAIAgent:
    def __init__(self):
        self.exception_handler = AgentExceptionHandler()
        self.retry_handler = RetryHandler()
        self.circuit_breakers = {}
        self.fallback_strategies = {}
        self.logger = logging.getLogger(__name__)

        self._setup_error_strategies()

    def _setup_error_strategies(self):
        """Configure error handling strategies"""
        self.exception_handler.register_strategy(
            ConnectionError,
            self._handle_connection_error
        )
        self.exception_handler.register_strategy(
            TimeoutError,
            self._handle_timeout_error
        )
        self.exception_handler.register_strategy(
            ValueError,
            self._handle_validation_error
        )

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main request processing with comprehensive error handling"""
        context = ErrorContext(
            agent_id=self.agent_id,
            operation="process_request",
            timestamp=time.time(),
            user_id=request.get("user_id"),
            session_id=request.get("session_id")
        )

        try:
            # Primary processing pathway
            return await self._execute_primary_processing(request, context)
        except Exception as e:
            return await self.exception_handler.handle_exception(e, context)

    async def _execute_primary_processing(self, request, context):
        """Execute primary processing logic with circuit breaker protection"""
        service_name = "llm_inference"

        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()

        circuit_breaker = self.circuit_breakers[service_name]

        return await circuit_breaker.call(
            self._perform_llm_inference, request, context
        )

    async def _perform_llm_inference(self, request, context):
        """Perform LLM inference with retry handling"""
        return await self.retry_handler.execute_with_retry(
            self._call_llm_api, request
        )

    async def _handle_connection_error(self, exception, context):
        """Handle connection-related errors"""
        self.logger.warning(f"Connection error in {context.operation}: {exception}")

        # Try fallback service
        if "llm_fallback" in self.fallback_strategies:
            return await self.fallback_strategies["llm_fallback"](context)

        return {
            "error": "service_unavailable",
            "message": "Unable to process request due to service issues",
            "retry_after": 30
        }

    async def _handle_timeout_error(self, exception, context):
        """Handle timeout errors"""
        self.logger.warning(f"Timeout in {context.operation}: {exception}")

        return {
            "error": "timeout",
            "message": "Request timed out, please try again",
            "suggestion": "Consider simplifying your request"
        }

    async def _handle_validation_error(self, exception, context):
        """Handle validation errors"""
        self.logger.info(f"Validation error in {context.operation}: {exception}")

        return {
            "error": "invalid_input",
            "message": str(exception),
            "suggestion": "Please check your input and try again"
        }
```

### Error Recovery and Learning System

```python
class ErrorRecoverySystem:
    def __init__(self):
        self.error_history = []
        self.recovery_patterns = {}
        self.success_rates = {}

    def record_error(self, error_type, context, recovery_action, success):
        """Record error occurrence and recovery attempt"""
        error_record = {
            "error_type": error_type,
            "context": context,
            "recovery_action": recovery_action,
            "success": success,
            "timestamp": time.time()
        }

        self.error_history.append(error_record)
        self._update_success_rates(error_type, recovery_action, success)

    def _update_success_rates(self, error_type, recovery_action, success):
        """Update success rates for recovery strategies"""
        key = (error_type, recovery_action)

        if key not in self.success_rates:
            self.success_rates[key] = {"attempts": 0, "successes": 0}

        self.success_rates[key]["attempts"] += 1
        if success:
            self.success_rates[key]["successes"] += 1

    def get_best_recovery_strategy(self, error_type):
        """Get the most successful recovery strategy for an error type"""
        strategies = [
            (action, rates) for (err_type, action), rates in self.success_rates.items()
            if err_type == error_type and rates["attempts"] >= 3
        ]

        if not strategies:
            return None

        return max(strategies, key=lambda x: x[1]["successes"] / x[1]["attempts"])[0]
```

## Best Practices

### Error Handling Design
- **Fail Fast**: Detect and handle errors as early as possible in the processing pipeline
- **Graceful Degradation**: Maintain core functionality even when non-essential components fail
- **User-Centric Messages**: Provide clear, actionable error messages to users
- **Comprehensive Logging**: Log detailed error information for debugging and analysis

### Recovery Strategy Implementation
- **Layered Recovery**: Implement multiple levels of recovery from immediate retries to complete fallbacks
- **Resource Management**: Ensure proper cleanup of resources during error conditions
- **State Consistency**: Maintain system state consistency during and after error recovery
- **Performance Monitoring**: Track error rates and recovery performance to identify patterns

### System Resilience
- **Circuit Breaker Tuning**: Configure appropriate thresholds and timeouts based on service characteristics
- **Retry Strategy Optimization**: Use exponential backoff with jitter to avoid thundering herd problems
- **Fallback Service Management**: Maintain and test fallback services regularly
- **Error Correlation**: Analyze error patterns to identify systemic issues

## Common Pitfalls

### Over-Aggressive Retries
Implementing retry mechanisms without proper backoff and limits can overwhelm failing services and worsen the situation. Always implement exponential backoff with maximum retry limits and consider the downstream impact of retries.

### Insufficient Error Context
Catching exceptions without preserving sufficient context makes debugging and recovery extremely difficult. Always capture relevant context information including user state, system state, and operation details.

### Silent Failures
Suppressing errors without proper logging or user notification can lead to invisible system degradation. Ensure all errors are appropriately logged and communicated.

### Circuit Breaker Misconfiguration
Poorly configured circuit breakers can either fail to protect the system (thresholds too high) or cause unnecessary service disruptions (thresholds too low). Monitor and tune circuit breaker parameters based on actual service behavior.

### Recovery Strategy Testing
Failing to regularly test recovery mechanisms means they may not work when actually needed. Implement chaos engineering practices to regularly test error handling and recovery systems.

### Error Message Security
Exposing sensitive information in error messages can create security vulnerabilities. Ensure error messages are sanitized and don't reveal internal system details or sensitive data.

---

*This chapter covers 8 pages of content from "Agentic Design Patterns" by Antonio Gulli, focusing on building resilient AI agent systems through comprehensive exception handling and recovery mechanisms.*