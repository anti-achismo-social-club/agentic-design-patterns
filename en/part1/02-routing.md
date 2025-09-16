# Chapter 2: Routing

*Original content: 13 pages - by Antonio Gulli*

## Brief Description

Routing is an agentic design pattern that directs user queries or inputs to the most appropriate handler, model, or processing pathway based on the content, intent, or characteristics of the request. This pattern enables efficient resource utilization and specialized processing for different types of tasks.

## Introduction

In complex agentic systems, not all tasks require the same type of processing or expertise. The routing pattern addresses this challenge by implementing intelligent decision-making mechanisms that direct requests to the most suitable processing pipeline, model, or agent.

Routing can occur at multiple levels: from simple keyword-based routing to sophisticated semantic understanding that determines the best approach for handling a specific request. This pattern is essential for building scalable systems that can handle diverse user needs while optimizing performance and resource usage.

## Key Concepts

### Intent Classification
- Analyzing user input to determine the underlying intent or purpose
- Mapping intents to appropriate processing pathways
- Handling ambiguous or multi-intent scenarios

### Capability Matching
- Matching request characteristics with available system capabilities
- Routing based on required expertise, tools, or knowledge domains
- Dynamic capability discovery and selection

### Load Balancing
- Distributing requests across multiple processing units
- Optimizing resource utilization and response times
- Implementing failover and redundancy mechanisms

### Context-Aware Routing
- Considering user context, history, and preferences
- Adapting routing decisions based on system state
- Implementing personalized routing strategies

## Implementation

### Basic Router Architecture
```python
class Router:
    def __init__(self):
        self.routes = {}
        self.default_handler = None

    def register_route(self, pattern, handler):
        self.routes[pattern] = handler

    def route(self, request):
        for pattern, handler in self.routes.items():
            if self.matches(request, pattern):
                return handler

        return self.default_handler
```

### Intelligent Routing System
```python
class IntelligentRouter:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.handlers = {}
        self.load_balancer = LoadBalancer()

    def route_request(self, request):
        # Classify intent
        intent = self.classifier.classify(request)

        # Get available handlers
        handlers = self.handlers.get(intent, [])

        # Select optimal handler
        selected_handler = self.load_balancer.select(handlers)

        return selected_handler.process(request)
```

## Code Examples

### Example 1: Multi-Domain Customer Service Router
```python
class CustomerServiceRouter:
    def __init__(self):
        self.technical_keywords = ['error', 'bug', 'not working', 'crash']
        self.billing_keywords = ['payment', 'charge', 'refund', 'invoice']
        self.general_keywords = ['account', 'password', 'login']

    def route_inquiry(self, customer_message):
        message_lower = customer_message.lower()

        # Technical issues
        if any(keyword in message_lower for keyword in self.technical_keywords):
            return self.route_to_technical_support(customer_message)

        # Billing issues
        elif any(keyword in message_lower for keyword in self.billing_keywords):
            return self.route_to_billing_department(customer_message)

        # General inquiries
        elif any(keyword in message_lower for keyword in self.general_keywords):
            return self.route_to_general_support(customer_message)

        # Fallback to human agent
        else:
            return self.route_to_human_agent(customer_message)

    def route_to_technical_support(self, message):
        return TechnicalSupportAgent().handle(message)

    def route_to_billing_department(self, message):
        return BillingAgent().handle(message)

    def route_to_general_support(self, message):
        return GeneralSupportAgent().handle(message)

    def route_to_human_agent(self, message):
        return HumanAgentQueue().add(message)
```

### Example 2: LLM Model Router
```python
class ModelRouter:
    def __init__(self):
        self.models = {
            'reasoning': ['gpt-4', 'claude-3'],
            'creative': ['gpt-3.5-turbo', 'llama-2'],
            'coding': ['codex', 'code-llama'],
            'summarization': ['distilbert', 'bart']
        }

    def classify_task(self, prompt):
        # Simple heuristic-based classification
        if any(word in prompt.lower() for word in ['solve', 'calculate', 'analyze']):
            return 'reasoning'
        elif any(word in prompt.lower() for word in ['write', 'create', 'story']):
            return 'creative'
        elif any(word in prompt.lower() for word in ['code', 'function', 'debug']):
            return 'coding'
        elif any(word in prompt.lower() for word in ['summarize', 'tldr', 'brief']):
            return 'summarization'
        else:
            return 'reasoning'  # default

    def route_to_model(self, prompt):
        task_type = self.classify_task(prompt)
        available_models = self.models[task_type]

        # Select best available model (could include load balancing)
        selected_model = self.select_optimal_model(available_models)

        return self.call_model(selected_model, prompt)

    def select_optimal_model(self, models):
        # Simple selection - in practice, consider load, cost, performance
        return models[0]

    def call_model(self, model, prompt):
        # Implementation depends on model API
        pass
```

### Example 3: Semantic Routing with Embeddings
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRouter:
    def __init__(self):
        self.route_embeddings = {}
        self.handlers = {}

    def register_semantic_route(self, description, handler):
        embedding = self.get_embedding(description)
        route_id = len(self.route_embeddings)
        self.route_embeddings[route_id] = embedding
        self.handlers[route_id] = handler

    def route(self, query):
        query_embedding = self.get_embedding(query)

        # Find most similar route
        similarities = {}
        for route_id, route_embedding in self.route_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                route_embedding.reshape(1, -1)
            )[0][0]
            similarities[route_id] = similarity

        # Select best match
        best_route = max(similarities, key=similarities.get)

        # Apply threshold for confidence
        if similarities[best_route] > 0.7:
            return self.handlers[best_route]
        else:
            return self.default_handler

    def get_embedding(self, text):
        # Implementation depends on embedding service
        pass
```

## Best Practices

### Router Design
- **Clear Decision Logic**: Make routing decisions transparent and auditable
- **Fallback Mechanisms**: Always provide default handlers for unmatched cases
- **Performance Optimization**: Minimize routing overhead and decision time
- **Scalability**: Design routers that can handle increasing numbers of routes and requests

### Intent Classification
- **Training Data Quality**: Use diverse, representative training data
- **Regular Updates**: Continuously improve classification accuracy
- **Multi-Intent Handling**: Support requests with multiple or ambiguous intents
- **Confidence Thresholds**: Implement confidence-based routing decisions

### Monitoring and Analytics
- **Route Performance**: Track success rates and response times for each route
- **Usage Patterns**: Analyze routing patterns to optimize system design
- **Error Tracking**: Monitor and alert on routing failures
- **A/B Testing**: Test different routing strategies and configurations

## Common Pitfalls

### Over-Complex Routing Logic
- **Problem**: Creating overly sophisticated routing systems for simple use cases
- **Solution**: Start with simple rules and add complexity incrementally
- **Mitigation**: Regular review and simplification of routing logic

### Misclassification Cascades
- **Problem**: Incorrect routing leads to poor user experiences
- **Solution**: Implement confidence thresholds and human oversight
- **Mitigation**: Provide easy mechanisms for users to report routing errors

### Route Proliferation
- **Problem**: Too many specific routes make the system hard to maintain
- **Solution**: Consolidate similar routes and use hierarchical routing
- **Mitigation**: Regular auditing and cleanup of unused or redundant routes

### Performance Bottlenecks
- **Problem**: Routing decisions become a system bottleneck
- **Solution**: Optimize routing algorithms and implement caching
- **Mitigation**: Profile routing performance and set performance targets

### Inconsistent User Experience
- **Problem**: Different routes provide varying quality of service
- **Solution**: Standardize interfaces and quality metrics across routes
- **Mitigation**: Implement quality monitoring and feedback mechanisms

### Missing Fallbacks
- **Problem**: System fails when no route matches the request
- **Solution**: Always implement comprehensive fallback mechanisms
- **Mitigation**: Test edge cases and unusual inputs regularly

## Advanced Concepts

### Dynamic Route Learning
- Automatically discovering new routing patterns from user behavior
- Adapting routing strategies based on success metrics
- Machine learning-based route optimization

### Multi-Stage Routing
- Hierarchical routing with multiple decision points
- Progressive refinement of routing decisions
- Context-dependent sub-routing

### Collaborative Routing
- Multiple agents collaborating on routing decisions
- Consensus-based routing for critical decisions
- Distributed routing across multiple systems

## Conclusion

Routing is a critical pattern for building scalable and efficient agentic systems. By intelligently directing requests to appropriate handlers, routing enables systems to provide specialized, high-quality responses while optimizing resource utilization. Success with routing requires careful attention to classification accuracy, performance optimization, and comprehensive fallback mechanisms.