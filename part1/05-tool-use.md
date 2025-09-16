# Chapter 5: Tool Use

*Original content: 20 pages - by Antonio Gulli*

## Brief Description

Tool use is an agentic design pattern where AI systems extend their capabilities by integrating with external tools, APIs, databases, and services. This pattern enables systems to perform actions beyond their native capabilities, such as retrieving real-time information, executing code, manipulating files, or interacting with external systems.

## Introduction

The tool use pattern represents a fundamental shift from isolated AI systems to integrated agents that can interact with the broader digital ecosystem. By enabling AI systems to use external tools, we dramatically expand their problem-solving capabilities and practical utility.

This pattern is inspired by human tool use, where we leverage instruments and technologies to amplify our natural abilities. Similarly, AI agents can use calculators for complex mathematics, databases for information retrieval, APIs for real-time data access, and specialized software for domain-specific tasks.

Tool use is particularly powerful because it allows AI systems to:
- Access up-to-date information beyond their training data
- Perform precise calculations and data processing
- Interact with external systems and services
- Execute code and manipulate digital environments
- Integrate with existing enterprise systems and workflows

The pattern encompasses tool discovery, selection, invocation, result interpretation, and error handling, creating a comprehensive framework for external system integration.

## Key Concepts

### Tool Registration and Discovery
- Defining available tools and their capabilities
- Dynamic tool discovery and capability matching
- Tool metadata management and versioning
- Automated tool registration and updates

### Tool Selection and Planning
- Choosing appropriate tools for specific tasks
- Sequencing tool usage for complex workflows
- Handling tool dependencies and prerequisites
- Optimizing tool selection for efficiency and accuracy

### Tool Invocation and Execution
- Proper parameter formatting and validation
- Secure tool execution with appropriate permissions
- Async and parallel tool execution capabilities
- Result parsing and interpretation

### Error Handling and Fallbacks
- Managing tool failures and unavailability
- Implementing retry logic and circuit breakers
- Providing fallback mechanisms for critical operations
- Error recovery and alternative tool selection

## Implementation

### Basic Tool Framework
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json

class Tool(ABC):
    """Base class for all tools"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name identifier"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable tool description"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters schema"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass

class ToolRegistry:
    """Registry for managing available tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters
            }
            for tool in self.tools.values()
        ]

class ToolExecutor:
    """Handles tool execution and error management"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.execution_history = []

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with error handling"""
        tool = self.registry.get_tool(tool_name)

        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        try:
            # Validate parameters
            self._validate_parameters(tool, parameters)

            # Execute tool
            result = await tool.execute(**parameters)

            # Log execution
            self.execution_history.append({
                'tool': tool_name,
                'parameters': parameters,
                'result': result,
                'success': True
            })

            return result

        except Exception as e:
            # Log error
            self.execution_history.append({
                'tool': tool_name,
                'parameters': parameters,
                'error': str(e),
                'success': False
            })
            raise

    def _validate_parameters(self, tool: Tool, parameters: Dict[str, Any]):
        """Validate tool parameters against schema"""
        # Implementation would validate against tool.parameters schema
        pass
```

### Advanced Tool Management System
```python
class AdvancedToolManager:
    """Advanced tool management with planning and optimization"""

    def __init__(self):
        self.registry = ToolRegistry()
        self.planner = ToolUsagePlanner()
        self.executor = ToolExecutor(self.registry)
        self.cache = ToolResultCache()

    async def execute_tool_plan(self, task_description: str, context: Dict[str, Any]):
        """Execute a complex task using multiple tools"""

        # Generate tool usage plan
        plan = await self.planner.create_plan(
            task_description, self.registry.list_tools(), context
        )

        # Execute plan steps
        results = {}
        for step in plan['steps']:
            step_result = await self._execute_plan_step(step, results)
            results[step['id']] = step_result

        return {
            'plan': plan,
            'results': results,
            'final_result': results.get(plan['final_step_id'])
        }

    async def _execute_plan_step(self, step: Dict[str, Any], previous_results: Dict[str, Any]):
        """Execute a single step in the tool usage plan"""

        # Prepare parameters using previous results
        parameters = self._prepare_step_parameters(step, previous_results)

        # Check cache first
        cache_key = self._generate_cache_key(step['tool'], parameters)
        cached_result = await self.cache.get(cache_key)

        if cached_result:
            return cached_result

        # Execute tool
        result = await self.executor.execute_tool(step['tool'], parameters)

        # Cache result
        await self.cache.set(cache_key, result, ttl=step.get('cache_ttl', 3600))

        return result

    def _prepare_step_parameters(self, step: Dict[str, Any], previous_results: Dict[str, Any]):
        """Prepare parameters for a step, incorporating previous results"""
        parameters = step['parameters'].copy()

        # Substitute references to previous results
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith('${'):
                # Extract reference (e.g., "${step1.result.data}")
                ref = value[2:-1]  # Remove ${ and }
                referenced_value = self._resolve_reference(ref, previous_results)
                parameters[key] = referenced_value

        return parameters

    def _resolve_reference(self, reference: str, results: Dict[str, Any]):
        """Resolve a reference to a previous result"""
        parts = reference.split('.')
        current = results

        for part in parts:
            current = current[part]

        return current
```

## Code Examples

### Example 1: Web Search and Analysis Tools
```python
class WebSearchTool(Tool):
    """Tool for web search functionality"""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information on a given topic"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "Search query",
                "required": True
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5,
                "required": False
            }
        }

    async def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        # Implementation would call actual search API
        search_results = await self._call_search_api(query, num_results)

        return {
            "query": query,
            "results": search_results,
            "num_results": len(search_results)
        }

class URLContentTool(Tool):
    """Tool for extracting content from URLs"""

    @property
    def name(self) -> str:
        return "extract_url_content"

    @property
    def description(self) -> str:
        return "Extract text content from a given URL"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "url": {
                "type": "string",
                "description": "URL to extract content from",
                "required": True
            },
            "extract_type": {
                "type": "string",
                "description": "Type of content to extract",
                "enum": ["text", "html", "structured"],
                "default": "text",
                "required": False
            }
        }

    async def execute(self, url: str, extract_type: str = "text") -> Dict[str, Any]:
        # Implementation would fetch and parse URL content
        content = await self._fetch_url_content(url, extract_type)

        return {
            "url": url,
            "content": content,
            "content_length": len(content),
            "extract_type": extract_type
        }

class ResearchAssistant:
    """Research assistant using multiple tools"""

    def __init__(self):
        self.tool_manager = AdvancedToolManager()

        # Register tools
        self.tool_manager.registry.register_tool(WebSearchTool())
        self.tool_manager.registry.register_tool(URLContentTool())

    async def research_topic(self, topic: str) -> Dict[str, Any]:
        """Conduct research on a topic using multiple tools"""

        task_description = f"""
        Research the topic: {topic}
        1. Search for relevant information
        2. Extract content from top sources
        3. Analyze and summarize findings
        """

        context = {
            "topic": topic,
            "research_depth": "comprehensive"
        }

        return await self.tool_manager.execute_tool_plan(task_description, context)
```

### Example 2: Code Execution and File Management Tools
```python
class CodeExecutionTool(Tool):
    """Tool for executing code safely"""

    @property
    def name(self) -> str:
        return "execute_code"

    @property
    def description(self) -> str:
        return "Execute Python code in a sandboxed environment"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "code": {
                "type": "string",
                "description": "Python code to execute",
                "required": True
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds",
                "default": 30,
                "required": False
            }
        }

    async def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        # Implementation would execute code in sandbox
        result = await self._execute_in_sandbox(code, timeout)

        return {
            "code": code,
            "output": result.get("output", ""),
            "error": result.get("error"),
            "execution_time": result.get("execution_time"),
            "success": result.get("success", False)
        }

class FileManagerTool(Tool):
    """Tool for file operations"""

    @property
    def name(self) -> str:
        return "file_manager"

    @property
    def description(self) -> str:
        return "Perform file operations like read, write, list"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "operation": {
                "type": "string",
                "description": "File operation to perform",
                "enum": ["read", "write", "list", "delete"],
                "required": True
            },
            "path": {
                "type": "string",
                "description": "File or directory path",
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Content for write operations",
                "required": False
            }
        }

    async def execute(self, operation: str, path: str, content: str = None) -> Dict[str, Any]:
        # Implementation would perform file operations securely
        if operation == "read":
            result = await self._read_file(path)
        elif operation == "write":
            result = await self._write_file(path, content)
        elif operation == "list":
            result = await self._list_directory(path)
        elif operation == "delete":
            result = await self._delete_file(path)

        return {
            "operation": operation,
            "path": path,
            "result": result
        }

class DevelopmentAssistant:
    """Development assistant with code and file tools"""

    def __init__(self):
        self.tool_manager = AdvancedToolManager()

        # Register development tools
        self.tool_manager.registry.register_tool(CodeExecutionTool())
        self.tool_manager.registry.register_tool(FileManagerTool())

    async def debug_and_fix_code(self, code_file_path: str, error_description: str):
        """Debug and fix code using tools"""

        # Read the problematic code
        file_content = await self.tool_manager.executor.execute_tool(
            "file_manager",
            {"operation": "read", "path": code_file_path}
        )

        original_code = file_content["result"]

        # Try to execute and identify issues
        execution_result = await self.tool_manager.executor.execute_tool(
            "execute_code",
            {"code": original_code}
        )

        if execution_result["success"]:
            return {"status": "no_issues", "original_code": original_code}

        # Generate fix based on error and description
        fixed_code = await self._generate_fix(
            original_code, execution_result["error"], error_description
        )

        # Test the fix
        test_result = await self.tool_manager.executor.execute_tool(
            "execute_code",
            {"code": fixed_code}
        )

        if test_result["success"]:
            # Save the fixed code
            await self.tool_manager.executor.execute_tool(
                "file_manager",
                {
                    "operation": "write",
                    "path": code_file_path + ".fixed",
                    "content": fixed_code
                }
            )

        return {
            "status": "fixed" if test_result["success"] else "fix_failed",
            "original_code": original_code,
            "fixed_code": fixed_code,
            "test_result": test_result
        }
```

### Example 3: Database and API Integration Tools
```python
class DatabaseTool(Tool):
    """Tool for database operations"""

    @property
    def name(self) -> str:
        return "database_query"

    @property
    def description(self) -> str:
        return "Execute SQL queries against the database"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "SQL query to execute",
                "required": True
            },
            "database": {
                "type": "string",
                "description": "Database name",
                "required": True
            },
            "operation_type": {
                "type": "string",
                "description": "Type of operation",
                "enum": ["select", "insert", "update", "delete"],
                "required": True
            }
        }

    async def execute(self, query: str, database: str, operation_type: str) -> Dict[str, Any]:
        # Implementation would execute database query securely
        if operation_type == "select":
            results = await self._execute_select(query, database)
            return {
                "query": query,
                "results": results,
                "row_count": len(results)
            }
        else:
            affected_rows = await self._execute_modification(query, database)
            return {
                "query": query,
                "affected_rows": affected_rows,
                "operation": operation_type
            }

class APICallTool(Tool):
    """Tool for making API calls"""

    @property
    def name(self) -> str:
        return "api_call"

    @property
    def description(self) -> str:
        return "Make HTTP API calls to external services"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "url": {
                "type": "string",
                "description": "API endpoint URL",
                "required": True
            },
            "method": {
                "type": "string",
                "description": "HTTP method",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "default": "GET",
                "required": False
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers",
                "required": False
            },
            "data": {
                "type": "object",
                "description": "Request body data",
                "required": False
            }
        }

    async def execute(self, url: str, method: str = "GET", headers: Dict = None, data: Dict = None) -> Dict[str, Any]:
        # Implementation would make HTTP request
        response = await self._make_http_request(url, method, headers, data)

        return {
            "url": url,
            "method": method,
            "status_code": response.status_code,
            "response_data": response.data,
            "success": 200 <= response.status_code < 300
        }

class DataAnalysisAgent:
    """Agent for data analysis using database and API tools"""

    def __init__(self):
        self.tool_manager = AdvancedToolManager()

        # Register data tools
        self.tool_manager.registry.register_tool(DatabaseTool())
        self.tool_manager.registry.register_tool(APICallTool())

    async def analyze_sales_performance(self, period: str) -> Dict[str, Any]:
        """Analyze sales performance using database and external API data"""

        # Get sales data from database
        sales_query = f"""
        SELECT product_id, SUM(quantity) as total_quantity, SUM(revenue) as total_revenue
        FROM sales
        WHERE sale_date >= '{period}'
        GROUP BY product_id
        ORDER BY total_revenue DESC
        """

        sales_data = await self.tool_manager.executor.execute_tool(
            "database_query",
            {
                "query": sales_query,
                "database": "sales_db",
                "operation_type": "select"
            }
        )

        # Get market data from external API
        market_data = await self.tool_manager.executor.execute_tool(
            "api_call",
            {
                "url": "https://api.marketdata.com/trends",
                "method": "GET",
                "headers": {"Authorization": "Bearer token"}
            }
        )

        # Combine and analyze data
        analysis = self._combine_sales_and_market_data(
            sales_data["results"],
            market_data["response_data"]
        )

        return {
            "period": period,
            "sales_summary": sales_data["results"][:10],  # Top 10 products
            "market_trends": market_data["response_data"],
            "analysis": analysis
        }
```

## Best Practices

### Tool Design Principles
- **Single Responsibility**: Each tool should have a clear, specific purpose
- **Consistent Interface**: Use standardized parameter and return formats
- **Comprehensive Documentation**: Provide detailed descriptions and examples
- **Error Handling**: Implement robust error reporting and recovery

### Security Considerations
- **Input Validation**: Validate all tool parameters thoroughly
- **Permission Management**: Implement appropriate access controls
- **Sandboxing**: Execute potentially dangerous tools in isolated environments
- **Audit Logging**: Log all tool executions for security monitoring

### Performance Optimization
- **Caching**: Cache tool results where appropriate
- **Parallel Execution**: Execute independent tools concurrently
- **Resource Management**: Monitor and limit resource usage
- **Rate Limiting**: Implement rate limits for external API calls

### Tool Integration
- **Dependency Management**: Handle tool dependencies and prerequisites
- **Version Control**: Manage tool versions and compatibility
- **Discovery**: Implement dynamic tool discovery mechanisms
- **Composition**: Enable tools to work together effectively

## Common Pitfalls

### Tool Proliferation
- **Problem**: Creating too many specialized tools that overlap in functionality
- **Solution**: Consolidate similar tools and create composable tool interfaces
- **Mitigation**: Regular auditing and refactoring of tool collections

### Security Vulnerabilities
- **Problem**: Tools providing unauthorized access to sensitive systems
- **Solution**: Implement comprehensive security controls and access management
- **Mitigation**: Regular security audits and penetration testing

### Tool Dependency Hell
- **Problem**: Complex dependency chains between tools causing brittle systems
- **Solution**: Minimize dependencies and implement graceful degradation
- **Mitigation**: Design tools to be as independent as possible

### Performance Bottlenecks
- **Problem**: Tool execution becoming a system performance bottleneck
- **Solution**: Optimize tool performance and implement caching strategies
- **Mitigation**: Monitor tool performance and implement async execution

### Inconsistent Error Handling
- **Problem**: Different tools handling errors in incompatible ways
- **Solution**: Standardize error reporting and handling across all tools
- **Mitigation**: Implement comprehensive error handling frameworks

### Tool Maintenance Overhead
- **Problem**: Maintaining large numbers of tools becomes unsustainable
- **Solution**: Implement automated testing and maintenance workflows
- **Mitigation**: Design tools for easy maintenance and updates

## Advanced Concepts

### Dynamic Tool Generation
- Automatically creating tools based on API specifications
- Self-modifying tool capabilities based on usage patterns
- AI-generated tools for specific task requirements

### Tool Composition and Chaining
- Creating complex workflows by chaining multiple tools
- Automatic tool composition based on task requirements
- Reusable tool pipeline templates

### Intelligent Tool Selection
- AI-powered tool recommendation based on task analysis
- Learning optimal tool combinations from successful executions
- Context-aware tool selection strategies

### Tool Ecosystem Management
- Managing large-scale tool ecosystems across organizations
- Tool marketplace and sharing mechanisms
- Collaborative tool development and maintenance

## Conclusion

Tool use is a transformative pattern that extends AI capabilities far beyond their native limitations. By providing structured interfaces to external systems, tools enable AI agents to perform practical, real-world tasks while maintaining security and reliability. Success with tool use requires careful attention to security, performance, and maintainability, along with thoughtful design of tool interfaces and execution frameworks. As AI systems become more sophisticated, the tool use pattern will become increasingly critical for building practical, powerful agentic applications.