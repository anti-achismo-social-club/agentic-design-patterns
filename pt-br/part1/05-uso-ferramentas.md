# Capítulo 5: Uso de Ferramentas

*Conteúdo original: 20 páginas - por Antonio Gulli*
*Tradução para PT-BR: Esta tradução visa tornar o conteúdo acessível para desenvolvedores brasileiros, mantendo a precisão técnica do material original.*

## Breve Descrição

Uso de ferramentas é um padrão de design agêntico onde sistemas de IA estendem suas capacidades integrando-se com ferramentas externas, APIs, bancos de dados e serviços. Este padrão permite que sistemas realizem ações além de suas capacidades nativas, como recuperar informações em tempo real, executar código, manipular arquivos ou interagir com sistemas externos.

## Introdução

O padrão de uso de ferramentas representa uma mudança fundamental de sistemas de IA isolados para agentes integrados que podem interagir com o ecossistema digital mais amplo. Ao permitir que sistemas de IA usem ferramentas externas, expandimos dramaticamente suas capacidades de resolução de problemas e utilidade prática.

Este padrão é inspirado pelo uso humano de ferramentas, onde aproveitamos instrumentos e tecnologias para amplificar nossas habilidades naturais. Similarmente, agentes de IA podem usar calculadoras para matemática complexa, bancos de dados para recuperação de informação, APIs para acesso a dados em tempo real e software especializado para tarefas específicas de domínio.

O uso de ferramentas é particularmente poderoso porque permite que sistemas de IA:
- Acessem informação atualizada além de seus dados de treinamento
- Realizem cálculos precisos e processamento de dados
- Interajam com sistemas e serviços externos
- Executem código e manipulem ambientes digitais
- Integrem-se com sistemas empresariais e fluxos de trabalho existentes

O padrão engloba descoberta de ferramentas, seleção, invocação, interpretação de resultados e tratamento de erros, criando um framework abrangente para integração de sistemas externos.

## Conceitos Chave

### Registro e Descoberta de Ferramentas
- Definir ferramentas disponíveis e suas capacidades
- Descoberta dinâmica de ferramentas e correspondência de capacidades
- Gerenciamento de metadados de ferramentas e versionamento
- Registro automatizado e atualizações de ferramentas

### Seleção e Planejamento de Ferramentas
- Escolher ferramentas apropriadas para tarefas específicas
- Sequenciar uso de ferramentas para fluxos de trabalho complexos
- Lidar com dependências e pré-requisitos de ferramentas
- Otimizar seleção de ferramentas para eficiência e precisão

### Invocação e Execução de Ferramentas
- Formatação e validação adequada de parâmetros
- Execução segura de ferramentas com permissões apropriadas
- Capacidades de execução assíncrona e paralela de ferramentas
- Parsing e interpretação de resultados

### Tratamento de Erros e Fallbacks
- Gerenciar falhas e indisponibilidade de ferramentas
- Implementar lógica de retry e circuit breakers
- Fornecer mecanismos de fallback para operações críticas
- Recuperação de erros e seleção de ferramentas alternativas

## Implementação

### Framework Básico de Ferramentas
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json

class Tool(ABC):
    """Classe base para todas as ferramentas"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador do nome da ferramenta"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Descrição legível da ferramenta"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Schema de parâmetros da ferramenta"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Executar a ferramenta com parâmetros dados"""
        pass

class ToolRegistry:
    """Registro para gerenciar ferramentas disponíveis"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """Registrar uma nova ferramenta"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Obter ferramenta por nome"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Listar todas as ferramentas disponíveis"""
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters
            }
            for tool in self.tools.values()
        ]

class ToolExecutor:
    """Trata execução de ferramentas e gerenciamento de erros"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.execution_history = []

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Executar uma ferramenta com tratamento de erros"""
        tool = self.registry.get_tool(tool_name)

        if not tool:
            raise ValueError(f"Ferramenta '{tool_name}' não encontrada")

        try:
            # Validar parâmetros
            self._validate_parameters(tool, parameters)

            # Executar ferramenta
            result = await tool.execute(**parameters)

            # Log da execução
            self.execution_history.append({
                'tool': tool_name,
                'parameters': parameters,
                'result': result,
                'success': True
            })

            return result

        except Exception as e:
            # Log do erro
            self.execution_history.append({
                'tool': tool_name,
                'parameters': parameters,
                'error': str(e),
                'success': False
            })
            raise

    def _validate_parameters(self, tool: Tool, parameters: Dict[str, Any]):
        """Validar parâmetros da ferramenta contra schema"""
        # Implementação validaria contra schema tool.parameters
        pass
```

### Sistema Avançado de Gerenciamento de Ferramentas
```python
class AdvancedToolManager:
    """Gerenciamento avançado de ferramentas com planejamento e otimização"""

    def __init__(self):
        self.registry = ToolRegistry()
        self.planner = ToolUsagePlanner()
        self.executor = ToolExecutor(self.registry)
        self.cache = ToolResultCache()

    async def execute_tool_plan(self, task_description: str, context: Dict[str, Any]):
        """Executar uma tarefa complexa usando múltiplas ferramentas"""

        # Gerar plano de uso de ferramentas
        plan = await self.planner.create_plan(
            task_description, self.registry.list_tools(), context
        )

        # Executar etapas do plano
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
        """Executar uma única etapa no plano de uso de ferramentas"""

        # Preparar parâmetros usando resultados anteriores
        parameters = self._prepare_step_parameters(step, previous_results)

        # Verificar cache primeiro
        cache_key = self._generate_cache_key(step['tool'], parameters)
        cached_result = await self.cache.get(cache_key)

        if cached_result:
            return cached_result

        # Executar ferramenta
        result = await self.executor.execute_tool(step['tool'], parameters)

        # Cache do resultado
        await self.cache.set(cache_key, result, ttl=step.get('cache_ttl', 3600))

        return result

    def _prepare_step_parameters(self, step: Dict[str, Any], previous_results: Dict[str, Any]):
        """Preparar parâmetros para uma etapa, incorporando resultados anteriores"""
        parameters = step['parameters'].copy()

        # Substituir referências a resultados anteriores
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith('${'):
                # Extrair referência (ex: "${step1.result.data}")
                ref = value[2:-1]  # Remover ${ e }
                referenced_value = self._resolve_reference(ref, previous_results)
                parameters[key] = referenced_value

        return parameters

    def _resolve_reference(self, reference: str, results: Dict[str, Any]):
        """Resolver uma referência a um resultado anterior"""
        parts = reference.split('.')
        current = results

        for part in parts:
            current = current[part]

        return current
```

## Exemplos de Código

### Exemplo 1: Ferramentas de Busca Web e Análise
```python
class WebSearchTool(Tool):
    """Ferramenta para funcionalidade de busca web"""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Buscar na web informações sobre um tópico dado"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "Consulta de busca",
                "required": True
            },
            "num_results": {
                "type": "integer",
                "description": "Número de resultados a retornar",
                "default": 5,
                "required": False
            }
        }

    async def execute(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        # Implementação chamaria API de busca real
        search_results = await self._call_search_api(query, num_results)

        return {
            "query": query,
            "results": search_results,
            "num_results": len(search_results)
        }

class URLContentTool(Tool):
    """Ferramenta para extrair conteúdo de URLs"""

    @property
    def name(self) -> str:
        return "extract_url_content"

    @property
    def description(self) -> str:
        return "Extrair conteúdo de texto de uma URL dada"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "url": {
                "type": "string",
                "description": "URL para extrair conteúdo",
                "required": True
            },
            "extract_type": {
                "type": "string",
                "description": "Tipo de conteúdo a extrair",
                "enum": ["text", "html", "structured"],
                "default": "text",
                "required": False
            }
        }

    async def execute(self, url: str, extract_type: str = "text") -> Dict[str, Any]:
        # Implementação buscaria e analisaria conteúdo da URL
        content = await self._fetch_url_content(url, extract_type)

        return {
            "url": url,
            "content": content,
            "content_length": len(content),
            "extract_type": extract_type
        }

class ResearchAssistant:
    """Assistente de pesquisa usando múltiplas ferramentas"""

    def __init__(self):
        self.tool_manager = AdvancedToolManager()

        # Registrar ferramentas
        self.tool_manager.registry.register_tool(WebSearchTool())
        self.tool_manager.registry.register_tool(URLContentTool())

    async def research_topic(self, topic: str) -> Dict[str, Any]:
        """Conduzir pesquisa sobre um tópico usando múltiplas ferramentas"""

        task_description = f"""
        Pesquisar o tópico: {topic}
        1. Buscar informação relevante
        2. Extrair conteúdo das principais fontes
        3. Analisar e resumir descobertas
        """

        context = {
            "topic": topic,
            "research_depth": "comprehensive"
        }

        return await self.tool_manager.execute_tool_plan(task_description, context)
```

### Exemplo 2: Ferramentas de Execução de Código e Gerenciamento de Arquivos
```python
class CodeExecutionTool(Tool):
    """Ferramenta para executar código com segurança"""

    @property
    def name(self) -> str:
        return "execute_code"

    @property
    def description(self) -> str:
        return "Executar código Python em ambiente sandboxed"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "code": {
                "type": "string",
                "description": "Código Python para executar",
                "required": True
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout de execução em segundos",
                "default": 30,
                "required": False
            }
        }

    async def execute(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        # Implementação executaria código em sandbox
        result = await self._execute_in_sandbox(code, timeout)

        return {
            "code": code,
            "output": result.get("output", ""),
            "error": result.get("error"),
            "execution_time": result.get("execution_time"),
            "success": result.get("success", False)
        }

class FileManagerTool(Tool):
    """Ferramenta para operações de arquivo"""

    @property
    def name(self) -> str:
        return "file_manager"

    @property
    def description(self) -> str:
        return "Realizar operações de arquivo como ler, escrever, listar"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "operation": {
                "type": "string",
                "description": "Operação de arquivo a realizar",
                "enum": ["read", "write", "list", "delete"],
                "required": True
            },
            "path": {
                "type": "string",
                "description": "Caminho do arquivo ou diretório",
                "required": True
            },
            "content": {
                "type": "string",
                "description": "Conteúdo para operações de escrita",
                "required": False
            }
        }

    async def execute(self, operation: str, path: str, content: str = None) -> Dict[str, Any]:
        # Implementação realizaria operações de arquivo com segurança
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
    """Assistente de desenvolvimento com ferramentas de código e arquivo"""

    def __init__(self):
        self.tool_manager = AdvancedToolManager()

        # Registrar ferramentas de desenvolvimento
        self.tool_manager.registry.register_tool(CodeExecutionTool())
        self.tool_manager.registry.register_tool(FileManagerTool())

    async def debug_and_fix_code(self, code_file_path: str, error_description: str):
        """Debugar e corrigir código usando ferramentas"""

        # Ler o código problemático
        file_content = await self.tool_manager.executor.execute_tool(
            "file_manager",
            {"operation": "read", "path": code_file_path}
        )

        original_code = file_content["result"]

        # Tentar executar e identificar problemas
        execution_result = await self.tool_manager.executor.execute_tool(
            "execute_code",
            {"code": original_code}
        )

        if execution_result["success"]:
            return {"status": "no_issues", "original_code": original_code}

        # Gerar correção baseada no erro e descrição
        fixed_code = await self._generate_fix(
            original_code, execution_result["error"], error_description
        )

        # Testar a correção
        test_result = await self.tool_manager.executor.execute_tool(
            "execute_code",
            {"code": fixed_code}
        )

        if test_result["success"]:
            # Salvar o código corrigido
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

### Exemplo 3: Ferramentas de Integração com Banco de Dados e API
```python
class DatabaseTool(Tool):
    """Ferramenta para operações de banco de dados"""

    @property
    def name(self) -> str:
        return "database_query"

    @property
    def description(self) -> str:
        return "Executar consultas SQL contra o banco de dados"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "Consulta SQL para executar",
                "required": True
            },
            "database": {
                "type": "string",
                "description": "Nome do banco de dados",
                "required": True
            },
            "operation_type": {
                "type": "string",
                "description": "Tipo de operação",
                "enum": ["select", "insert", "update", "delete"],
                "required": True
            }
        }

    async def execute(self, query: str, database: str, operation_type: str) -> Dict[str, Any]:
        # Implementação executaria consulta de banco de dados com segurança
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
    """Ferramenta para fazer chamadas de API"""

    @property
    def name(self) -> str:
        return "api_call"

    @property
    def description(self) -> str:
        return "Fazer chamadas HTTP API para serviços externos"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "url": {
                "type": "string",
                "description": "URL do endpoint da API",
                "required": True
            },
            "method": {
                "type": "string",
                "description": "Método HTTP",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "default": "GET",
                "required": False
            },
            "headers": {
                "type": "object",
                "description": "Cabeçalhos HTTP",
                "required": False
            },
            "data": {
                "type": "object",
                "description": "Dados do corpo da requisição",
                "required": False
            }
        }

    async def execute(self, url: str, method: str = "GET", headers: Dict = None, data: Dict = None) -> Dict[str, Any]:
        # Implementação faria requisição HTTP
        response = await self._make_http_request(url, method, headers, data)

        return {
            "url": url,
            "method": method,
            "status_code": response.status_code,
            "response_data": response.data,
            "success": 200 <= response.status_code < 300
        }

class DataAnalysisAgent:
    """Agente para análise de dados usando ferramentas de banco de dados e API"""

    def __init__(self):
        self.tool_manager = AdvancedToolManager()

        # Registrar ferramentas de dados
        self.tool_manager.registry.register_tool(DatabaseTool())
        self.tool_manager.registry.register_tool(APICallTool())

    async def analyze_sales_performance(self, period: str) -> Dict[str, Any]:
        """Analisar performance de vendas usando dados de banco de dados e API externa"""

        # Obter dados de vendas do banco de dados
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

        # Obter dados de mercado de API externa
        market_data = await self.tool_manager.executor.execute_tool(
            "api_call",
            {
                "url": "https://api.marketdata.com/trends",
                "method": "GET",
                "headers": {"Authorization": "Bearer token"}
            }
        )

        # Combinar e analisar dados
        analysis = self._combine_sales_and_market_data(
            sales_data["results"],
            market_data["response_data"]
        )

        return {
            "period": period,
            "sales_summary": sales_data["results"][:10],  # Top 10 produtos
            "market_trends": market_data["response_data"],
            "analysis": analysis
        }
```

## Melhores Práticas

### Princípios de Design de Ferramentas
- **Responsabilidade Única**: Cada ferramenta deve ter um propósito claro e específico
- **Interface Consistente**: Usar formatos padronizados de parâmetros e retorno
- **Documentação Abrangente**: Fornecer descrições detalhadas e exemplos
- **Tratamento de Erros**: Implementar relatório robusto de erros e recuperação

### Considerações de Segurança
- **Validação de Entrada**: Validar todos os parâmetros de ferramenta completamente
- **Gerenciamento de Permissões**: Implementar controles de acesso apropriados
- **Sandboxing**: Executar ferramentas potencialmente perigosas em ambientes isolados
- **Log de Auditoria**: Registrar todas as execuções de ferramentas para monitoramento de segurança

### Otimização de Performance
- **Cache**: Cache de resultados de ferramentas quando apropriado
- **Execução Paralela**: Executar ferramentas independentes concorrentemente
- **Gerenciamento de Recursos**: Monitorar e limitar uso de recursos
- **Rate Limiting**: Implementar limites de taxa para chamadas de API externas

### Integração de Ferramentas
- **Gerenciamento de Dependências**: Lidar com dependências e pré-requisitos de ferramentas
- **Controle de Versão**: Gerenciar versões e compatibilidade de ferramentas
- **Descoberta**: Implementar mecanismos dinâmicos de descoberta de ferramentas
- **Composição**: Permitir que ferramentas trabalhem juntas efetivamente

## Armadilhas Comuns

### Proliferação de Ferramentas
- **Problema**: Criar muitas ferramentas especializadas que se sobrepõem em funcionalidade
- **Solução**: Consolidar ferramentas similares e criar interfaces composáveis de ferramentas
- **Mitigação**: Auditoria regular e refatoração de coleções de ferramentas

### Vulnerabilidades de Segurança
- **Problema**: Ferramentas fornecendo acesso não autorizado a sistemas sensíveis
- **Solução**: Implementar controles abrangentes de segurança e gerenciamento de acesso
- **Mitigação**: Auditorias regulares de segurança e testes de penetração

### Inferno de Dependências de Ferramentas
- **Problema**: Cadeias complexas de dependência entre ferramentas causando sistemas frágeis
- **Solução**: Minimizar dependências e implementar degradação graceful
- **Mitigação**: Projetar ferramentas para serem o mais independentes possível

### Gargalos de Performance
- **Problema**: Execução de ferramentas se tornando um gargalo de performance do sistema
- **Solução**: Otimizar performance de ferramentas e implementar estratégias de cache
- **Mitigação**: Monitorar performance de ferramentas e implementar execução assíncrona

### Tratamento Inconsistente de Erros
- **Problema**: Diferentes ferramentas lidando com erros de maneiras incompatíveis
- **Solução**: Padronizar relatório e tratamento de erros entre todas as ferramentas
- **Mitigação**: Implementar frameworks abrangentes de tratamento de erros

### Overhead de Manutenção de Ferramentas
- **Problema**: Manter grandes números de ferramentas se torna insustentável
- **Solução**: Implementar fluxos de trabalho automatizados de teste e manutenção
- **Mitigação**: Projetar ferramentas para fácil manutenção e atualizações

## Conceitos Avançados

### Geração Dinâmica de Ferramentas
- Criar automaticamente ferramentas baseadas em especificações de API
- Capacidades de ferramentas auto-modificadoras baseadas em padrões de uso
- Ferramentas geradas por IA para requisitos específicos de tarefas

### Composição e Encadeamento de Ferramentas
- Criar fluxos de trabalho complexos encadeando múltiplas ferramentas
- Composição automática de ferramentas baseada em requisitos de tarefas
- Templates reutilizáveis de pipeline de ferramentas

### Seleção Inteligente de Ferramentas
- Recomendação de ferramentas baseada em IA através de análise de tarefas
- Aprender combinações ótimas de ferramentas de execuções bem-sucedidas
- Estratégias de seleção de ferramentas conscientes de contexto

### Gerenciamento de Ecossistema de Ferramentas
- Gerenciar ecossistemas de ferramentas em larga escala entre organizações
- Marketplace de ferramentas e mecanismos de compartilhamento
- Desenvolvimento e manutenção colaborativos de ferramentas

## Conclusão

Uso de ferramentas é um padrão transformativo que estende capacidades de IA muito além de suas limitações nativas. Ao fornecer interfaces estruturadas para sistemas externos, ferramentas permitem que agentes de IA realizem tarefas práticas do mundo real mantendo segurança e confiabilidade. O sucesso com uso de ferramentas requer atenção cuidadosa à segurança, performance e manutenibilidade, junto com design cuidadoso de interfaces de ferramentas e frameworks de execução. À medida que sistemas de IA se tornam mais sofisticados, o padrão de uso de ferramentas se tornará cada vez mais crítico para construir aplicações agênticas práticas e poderosas.