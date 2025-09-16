# Capítulo 12: Tratamento de Exceções e Recuperação

Um padrão abrangente para gerenciar erros, falhas e situações inesperadas em sistemas de agentes de IA através de mecanismos robustos de tratamento de exceções e estratégias de recuperação.

## Introdução

Padrões de tratamento de exceções e recuperação fornecem abordagens sistemáticas para gerenciar erros, falhas e situações inesperadas que inevitavelmente ocorrem em sistemas de agentes de IA. Esses padrões garantem resiliência do sistema, degradação graciosa e capacidades de recuperação automática, mantendo a experiência do usuário e a confiabilidade do sistema.

Agentes de IA modernos operam em ambientes complexos e distribuídos onde falhas podem ocorrer em múltiplos níveis - desde problemas de conectividade de rede e limites de taxa de API até erros de inferência de modelo e falhas de processamento de dados. Sem tratamento adequado de exceções, essas falhas podem cascatear através do sistema, causando quebras completas e experiências ruins para o usuário.

O padrão de tratamento de exceções abrange múltiplas estratégias incluindo detecção de erro, classificação, contenção, mecanismos de recuperação e aprendizado com falhas para prevenir ocorrências futuras.

## Conceitos-Chave

### Classificação de Erros
Compreender diferentes tipos de erros e suas estratégias de tratamento apropriadas:

- **Erros Transitórios**: Falhas temporárias que podem se resolver automaticamente (timeouts de rede, limites de taxa)
- **Erros Persistentes**: Falhas consistentes que requerem intervenção (falhas de autenticação, recursos ausentes)
- **Erros Críticos**: Falhas que ameaçam o sistema e requerem atenção imediata (esgotamento de memória, violações de segurança)
- **Erros Recuperáveis**: Falhas com estratégias de recuperação conhecidas (quedas de conexão, indisponibilidade de serviço)

### Estratégias de Recuperação
Múltiplas abordagens para recuperar-se de diferentes tipos de falhas:

- **Mecanismos de Retry**: Retry automático com backoff exponencial para falhas transitórias
- **Sistemas de Fallback**: Caminhos alternativos quando sistemas primários falham
- **Circuit Breakers**: Prevenir falhas em cascata desabilitando temporariamente serviços com falha
- **Degradação Graciosa**: Manter funcionalidade principal enquanto recursos não essenciais são desabilitados

### Propagação de Erros
Gerenciar como erros fluem através do sistema:

- **Isolamento de Erros**: Conter falhas para prevenir impactos em todo o sistema
- **Agregação de Erros**: Coletar e analisar padrões de erro para melhorias sistemáticas
- **Comunicação com Usuário**: Fornecer mensagens de erro significativas e orientação de recuperação

## Implementação

### Estrutura Básica do Manipulador de Exceções

```python
class AgentExceptionHandler:
    def __init__(self):
        self.error_strategies = {}
        self.retry_configs = {}
        self.circuit_breakers = {}
        self.fallback_handlers = {}

    def register_strategy(self, error_type, strategy):
        """Registrar estratégia de tratamento para tipos específicos de erro"""
        self.error_strategies[error_type] = strategy

    def handle_exception(self, exception, context):
        """Ponto de entrada principal para tratamento de exceções"""
        error_type = type(exception)

        if error_type in self.error_strategies:
            return self.error_strategies[error_type](exception, context)

        return self.default_handler(exception, context)
```

### Mecanismo de Retry com Backoff Exponencial

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
        """Executar função com retry de backoff exponencial"""
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

### Padrão Circuit Breaker

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
        """Executar função através do circuit breaker"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker está ABERTO")

        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except self.expected_exception as e:
            self.on_failure()
            raise e

    def on_success(self):
        """Resetar circuit breaker em execução bem-sucedida"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def on_failure(self):
        """Lidar com falha no circuit breaker"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

## Exemplos de Código

### Agente de IA Abrangente com Tratamento de Exceções

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
        """Configurar estratégias de tratamento de erro"""
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
        """Processamento principal de requisição com tratamento abrangente de erros"""
        context = ErrorContext(
            agent_id=self.agent_id,
            operation="process_request",
            timestamp=time.time(),
            user_id=request.get("user_id"),
            session_id=request.get("session_id")
        )

        try:
            # Caminho de processamento primário
            return await self._execute_primary_processing(request, context)
        except Exception as e:
            return await self.exception_handler.handle_exception(e, context)

    async def _execute_primary_processing(self, request, context):
        """Executar lógica de processamento primário com proteção de circuit breaker"""
        service_name = "llm_inference"

        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()

        circuit_breaker = self.circuit_breakers[service_name]

        return await circuit_breaker.call(
            self._perform_llm_inference, request, context
        )

    async def _perform_llm_inference(self, request, context):
        """Realizar inferência LLM com tratamento de retry"""
        return await self.retry_handler.execute_with_retry(
            self._call_llm_api, request
        )

    async def _handle_connection_error(self, exception, context):
        """Tratar erros relacionados à conexão"""
        self.logger.warning(f"Erro de conexão em {context.operation}: {exception}")

        # Tentar serviço de fallback
        if "llm_fallback" in self.fallback_strategies:
            return await self.fallback_strategies["llm_fallback"](context)

        return {
            "error": "service_unavailable",
            "message": "Não é possível processar requisição devido a problemas de serviço",
            "retry_after": 30
        }

    async def _handle_timeout_error(self, exception, context):
        """Tratar erros de timeout"""
        self.logger.warning(f"Timeout em {context.operation}: {exception}")

        return {
            "error": "timeout",
            "message": "Requisição expirou, tente novamente",
            "suggestion": "Considere simplificar sua requisição"
        }

    async def _handle_validation_error(self, exception, context):
        """Tratar erros de validação"""
        self.logger.info(f"Erro de validação em {context.operation}: {exception}")

        return {
            "error": "invalid_input",
            "message": str(exception),
            "suggestion": "Verifique sua entrada e tente novamente"
        }
```

### Sistema de Recuperação de Erros e Aprendizado

```python
class ErrorRecoverySystem:
    def __init__(self):
        self.error_history = []
        self.recovery_patterns = {}
        self.success_rates = {}

    def record_error(self, error_type, context, recovery_action, success):
        """Registrar ocorrência de erro e tentativa de recuperação"""
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
        """Atualizar taxas de sucesso para estratégias de recuperação"""
        key = (error_type, recovery_action)

        if key not in self.success_rates:
            self.success_rates[key] = {"attempts": 0, "successes": 0}

        self.success_rates[key]["attempts"] += 1
        if success:
            self.success_rates[key]["successes"] += 1

    def get_best_recovery_strategy(self, error_type):
        """Obter a estratégia de recuperação mais bem-sucedida para um tipo de erro"""
        strategies = [
            (action, rates) for (err_type, action), rates in self.success_rates.items()
            if err_type == error_type and rates["attempts"] >= 3
        ]

        if not strategies:
            return None

        return max(strategies, key=lambda x: x[1]["successes"] / x[1]["attempts"])[0]
```

## Melhores Práticas

### Design de Tratamento de Erros
- **Fail Fast**: Detectar e tratar erros o mais cedo possível no pipeline de processamento
- **Degradação Graciosa**: Manter funcionalidade principal mesmo quando componentes não essenciais falham
- **Mensagens Centradas no Usuário**: Fornecer mensagens de erro claras e acionáveis aos usuários
- **Log Abrangente**: Registrar informações detalhadas de erro para depuração e análise

### Implementação de Estratégia de Recuperação
- **Recuperação em Camadas**: Implementar múltiplos níveis de recuperação desde retries imediatos até fallbacks completos
- **Gerenciamento de Recursos**: Garantir limpeza adequada de recursos durante condições de erro
- **Consistência de Estado**: Manter consistência do estado do sistema durante e após recuperação de erro
- **Monitoramento de Desempenho**: Rastrear taxas de erro e desempenho de recuperação para identificar padrões

### Resiliência do Sistema
- **Ajuste de Circuit Breaker**: Configurar limiares e timeouts apropriados baseados nas características do serviço
- **Otimização de Estratégia de Retry**: Usar backoff exponencial com jitter para evitar problemas de thundering herd
- **Gerenciamento de Serviço de Fallback**: Manter e testar serviços de fallback regularmente
- **Correlação de Erros**: Analisar padrões de erro para identificar problemas sistêmicos

## Armadilhas Comuns

### Retries Excessivamente Agressivos
Implementar mecanismos de retry sem backoff adequado e limites pode sobrecarregar serviços com falha e piorar a situação. Sempre implemente backoff exponencial com limites máximos de retry e considere o impacto downstream dos retries.

### Contexto de Erro Insuficiente
Capturar exceções sem preservar contexto suficiente torna a depuração e recuperação extremamente difíceis. Sempre capture informações de contexto relevantes incluindo estado do usuário, estado do sistema e detalhes da operação.

### Falhas Silenciosas
Suprimir erros sem log adequado ou notificação ao usuário pode levar à degradação invisível do sistema. Garanta que todos os erros sejam adequadamente registrados e comunicados.

### Configuração Inadequada de Circuit Breaker
Circuit breakers mal configurados podem ou falhar em proteger o sistema (limiares muito altos) ou causar interrupções desnecessárias de serviço (limiares muito baixos). Monitore e ajuste parâmetros de circuit breaker baseados no comportamento real do serviço.

### Teste de Estratégia de Recuperação
Falhar em testar regularmente mecanismos de recuperação significa que eles podem não funcionar quando realmente necessários. Implemente práticas de chaos engineering para testar regularmente sistemas de tratamento de erro e recuperação.

### Segurança de Mensagem de Erro
Expor informações sensíveis em mensagens de erro pode criar vulnerabilidades de segurança. Garanta que mensagens de erro sejam sanitizadas e não revelem detalhes internos do sistema ou dados sensíveis.

---

*Este capítulo cobre 8 páginas de conteúdo de "Agentic Design Patterns" por Antonio Gulli, focando em construir sistemas de agentes de IA resilientes através de mecanismos abrangentes de tratamento de exceções e recuperação.*

---

*Nota de Tradução: Este capítulo foi traduzido do inglês para o português brasileiro. Alguns termos técnicos podem ter múltiplas traduções aceitas na literatura em português.*