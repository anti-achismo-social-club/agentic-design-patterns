# Capítulo 2: Roteamento

*Conteúdo original: 13 páginas - por Antonio Gulli*
*Tradução para PT-BR: Esta tradução visa tornar o conteúdo acessível para desenvolvedores brasileiros, mantendo a precisão técnica do material original.*

## Breve Descrição

Roteamento é um padrão de design agêntico que direciona consultas ou entradas de usuário para o manipulador, modelo ou caminho de processamento mais apropriado, baseado no conteúdo, intenção ou características da solicitação. Este padrão permite utilização eficiente de recursos e processamento especializado para diferentes tipos de tarefas.

## Introdução

Em sistemas agênticos complexos, nem todas as tarefas requerem o mesmo tipo de processamento ou expertise. O padrão de roteamento aborda este desafio implementando mecanismos inteligentes de tomada de decisão que direcionam solicitações para o pipeline de processamento, modelo ou agente mais adequado.

O roteamento pode ocorrer em múltiplos níveis: desde roteamento simples baseado em palavras-chave até compreensão semântica sofisticada que determina a melhor abordagem para lidar com uma solicitação específica. Este padrão é essencial para construir sistemas escaláveis que podem lidar com necessidades diversas de usuários enquanto otimiza performance e uso de recursos.

## Conceitos Chave

### Classificação de Intenção
- Analisar entrada do usuário para determinar a intenção ou propósito subjacente
- Mapear intenções para caminhos de processamento apropriados
- Lidar com cenários ambíguos ou de múltiplas intenções

### Correspondência de Capacidades
- Corresponder características da solicitação com capacidades disponíveis do sistema
- Roteamento baseado em expertise, ferramentas ou domínios de conhecimento necessários
- Descoberta e seleção dinâmica de capacidades

### Balanceamento de Carga
- Distribuir solicitações entre múltiplas unidades de processamento
- Otimizar utilização de recursos e tempos de resposta
- Implementar mecanismos de failover e redundância

### Roteamento Consciente de Contexto
- Considerar contexto do usuário, histórico e preferências
- Adaptar decisões de roteamento baseadas no estado do sistema
- Implementar estratégias de roteamento personalizadas

## Implementação

### Arquitetura Básica de Roteador
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

### Sistema de Roteamento Inteligente
```python
class IntelligentRouter:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.handlers = {}
        self.load_balancer = LoadBalancer()

    def route_request(self, request):
        # Classificar intenção
        intent = self.classifier.classify(request)

        # Obter manipuladores disponíveis
        handlers = self.handlers.get(intent, [])

        # Selecionar manipulador ótimo
        selected_handler = self.load_balancer.select(handlers)

        return selected_handler.process(request)
```

## Exemplos de Código

### Exemplo 1: Roteador de Atendimento ao Cliente Multi-Domínio
```python
class CustomerServiceRouter:
    def __init__(self):
        self.technical_keywords = ['erro', 'bug', 'não funciona', 'travou']
        self.billing_keywords = ['pagamento', 'cobrança', 'reembolso', 'fatura']
        self.general_keywords = ['conta', 'senha', 'login']

    def route_inquiry(self, customer_message):
        message_lower = customer_message.lower()

        # Problemas técnicos
        if any(keyword in message_lower for keyword in self.technical_keywords):
            return self.route_to_technical_support(customer_message)

        # Problemas de cobrança
        elif any(keyword in message_lower for keyword in self.billing_keywords):
            return self.route_to_billing_department(customer_message)

        # Consultas gerais
        elif any(keyword in message_lower for keyword in self.general_keywords):
            return self.route_to_general_support(customer_message)

        # Fallback para agente humano
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

### Exemplo 2: Roteador de Modelo LLM
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
        # Classificação simples baseada em heurística
        if any(word in prompt.lower() for word in ['resolver', 'calcular', 'analisar']):
            return 'reasoning'
        elif any(word in prompt.lower() for word in ['escrever', 'criar', 'história']):
            return 'creative'
        elif any(word in prompt.lower() for word in ['código', 'função', 'debug']):
            return 'coding'
        elif any(word in prompt.lower() for word in ['resumir', 'tldr', 'breve']):
            return 'summarization'
        else:
            return 'reasoning'  # padrão

    def route_to_model(self, prompt):
        task_type = self.classify_task(prompt)
        available_models = self.models[task_type]

        # Selecionar melhor modelo disponível (poderia incluir balanceamento de carga)
        selected_model = self.select_optimal_model(available_models)

        return self.call_model(selected_model, prompt)

    def select_optimal_model(self, models):
        # Seleção simples - na prática, considerar carga, custo, performance
        return models[0]

    def call_model(self, model, prompt):
        # Implementação depende da API do modelo
        pass
```

### Exemplo 3: Roteamento Semântico com Embeddings
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

        # Encontrar rota mais similar
        similarities = {}
        for route_id, route_embedding in self.route_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                route_embedding.reshape(1, -1)
            )[0][0]
            similarities[route_id] = similarity

        # Selecionar melhor correspondência
        best_route = max(similarities, key=similarities.get)

        # Aplicar limiar para confiança
        if similarities[best_route] > 0.7:
            return self.handlers[best_route]
        else:
            return self.default_handler

    def get_embedding(self, text):
        # Implementação depende do serviço de embedding
        pass
```

## Melhores Práticas

### Design do Roteador
- **Lógica de Decisão Clara**: Tornar decisões de roteamento transparentes e auditáveis
- **Mecanismos de Fallback**: Sempre fornecer manipuladores padrão para casos não correspondidos
- **Otimização de Performance**: Minimizar overhead de roteamento e tempo de decisão
- **Escalabilidade**: Projetar roteadores que podem lidar com números crescentes de rotas e solicitações

### Classificação de Intenção
- **Qualidade de Dados de Treinamento**: Usar dados de treinamento diversos e representativos
- **Atualizações Regulares**: Melhorar continuamente a precisão da classificação
- **Tratamento de Múltiplas Intenções**: Suportar solicitações com intenções múltiplas ou ambíguas
- **Limiares de Confiança**: Implementar decisões de roteamento baseadas em confiança

### Monitoramento e Analytics
- **Performance da Rota**: Rastrear taxas de sucesso e tempos de resposta para cada rota
- **Padrões de Uso**: Analisar padrões de roteamento para otimizar design do sistema
- **Rastreamento de Erros**: Monitorar e alertar sobre falhas de roteamento
- **Testes A/B**: Testar diferentes estratégias e configurações de roteamento

## Armadilhas Comuns

### Lógica de Roteamento Excessivamente Complexa
- **Problema**: Criar sistemas de roteamento excessivamente sofisticados para casos de uso simples
- **Solução**: Começar com regras simples e adicionar complexidade incrementalmente
- **Mitigação**: Revisão regular e simplificação da lógica de roteamento

### Cascatas de Classificação Incorreta
- **Problema**: Roteamento incorreto leva a experiências ruins do usuário
- **Solução**: Implementar limiares de confiança e supervisão humana
- **Mitigação**: Fornecer mecanismos fáceis para usuários reportarem erros de roteamento

### Proliferação de Rotas
- **Problema**: Muitas rotas específicas tornam o sistema difícil de manter
- **Solução**: Consolidar rotas similares e usar roteamento hierárquico
- **Mitigação**: Auditoria regular e limpeza de rotas não utilizadas ou redundantes

### Gargalos de Performance
- **Problema**: Decisões de roteamento se tornam um gargalo do sistema
- **Solução**: Otimizar algoritmos de roteamento e implementar cache
- **Mitigação**: Fazer profiling da performance de roteamento e definir metas de performance

### Experiência de Usuário Inconsistente
- **Problema**: Diferentes rotas fornecem qualidade variável de serviço
- **Solução**: Padronizar interfaces e métricas de qualidade entre rotas
- **Mitigação**: Implementar monitoramento de qualidade e mecanismos de feedback

### Fallbacks Ausentes
- **Problema**: Sistema falha quando nenhuma rota corresponde à solicitação
- **Solução**: Sempre implementar mecanismos abrangentes de fallback
- **Mitigação**: Testar casos extremos e entradas incomuns regularmente

## Conceitos Avançados

### Aprendizado Dinâmico de Rotas
- Descobrir automaticamente novos padrões de roteamento do comportamento do usuário
- Adaptar estratégias de roteamento baseadas em métricas de sucesso
- Otimização de rotas baseada em machine learning

### Roteamento Multi-Estágio
- Roteamento hierárquico com múltiplos pontos de decisão
- Refinamento progressivo de decisões de roteamento
- Sub-roteamento dependente de contexto

### Roteamento Colaborativo
- Múltiplos agentes colaborando em decisões de roteamento
- Roteamento baseado em consenso para decisões críticas
- Roteamento distribuído entre múltiplos sistemas

## Conclusão

Roteamento é um padrão crítico para construir sistemas agênticos escaláveis e eficientes. Ao direcionar inteligentemente solicitações para manipuladores apropriados, o roteamento permite que sistemas forneçam respostas especializadas e de alta qualidade enquanto otimiza a utilização de recursos. O sucesso com roteamento requer atenção cuidadosa à precisão da classificação, otimização de performance e mecanismos abrangentes de fallback.