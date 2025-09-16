# Capítulo 20: Priorização

**Descrição do Padrão:** Os padrões de Priorização permitem que agentes de IA classifiquem, ordenem e agendem efetivamente tarefas, recursos e decisões baseados em múltiplos critérios incluindo urgência, importância, dependências e recursos disponíveis para otimizar performance geral do sistema.

## Introdução

A priorização é uma capacidade fundamental que distingue agentes inteligentes de sistemas simplesmente reativos. Em ambientes complexos com múltiplas demandas competitivas, recursos limitados e restrições variáveis, a capacidade de priorizar efetivamente determina o sucesso de um sistema de agente de IA. Os padrões de priorização fornecem abordagens sistemáticas para agentes tomarem decisões informadas sobre o que fazer primeiro, o que adiar e o que abandonar completamente.

Agentes de IA modernos frequentemente operam em ambientes dinâmicos onde prioridades podem mudar rapidamente baseadas em condições mutáveis, novas informações ou objetivos em evolução. Priorização eficaz requer não apenas mecanismos de classificação, mas também algoritmos adaptativos que possam responder a circunstâncias em mudança mantendo coerência geral do sistema e alinhamento de objetivos.

O desafio da priorização em sistemas de IA se estende além da simples ordenação para incluir alocação de recursos, gerenciamento de prazos, resolução de dependências e otimização através de múltiplos objetivos. Estes padrões fornecem frameworks para lidar com estes cenários complexos de priorização sistemática e eficientemente.

## Conceitos-Chave

### Dimensões de Priorização

#### Urgência e Importância
- **Matriz de Eisenhower**: Categorizar tarefas por dimensões urgente/importante
- **Priorização Orientada por Prazo**: Agendamento baseado em restrições de tempo e prazos
- **Análise de Caminho Crítico**: Identificar tarefas que impactam diretamente cronograma geral
- **Avaliação de Impacto**: Avaliar consequências da conclusão ou atraso de tarefas

#### Requisitos de Recursos
- **Disponibilidade de Recursos**: Priorizar baseado em recursos atualmente disponíveis
- **Eficiência de Recursos**: Favorecer tarefas com melhores taxas de utilização de recursos
- **Contenção de Recursos**: Gerenciar demandas competitivas por recursos limitados
- **Custo de Oportunidade**: Considerar o que deve ser abandonado ao escolher prioridades

#### Dependências e Relacionamentos
- **Ordenação de Pré-requisitos**: Garantir que dependências sejam satisfeitas antes da execução da tarefa
- **Execução Paralela**: Identificar tarefas que podem ser realizadas simultaneamente
- **Relacionamentos Bloqueantes**: Priorizar tarefas que desbloqueiam outro trabalho importante
- **Efeitos em Cascata**: Considerar como a conclusão de tarefas afeta outras tarefas

#### Alinhamento Estratégico
- **Contribuição para Objetivos**: Priorizar tarefas que melhor avançam objetivos gerais
- **Valor Estratégico**: Pesar benefícios de longo prazo versus curto prazo
- **Mitigação de Riscos**: Priorizar tarefas que reduzem vulnerabilidades do sistema
- **Potencial de Inovação**: Balancear tarefas de manutenção com trabalho exploratório

### Algoritmos de Priorização

#### Métodos Simples de Classificação
- **Pontuação Ponderada**: Atribuir pontuações baseadas em múltiplos critérios ponderados
- **Comparação em Pares**: Comparar tarefas umas contra as outras sistematicamente
- **Esquemas de Prioridade Fixa**: Usar níveis ou classes de prioridade predeterminados
- **FIFO/LIFO**: Ordenação primeiro-a-entrar-primeiro-a-sair ou último-a-entrar-primeiro-a-sair

#### Priorização Dinâmica
- **Pontuação em Tempo Real**: Atualizar continuamente prioridades baseadas em condições mutáveis
- **Pesos Adaptativos**: Ajustar critérios de priorização baseados em feedback de performance
- **Priorização Consciente de Contexto**: Modificar prioridades baseadas no estado atual do sistema
- **Priorização Baseada em Aprendizado**: Usar machine learning para melhorar decisões de prioridade

#### Otimização Multi-Objetivo
- **Otimização de Pareto**: Encontrar trade-offs ótimos entre objetivos competitivos
- **Maximização de Utilidade**: Otimizar para utilidade ou valor geral do sistema
- **Satisfação de Restrições**: Priorizar dentro de restrições rígidas e flexíveis
- **Abordagens de Teoria dos Jogos**: Lidar com prioridades conflitantes de múltiplos stakeholders

## Implementação

### Sistema de Fila de Prioridade

```python
import heapq
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

class PriorityLevel(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    DEFERRED = 4

@dataclass
class Task:
    id: str
    name: str
    description: str
    priority_score: float
    created_at: float
    deadline: Optional[float] = None
    estimated_duration: float = 0.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        # Lower scores have higher priority (for min-heap)
        return self.priority_score < other.priority_score

class PriorityCalculator:
    def __init__(self):
        self.weights = {
            'urgency': 0.3,
            'importance': 0.25,
            'resource_efficiency': 0.2,
            'dependency_impact': 0.15,
            'strategic_value': 0.1
        }
        self.urgency_decay_rate = 0.1  # How quickly urgency increases

    def calculate_priority(self, task: Task, context: Dict[str, Any] = None) -> float:
        """Calculate comprehensive priority score for a task"""
        context = context or {}
        current_time = time.time()

        # Calculate individual components
        urgency_score = self._calculate_urgency(task, current_time)
        importance_score = self._calculate_importance(task, context)
        efficiency_score = self._calculate_resource_efficiency(task, context)
        dependency_score = self._calculate_dependency_impact(task, context)
        strategic_score = self._calculate_strategic_value(task, context)

        # Weighted combination (lower scores = higher priority)
        priority_score = (
            self.weights['urgency'] * urgency_score +
            self.weights['importance'] * importance_score +
            self.weights['resource_efficiency'] * efficiency_score +
            self.weights['dependency_impact'] * dependency_score +
            self.weights['strategic_value'] * strategic_score
        )

        return priority_score

    def _calculate_urgency(self, task: Task, current_time: float) -> float:
        """Calculate urgency based on deadline and age"""
        base_urgency = 0.5

        # Deadline urgency
        if task.deadline:
            time_until_deadline = task.deadline - current_time
            if time_until_deadline <= 0:
                return 0.0  # Overdue = highest urgency
            elif time_until_deadline < 3600:  # Less than 1 hour
                base_urgency = 0.1
            elif time_until_deadline < 86400:  # Less than 1 day
                base_urgency = 0.3
            elif time_until_deadline < 604800:  # Less than 1 week
                base_urgency = 0.5

        # Age-based urgency increase
        age_hours = (current_time - task.created_at) / 3600
        age_factor = 1 - (age_hours * self.urgency_decay_rate / 100)
        age_factor = max(0.1, min(1.0, age_factor))

        return base_urgency * age_factor

    def _calculate_importance(self, task: Task, context: Dict[str, Any]) -> float:
        """Calculate importance based on task characteristics and context"""
        base_importance = 0.5

        # Tag-based importance
        important_tags = context.get('important_tags', [])
        if any(tag in task.tags for tag in important_tags):
            base_importance = 0.2

        # Critical system tags
        if 'critical' in task.tags:
            base_importance = 0.1
        elif 'high' in task.tags:
            base_importance = 0.3
        elif 'low' in task.tags:
            base_importance = 0.8

        return base_importance

    def _calculate_resource_efficiency(self, task: Task, context: Dict[str, Any]) -> float:
        """Calculate resource efficiency score"""
        available_resources = context.get('available_resources', {})

        if not task.resource_requirements or not available_resources:
            return 0.5

        # Calculate resource utilization ratio
        total_required = sum(task.resource_requirements.values())
        total_available = sum(available_resources.values())

        if total_available == 0:
            return 1.0  # No resources available = low priority

        utilization_ratio = total_required / total_available
        # Lower utilization = higher priority (lower score)
        return min(utilization_ratio, 1.0)

    def _calculate_dependency_impact(self, task: Task, context: Dict[str, Any]) -> float:
        """Calculate impact based on dependencies"""
        blocked_tasks = context.get('blocked_by_task', {}).get(task.id, [])

        if not blocked_tasks:
            return 0.5  # No blocking impact

        # More blocked tasks = higher priority (lower score)
        blocking_impact = max(0.1, 0.5 - (len(blocked_tasks) * 0.1))
        return blocking_impact

    def _calculate_strategic_value(self, task: Task, context: Dict[str, Any]) -> float:
        """Calculate strategic value score"""
        strategic_goals = context.get('strategic_goals', [])

        if not strategic_goals:
            return 0.5

        # Check alignment with strategic goals
        aligned_goals = sum(1 for goal in strategic_goals if goal in task.tags)
        if aligned_goals > 0:
            return max(0.1, 0.5 - (aligned_goals * 0.2))

        return 0.5

class PriorityQueue:
    def __init__(self, priority_calculator: PriorityCalculator = None):
        self.heap: List[Task] = []
        self.task_map: Dict[str, Task] = {}
        self.priority_calculator = priority_calculator or PriorityCalculator()
        self.context: Dict[str, Any] = {}

    def add_task(self, task: Task):
        """Add a task to the priority queue"""
        # Calculate priority score
        task.priority_score = self.priority_calculator.calculate_priority(task, self.context)

        # Add to heap and task map
        heapq.heappush(self.heap, task)
        self.task_map[task.id] = task

    def get_next_task(self) -> Optional[Task]:
        """Get the highest priority task"""
        while self.heap:
            task = heapq.heappop(self.heap)
            if task.id in self.task_map:
                del self.task_map[task.id]
                return task
        return None

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue"""
        if task_id in self.task_map:
            del self.task_map[task_id]
            # Task will be removed from heap when popped
            return True
        return False

    def update_priorities(self):
        """Recalculate all task priorities"""
        tasks = list(self.task_map.values())
        self.heap.clear()
        self.task_map.clear()

        for task in tasks:
            self.add_task(task)

    def update_context(self, new_context: Dict[str, Any]):
        """Update context and recalculate priorities"""
        self.context.update(new_context)
        self.update_priorities()

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        priority_distribution = {}
        for task in self.task_map.values():
            if task.priority_score < 0.2:
                level = "CRITICAL"
            elif task.priority_score < 0.4:
                level = "HIGH"
            elif task.priority_score < 0.6:
                level = "MEDIUM"
            else:
                level = "LOW"

            priority_distribution[level] = priority_distribution.get(level, 0) + 1

        return {
            'total_tasks': len(self.task_map),
            'priority_distribution': priority_distribution,
            'next_task_id': self.heap[0].id if self.heap else None
        }

class AdaptivePriorityAgent:
    def __init__(self):
        self.priority_queue = PriorityQueue()
        self.completed_tasks: List[Task] = []
        self.performance_metrics: Dict[str, float] = {}
        self.running = False

    async def add_task(self, name: str, description: str, **kwargs):
        """Add a new task to the agent"""
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            priority_score=0.0,
            created_at=time.time(),
            **kwargs
        )

        self.priority_queue.add_task(task)
        print(f"Added task: {name} (ID: {task.id})")

    async def process_tasks(self):
        """Main task processing loop"""
        self.running = True

        while self.running:
            task = self.priority_queue.get_next_task()

            if task:
                print(f"Processing task: {task.name} (Priority: {task.priority_score:.3f})")

                # Simulate task processing
                start_time = time.time()
                processing_time = max(0.1, task.estimated_duration)
                await asyncio.sleep(processing_time)

                # Mark task as completed
                task.metadata['completed_at'] = time.time()
                task.metadata['actual_duration'] = time.time() - start_time
                self.completed_tasks.append(task)

                print(f"Completed task: {task.name}")

                # Update performance metrics
                self._update_performance_metrics(task)

            else:
                # No tasks available, wait a bit
                await asyncio.sleep(1.0)

    def _update_performance_metrics(self, completed_task: Task):
        """Update performance metrics based on completed task"""
        actual_duration = completed_task.metadata.get('actual_duration', 0)
        estimated_duration = completed_task.estimated_duration

        if estimated_duration > 0:
            accuracy = 1 - abs(actual_duration - estimated_duration) / estimated_duration
            self.performance_metrics['estimation_accuracy'] = (
                self.performance_metrics.get('estimation_accuracy', 0.5) * 0.9 + accuracy * 0.1
            )

    async def update_system_context(self, context_update: Dict[str, Any]):
        """Update system context and reprioritize tasks"""
        self.priority_queue.update_context(context_update)
        print(f"Context updated: {context_update}")

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        queue_status = self.priority_queue.get_queue_status()

        return {
            'queue_status': queue_status,
            'completed_tasks': len(self.completed_tasks),
            'performance_metrics': self.performance_metrics,
            'is_running': self.running
        }

    def stop(self):
        """Stop the agent"""
        self.running = False
```

## Exemplos de Uso

```python
# Exemplo de uso do sistema de priorização
async def prioritization_example():
    agent = AdaptivePriorityAgent()

    # Adicionar algumas tarefas
    await agent.add_task(
        "Backup do banco de dados",
        "Realizar backup diário do banco de dados",
        deadline=time.time() + 3600,  # 1 hora
        estimated_duration=0.5,
        tags=['critical', 'maintenance'],
        resource_requirements={'cpu': 0.3, 'disk': 0.8}
    )

    await agent.add_task(
        "Atualizar documentação",
        "Atualizar documentação da API",
        estimated_duration=2.0,
        tags=['documentation', 'low'],
        resource_requirements={'cpu': 0.1}
    )

    await agent.add_task(
        "Corrigir bug crítico",
        "Corrigir bug que afeta performance",
        deadline=time.time() + 1800,  # 30 minutos
        estimated_duration=1.0,
        tags=['critical', 'bugfix'],
        resource_requirements={'cpu': 0.5, 'memory': 0.3}
    )

    # Atualizar contexto do sistema
    await agent.update_system_context({
        'available_resources': {'cpu': 1.0, 'memory': 1.0, 'disk': 1.0},
        'important_tags': ['critical', 'bugfix'],
        'strategic_goals': ['reliability', 'performance']
    })

    # Obter relatório de status
    status = agent.get_status_report()
    print(f"Status inicial: {status}")

    # Processar algumas tarefas
    await asyncio.sleep(2)

    status = agent.get_status_report()
    print(f"Status após processamento: {status}")

# asyncio.run(prioritization_example())
```

## Melhores Práticas

### Design de Sistema de Priorização
- **Critérios Múltiplos**: Use múltiplas dimensões de priorização para decisões robustas
- **Adaptabilidade**: Implemente mecanismos para ajustar prioridades dinamicamente
- **Transparência**: Forneça visibilidade clara sobre como prioridades são calculadas
- **Eficiência**: Otimize algoritmos de priorização para performance em tempo real

### Gerenciamento de Dependências
- **Análise de Dependências**: Identifique e gerencie dependências entre tarefas
- **Detecção de Ciclos**: Previna dependências circulares no sistema
- **Paralelização**: Maximize execução paralela onde possível
- **Resolução de Bloqueios**: Priorize tarefas que desbloqueiam outras

### Otimização de Recursos
- **Balanceamento de Carga**: Distribua carga de trabalho eficientemente
- **Previsão de Recursos**: Antecipe necessidades futuras de recursos
- **Elasticidade**: Ajuste alocação de recursos baseada na demanda
- **Monitoramento**: Rastreie utilização de recursos e performance

## Armadilhas Comuns

### Over-optimization
- **Problema**: Gastar muito tempo otimizando prioridades em vez de executar tarefas
- **Solução**: Balance overhead de otimização com benefícios práticos

### Mudanças Frequentes de Prioridade
- **Problema**: Prioridades mudando constantemente, causando thrashing
- **Solução**: Implemente histerese e períodos mínimos de estabilidade

### Negligência de Tarefas de Baixa Prioridade
- **Problema**: Tarefas de baixa prioridade nunca sendo executadas
- **Solução**: Implemente aging e garantias de progresso mínimo

### Complexidade Excessiva
- **Problema**: Algoritmos de priorização muito complexos reduzindo performance
- **Solução**: Use heurísticas simples e otimize apenas onde necessário

---

*Este capítulo aborda conceitos de Priorização para sistemas de agentes de IA eficazes e adaptativos.*

---

**Nota de Tradução**: Este documento foi traduzido do inglês para português brasileiro. O conteúdo técnico original foi preservado, mantendo termos técnicos estabelecidos em inglês quando apropriado.