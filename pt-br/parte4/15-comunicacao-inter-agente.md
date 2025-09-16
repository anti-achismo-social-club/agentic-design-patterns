# Capítulo 15: Comunicação Inter-Agente (A2A)

**Descrição do Padrão:** A Comunicação Inter-Agente (A2A) permite que múltiplos agentes de IA troquem informações, coordenem ações e colaborem efetivamente para alcançar objetivos complexos que requerem inteligência distribuída e resolução cooperativa de problemas.

## Introdução

A Comunicação Inter-Agente representa uma mudança fundamental de sistemas de agente único para arquiteturas multi-agente onde agentes autônomos podem se comunicar, negociar e colaborar. Este padrão permite a criação de ecossistemas sofisticados de IA onde agentes especializados trabalham juntos, compartilhando conhecimento e coordenando suas ações para resolver problemas complexos que seriam difíceis ou impossíveis para um único agente lidar sozinho.

O padrão A2A se inspira em sistemas distribuídos, pesquisa de sistemas multi-agente e conceitos de inteligência colaborativa. Ele aborda a crescente necessidade de sistemas de IA que possam escalar além das capacidades individuais dos agentes, mantendo coordenação e evitando conflitos.

## Conceitos-Chave

### Protocolos de Comunicação
- **Passagem de Mensagens**: Protocolos padronizados para agentes trocarem informações estruturadas
- **Comunicação Orientada a Eventos**: Mensagens assíncronas baseadas em eventos do sistema e mudanças de estado
- **Padrões Requisição-Resposta**: Comunicação síncrona para necessidades de coordenação imediata
- **Mensagens Broadcast**: Comunicação um-para-muitos para anúncios em todo o sistema

### Mecanismos de Coordenação
- **Distribuição de Tarefas**: Algoritmos para dividir tarefas complexas entre múltiplos agentes
- **Compartilhamento de Recursos**: Protocolos para gerenciar recursos computacionais e de dados compartilhados
- **Resolução de Conflitos**: Mecanismos para lidar com objetivos competitivos dos agentes e conflitos de recursos
- **Construção de Consenso**: Métodos para agentes chegarem a acordos sobre decisões e ações

### Papéis e Hierarquias de Agentes
- **Agentes Coordenadores**: Agentes centrais responsáveis por orquestrar workflows multi-agente
- **Agentes Especialistas**: Agentes específicos de domínio com capacidades especializadas
- **Agentes Intermediários**: Agentes intermediários que facilitam a comunicação entre outros agentes
- **Agentes Monitores**: Agentes de supervisão que rastreiam performance e saúde do sistema

### Formatos de Troca de Informações
- **Formatos de Dados Estruturados**: JSON, XML ou esquemas customizados para troca confiável de dados
- **Protocolos Semânticos**: Comunicação baseada em ontologia para preservação de significado
- **Compartilhamento de Contexto**: Mecanismos para compartilhar consciência situacional entre agentes
- **Bases de Conhecimento**: Repositórios compartilhados de informações acessíveis a múltiplos agentes

## Implementação

### Infraestrutura Básica de Comunicação

```python
import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import uuid

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"

@dataclass
class Message:
    id: str
    sender_id: str
    recipient_id: Optional[str]
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None

class CommunicationBus:
    def __init__(self):
        self.agents: Dict[str, 'Agent'] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False

    async def register_agent(self, agent: 'Agent'):
        self.agents[agent.agent_id] = agent
        await agent.set_communication_bus(self)

    async def send_message(self, message: Message):
        await self.message_queue.put(message)

    async def start(self):
        self.running = True
        while self.running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=0.1
                )
                await self._route_message(message)
            except asyncio.TimeoutError:
                continue

    async def _route_message(self, message: Message):
        if message.recipient_id:
            # Direct message
            if message.recipient_id in self.agents:
                await self.agents[message.recipient_id].receive_message(message)
        else:
            # Broadcast message
            for agent in self.agents.values():
                if agent.agent_id != message.sender_id:
                    await agent.receive_message(message)
```

### Classe Base de Agente com Comunicação

```python
class Agent:
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.communication_bus: Optional[CommunicationBus] = None
        self.message_handlers: Dict[str, callable] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}

    async def set_communication_bus(self, bus: CommunicationBus):
        self.communication_bus = bus

    async def send_request(self, recipient_id: str, action: str,
                          data: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=MessageType.REQUEST,
            content={"action": action, "data": data},
            timestamp=asyncio.get_event_loop().time(),
            correlation_id=request_id
        )

        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future

        await self.communication_bus.send_message(message)

        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        finally:
            self.pending_requests.pop(request_id, None)

    async def send_response(self, original_message: Message,
                           response_data: Dict[str, Any]):
        response = Message(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=original_message.sender_id,
            message_type=MessageType.RESPONSE,
            content=response_data,
            timestamp=asyncio.get_event_loop().time(),
            correlation_id=original_message.correlation_id
        )

        await self.communication_bus.send_message(response)

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=None,
            message_type=MessageType.BROADCAST,
            content={"event_type": event_type, "data": data},
            timestamp=asyncio.get_event_loop().time()
        )

        await self.communication_bus.send_message(message)

    async def receive_message(self, message: Message):
        if message.message_type == MessageType.REQUEST:
            await self._handle_request(message)
        elif message.message_type == MessageType.RESPONSE:
            await self._handle_response(message)
        elif message.message_type == MessageType.BROADCAST:
            await self._handle_broadcast(message)

    async def _handle_request(self, message: Message):
        action = message.content.get("action")
        data = message.content.get("data", {})

        if action in self.message_handlers:
            try:
                result = await self.message_handlers[action](data)
                await self.send_response(message, {"success": True, "result": result})
            except Exception as e:
                await self.send_response(message, {"success": False, "error": str(e)})
        else:
            await self.send_response(message, {
                "success": False,
                "error": f"Unknown action: {action}"
            })

    async def _handle_response(self, message: Message):
        correlation_id = message.correlation_id
        if correlation_id in self.pending_requests:
            future = self.pending_requests[correlation_id]
            if not future.done():
                future.set_result(message.content)

    async def _handle_broadcast(self, message: Message):
        event_type = message.content.get("event_type")
        data = message.content.get("data", {})

        # Override in subclasses to handle specific broadcast events
        await self.on_broadcast_received(event_type, data)

    async def on_broadcast_received(self, event_type: str, data: Dict[str, Any]):
        """Override in subclasses to handle broadcast events"""
        pass

    def register_handler(self, action: str, handler: callable):
        self.message_handlers[action] = handler
```

### Exemplos de Agentes Especializados

```python
class CoordinatorAgent(Agent):
    def __init__(self):
        super().__init__("coordinator", ["task_management", "coordination"])
        self.active_tasks: Dict[str, Dict] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}

        # Register message handlers
        self.register_handler("submit_task", self._handle_submit_task)
        self.register_handler("report_capability", self._handle_report_capability)
        self.register_handler("task_completed", self._handle_task_completed)

    async def _handle_submit_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "description": data.get("description"),
            "requirements": data.get("requirements", []),
            "status": "pending",
            "assigned_agents": []
        }

        self.active_tasks[task_id] = task

        # Find suitable agents
        suitable_agents = self._find_suitable_agents(task["requirements"])

        if suitable_agents:
            await self._assign_task(task, suitable_agents)
            return {"task_id": task_id, "status": "assigned", "agents": suitable_agents}
        else:
            return {"task_id": task_id, "status": "pending", "message": "No suitable agents found"}

    async def _handle_report_capability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        agent_id = data.get("agent_id")
        capabilities = data.get("capabilities", [])
        self.agent_capabilities[agent_id] = capabilities
        return {"status": "registered"}

    async def _handle_task_completed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        task_id = data.get("task_id")
        result = data.get("result")

        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["result"] = result

            # Broadcast completion
            await self.broadcast("task_completed", {
                "task_id": task_id,
                "result": result
            })

        return {"status": "acknowledged"}

    def _find_suitable_agents(self, requirements: List[str]) -> List[str]:
        suitable_agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            if all(req in capabilities for req in requirements):
                suitable_agents.append(agent_id)
        return suitable_agents

    async def _assign_task(self, task: Dict, agents: List[str]):
        for agent_id in agents:
            try:
                response = await self.send_request(agent_id, "assign_task", {
                    "task_id": task["id"],
                    "description": task["description"]
                })

                if response.get("success"):
                    task["assigned_agents"].append(agent_id)
                    task["status"] = "assigned"
                    break
            except Exception as e:
                print(f"Failed to assign task to {agent_id}: {e}")

class DataProcessingAgent(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, ["data_processing", "analysis"])

        self.register_handler("assign_task", self._handle_assign_task)
        self.register_handler("process_data", self._handle_process_data)

    async def _handle_assign_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        task_id = data.get("task_id")
        description = data.get("description")

        # Simulate task acceptance
        print(f"Agent {self.agent_id} accepted task {task_id}: {description}")

        # Start processing (simulate with delay)
        asyncio.create_task(self._process_task(task_id, description))

        return {"success": True, "message": "Task accepted"}

    async def _handle_process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate data processing
        input_data = data.get("data", [])
        processed_data = [x * 2 for x in input_data if isinstance(x, (int, float))]

        return {"processed_data": processed_data}

    async def _process_task(self, task_id: str, description: str):
        # Simulate processing time
        await asyncio.sleep(2)

        # Report completion to coordinator
        await self.send_request("coordinator", "task_completed", {
            "task_id": task_id,
            "result": f"Processed: {description}"
        })
```

### Exemplo de Workflow Multi-Agente

```python
class MultiAgentWorkflow:
    def __init__(self):
        self.communication_bus = CommunicationBus()
        self.agents: List[Agent] = []

    async def setup(self):
        # Create coordinator
        coordinator = CoordinatorAgent()
        await self.communication_bus.register_agent(coordinator)
        self.agents.append(coordinator)

        # Create processing agents
        for i in range(3):
            agent = DataProcessingAgent(f"processor_{i}")
            await self.communication_bus.register_agent(agent)
            self.agents.append(agent)

            # Report capabilities to coordinator
            await agent.send_request("coordinator", "report_capability", {
                "agent_id": agent.agent_id,
                "capabilities": agent.capabilities
            })

        # Start communication bus
        asyncio.create_task(self.communication_bus.start())

    async def submit_task(self, description: str, requirements: List[str]):
        coordinator = self.agents[0]  # First agent is coordinator

        response = await coordinator.send_request("coordinator", "submit_task", {
            "description": description,
            "requirements": requirements
        })

        return response

# Usage example
async def main():
    workflow = MultiAgentWorkflow()
    await workflow.setup()

    # Submit a task
    result = await workflow.submit_task(
        "Process customer data",
        ["data_processing"]
    )

    print(f"Task submission result: {result}")

    # Wait for processing
    await asyncio.sleep(5)

# Run the example
# asyncio.run(main())
```

## Exemplos de Código

### Protocolo Avançado de Negociação

```python
class NegotiationAgent(Agent):
    def __init__(self, agent_id: str, resources: Dict[str, int]):
        super().__init__(agent_id, ["negotiation", "resource_management"])
        self.resources = resources
        self.active_negotiations: Dict[str, Dict] = {}

        self.register_handler("initiate_negotiation", self._handle_negotiation_request)
        self.register_handler("negotiation_proposal", self._handle_proposal)
        self.register_handler("negotiation_response", self._handle_response)

    async def _handle_negotiation_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        negotiation_id = str(uuid.uuid4())
        requested_resources = data.get("resources", {})

        # Check if we can potentially fulfill the request
        can_negotiate = all(
            self.resources.get(resource, 0) >= amount
            for resource, amount in requested_resources.items()
        )

        if can_negotiate:
            self.active_negotiations[negotiation_id] = {
                "partner": data.get("requester"),
                "requested": requested_resources,
                "status": "active"
            }

            # Make counter-proposal
            counter_proposal = self._generate_counter_proposal(requested_resources)

            await self.send_request(data.get("requester"), "negotiation_proposal", {
                "negotiation_id": negotiation_id,
                "proposal": counter_proposal
            })

            return {"success": True, "negotiation_id": negotiation_id}
        else:
            return {"success": False, "reason": "Insufficient resources"}

    def _generate_counter_proposal(self, requested: Dict[str, int]) -> Dict[str, Any]:
        # Simple negotiation: offer 80% of requested resources for a favor
        counter = {}
        for resource, amount in requested.items():
            counter[resource] = int(amount * 0.8)

        return {
            "resources": counter,
            "conditions": {"future_favor": True}
        }

class ConsensusAgent(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, ["consensus", "voting"])
        self.voting_sessions: Dict[str, Dict] = {}

        self.register_handler("initiate_vote", self._handle_vote_initiation)
        self.register_handler("cast_vote", self._handle_vote_cast)

    async def _handle_vote_initiation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        vote_id = str(uuid.uuid4())
        proposal = data.get("proposal")
        participants = data.get("participants", [])

        self.voting_sessions[vote_id] = {
            "proposal": proposal,
            "participants": participants,
            "votes": {},
            "status": "active"
        }

        # Broadcast vote to all participants
        await self.broadcast("vote_initiated", {
            "vote_id": vote_id,
            "proposal": proposal,
            "participants": participants
        })

        return {"vote_id": vote_id}

    async def _handle_vote_cast(self, data: Dict[str, Any]) -> Dict[str, Any]:
        vote_id = data.get("vote_id")
        voter = data.get("voter")
        decision = data.get("decision")  # "yes", "no", "abstain"

        if vote_id in self.voting_sessions:
            session = self.voting_sessions[vote_id]
            session["votes"][voter] = decision

            # Check if all votes are in
            if len(session["votes"]) == len(session["participants"]):
                result = self._calculate_result(session)
                session["status"] = "completed"
                session["result"] = result

                # Broadcast result
                await self.broadcast("vote_completed", {
                    "vote_id": vote_id,
                    "result": result
                })

        return {"success": True}

    def _calculate_result(self, session: Dict) -> Dict[str, Any]:
        votes = session["votes"]
        yes_votes = sum(1 for vote in votes.values() if vote == "yes")
        no_votes = sum(1 for vote in votes.values() if vote == "no")
        abstain_votes = sum(1 for vote in votes.values() if vote == "abstain")

        total_decisive = yes_votes + no_votes
        if total_decisive == 0:
            outcome = "no_decision"
        else:
            outcome = "approved" if yes_votes > no_votes else "rejected"

        return {
            "outcome": outcome,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "abstain_votes": abstain_votes
        }
```

## Melhores Práticas

### Design de Comunicação
- **Use Protocolos Padronizados**: Implemente formatos de mensagem consistentes e padrões de comunicação em todos os agentes
- **Gerencie Comunicação Assíncrona**: Projete para passagem de mensagens não-bloqueante e tratamento de timeouts
- **Implemente Degradação Graceful**: Garanta que o sistema continue funcionando quando agentes individuais estão indisponíveis
- **Versione Formatos de Mensagem**: Planeje para evolução de protocolo e compatibilidade reversa

### Estratégias de Coordenação
- **Defina Papéis Claros**: Estabeleça responsabilidades e capacidades específicas para cada tipo de agente
- **Implemente Resolução de Conflitos**: Desenvolva mecanismos para lidar com objetivos competitivos e conflitos de recursos
- **Use Mecanismos de Timeout**: Previna deadlocks com valores de timeout apropriados para requisições e respostas
- **Monitore Saúde do Sistema**: Rastreie disponibilidade de agentes e métricas de performance

### Considerações de Escalabilidade
- **Projete para Escalabilidade Horizontal**: Permita adição de novos agentes sem redesign do sistema
- **Implemente Balanceamento de Carga**: Distribua tarefas eficientemente entre agentes disponíveis
- **Use Organização Hierárquica**: Estruture redes de agentes para minimizar overhead de comunicação
- **Otimize Roteamento de Mensagens**: Implemente mecanismos eficientes de entrega de mensagens

### Segurança e Confiança
- **Autentique Agentes**: Verifique identidade do agente antes de processar requisições
- **Implemente Autorização**: Controle quais agentes podem realizar ações específicas
- **Valide Mensagens**: Verifique integridade e formato da mensagem antes do processamento
- **Monitore Anomalias**: Detecte padrões de comunicação ou comportamentos incomuns

## Armadilhas Comuns

### Overhead de Comunicação
- **Problema**: Mensagens excessivas levando à degradação de performance
- **Solução**: Implemente agrupamento de mensagens, compressão e algoritmos de roteamento eficientes

### Situações de Deadlock
- **Problema**: Dependências circulares causando travamento do sistema
- **Solução**: Use mecanismos de timeout, detecção de deadlock e agendamento baseado em prioridade

### Estado Inconsistente
- **Problema**: Agentes tendo visões diferentes do estado do sistema
- **Solução**: Implemente mecanismos de consenso e protocolos de sincronização de estado

### Perda de Mensagens
- **Problema**: Mensagens críticas não chegando ao seu destino
- **Solução**: Use mecanismos de confirmação, persistência de mensagens e lógica de retry

### Gargalos de Escalabilidade
- **Problema**: Coordenadores centrais se tornando gargalos de performance
- **Solução**: Implemente coordenação distribuída, estruturas hierárquicas e balanceamento de carga

### Problemas de Confiança e Segurança
- **Problema**: Agentes maliciosos ou comprometidos afetando integridade do sistema
- **Solução**: Implemente autenticação, autorização, monitoramento e mecanismos de validação de agentes

---

*Este capítulo aborda 15 páginas de conteúdo de "Agentic Design Patterns" por Antonio Gulli, focando em padrões de Comunicação Inter-Agente (A2A) para construir sistemas multi-agente colaborativos.*

---

**Nota de Tradução**: Este documento foi traduzido do inglês para português brasileiro. O conteúdo técnico original foi preservado, mantendo termos técnicos estabelecidos em inglês quando apropriado.