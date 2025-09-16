# Capítulo 17: Técnicas de Raciocínio

**Descrição do Padrão:** As Técnicas de Raciocínio permitem que agentes de IA realizem inferência lógica, raciocínio causal e resolução estruturada de problemas através de abordagens sistemáticas incluindo raciocínio dedutivo, indutivo, abdutivo e analógico.

## Introdução

As Técnicas de Raciocínio representam as capacidades cognitivas que permitem aos agentes de IA tirar conclusões, fazer inferências e resolver problemas sistematicamente. Estes padrões vão além de simples correspondência de padrões ou correlações estatísticas para implementar processos lógicos estruturados que espelham as capacidades de raciocínio humano.

A importância das técnicas de raciocínio em agentes de IA não pode ser subestimada. Elas permitem que os agentes lidem com situações novas, tomem decisões sob incerteza, expliquem suas conclusões e adaptem seu comportamento baseado em análise lógica em vez de respostas puramente aprendidas. Isso é particularmente crucial para aplicações que requerem IA explicável, decisões críticas de segurança e resolução de problemas complexos.

Os sistemas de IA modernos requerem cada vez mais capacidades sofisticadas de raciocínio para lidar com a complexidade do mundo real, manter consistência em suas conclusões e fornecer processos de tomada de decisão transparentes. Essas técnicas preenchem a lacuna entre sistemas de IA reativos e agentes verdadeiramente inteligentes capazes de pensamento deliberativo.

## Conceitos-Chave

### Tipos de Raciocínio

#### Raciocínio Dedutivo
- **Inferência Lógica**: Tirar conclusões específicas de princípios ou premissas gerais
- **Sistemas Baseados em Regras**: Aplicar regras se-então para derivar novo conhecimento
- **Lógica Formal**: Usar lógica proposicional e de predicados para inferência sistemática
- **Sistemas de Prova**: Construir provas formais para proposições matemáticas e lógicas

#### Raciocínio Indutivo
- **Reconhecimento de Padrões**: Identificar padrões gerais a partir de observações específicas
- **Formação de Hipóteses**: Gerar teorias baseadas em evidências empíricas
- **Inferência Estatística**: Usar métodos probabilísticos para tirar conclusões
- **Generalização**: Estender conclusões de amostras para populações

#### Raciocínio Abdutivo
- **Melhor Explicação**: Encontrar a explicação mais provável para fenômenos observados
- **Geração de Hipóteses**: Criar hipóteses plausíveis para explicar evidências
- **Raciocínio Diagnóstico**: Identificar causas a partir de sintomas ou efeitos observados
- **Resolução Criativa de Problemas**: Gerar soluções novas através de raciocínio explicativo

#### Raciocínio Analógico
- **Mapeamento de Similaridade**: Identificar similaridades estruturais entre diferentes domínios
- **Transferência de Conhecimento**: Aplicar soluções de domínios familiares a novos problemas
- **Pensamento Metafórico**: Usar metáforas e analogias para compreensão
- **Raciocínio Baseado em Casos**: Resolver novos problemas baseado em casos passados similares

### Frameworks de Raciocínio

#### Raciocínio Simbólico
- **Representação de Conhecimento**: Representação formal de fatos, regras e relacionamentos
- **Programação Lógica**: Usar linguagens lógicas como Prolog para raciocínio
- **Redes Semânticas**: Representação baseada em grafos de relacionamentos de conhecimento
- **Raciocínio Ontológico**: Usar ontologias formais para inferência estruturada

#### Raciocínio Probabilístico
- **Redes Bayesianas**: Representar dependências probabilísticas entre variáveis
- **Modelos de Markov**: Raciocínio sobre sequências e dependências temporais
- **Quantificação de Incerteza**: Gerenciar e propagar incerteza no raciocínio
- **Combinação de Evidências**: Integrar múltiplas fontes de evidência incerta

#### Raciocínio Temporal
- **Lógica Temporal**: Raciocínio sobre relacionamentos e sequências temporais
- **Planejamento**: Gerar sequências de ações para alcançar objetivos
- **Causalidade**: Compreender relacionamentos causa-efeito ao longo do tempo
- **Reconhecimento de Eventos**: Identificar padrões em sequências de eventos temporais

### Componentes da Arquitetura de Raciocínio

#### Base de Conhecimento
- **Armazenamento de Fatos**: Repositório de fatos e observações conhecidos
- **Gerenciamento de Regras**: Coleção de regras de inferência e heurísticas
- **Manutenção de Crenças**: Rastrear confiança e consistência de crenças
- **Atualizações de Conhecimento**: Mecanismos para incorporar novas informações

#### Motor de Inferência
- **Encadeamento Forward**: Raciocínio de fatos para conclusões
- **Encadeamento Backward**: Raciocínio de objetivos para evidências de suporte
- **Métodos de Resolução**: Procedimentos sistemáticos de prova para inferência lógica
- **Resolução de Conflitos**: Lidar com inferências contraditórias ou competitivas

#### Sistema de Explicação
- **Geração de Rastro**: Registrar o processo de raciocínio para transparência
- **Justificação**: Fornecer explicações para conclusões e decisões
- **Avaliação de Confiança**: Quantificar certeza nos resultados do raciocínio
- **Análise What-if**: Explorar cenários alternativos e suas implicações

## Implementação

### Motor de Raciocínio Básico

```python
from typing import Dict, List, Set, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re
from abc import ABC, abstractmethod

class FactType(Enum):
    ASSERTION = "assertion"
    RULE = "rule"
    QUERY = "query"

@dataclass
class Fact:
    id: str
    content: str
    fact_type: FactType
    confidence: float = 1.0
    timestamp: float = 0.0
    source: str = "unknown"

@dataclass
class Rule:
    id: str
    premises: List[str]
    conclusion: str
    confidence: float = 1.0
    description: str = ""

class ReasoningEngine:
    def __init__(self):
        self.facts: Dict[str, Fact] = {}
        self.rules: Dict[str, Rule] = {}
        self.derived_facts: Dict[str, Fact] = {}
        self.reasoning_trace: List[Dict] = []

    def add_fact(self, fact_id: str, content: str, confidence: float = 1.0,
                 source: str = "user"):
        """Add a fact to the knowledge base"""
        fact = Fact(
            id=fact_id,
            content=content,
            fact_type=FactType.ASSERTION,
            confidence=confidence,
            source=source
        )
        self.facts[fact_id] = fact

    def add_rule(self, rule_id: str, premises: List[str], conclusion: str,
                 confidence: float = 1.0, description: str = ""):
        """Add an inference rule"""
        rule = Rule(
            id=rule_id,
            premises=premises,
            conclusion=conclusion,
            confidence=confidence,
            description=description
        )
        self.rules[rule_id] = rule

    def forward_chain(self, max_iterations: int = 100) -> List[Fact]:
        """Forward chaining inference"""
        new_facts = []
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            facts_added_this_iteration = False

            for rule_id, rule in self.rules.items():
                if self._can_fire_rule(rule):
                    new_fact = self._fire_rule(rule)
                    if new_fact and new_fact.id not in self.derived_facts:
                        self.derived_facts[new_fact.id] = new_fact
                        new_facts.append(new_fact)
                        facts_added_this_iteration = True

                        # Record reasoning step
                        self.reasoning_trace.append({
                            'type': 'rule_application',
                            'rule_id': rule_id,
                            'premises': rule.premises,
                            'conclusion': new_fact.content,
                            'confidence': new_fact.confidence,
                            'iteration': iteration
                        })

            if not facts_added_this_iteration:
                break

        return new_facts

    def backward_chain(self, goal: str) -> Tuple[bool, List[Dict]]:
        """Backward chaining to prove a goal"""
        proof_trace = []
        return self._prove_goal(goal, proof_trace, set())

    def _prove_goal(self, goal: str, trace: List[Dict],
                   visited: Set[str]) -> Tuple[bool, List[Dict]]:
        """Recursively prove a goal using backward chaining"""
        if goal in visited:
            return False, trace  # Avoid infinite loops

        visited.add(goal)

        # Check if goal is already a known fact
        if self._is_known_fact(goal):
            trace.append({
                'type': 'fact_lookup',
                'goal': goal,
                'status': 'proven',
                'method': 'direct_fact'
            })
            return True, trace

        # Try to prove goal using rules
        for rule_id, rule in self.rules.items():
            if self._matches_conclusion(rule.conclusion, goal):
                # Try to prove all premises
                all_premises_proven = True
                premises_trace = []

                for premise in rule.premises:
                    proven, premise_trace = self._prove_goal(premise, [], visited.copy())
                    premises_trace.extend(premise_trace)

                    if not proven:
                        all_premises_proven = False
                        break

                if all_premises_proven:
                    trace.append({
                        'type': 'rule_application',
                        'rule_id': rule_id,
                        'goal': goal,
                        'premises': rule.premises,
                        'status': 'proven',
                        'premises_trace': premises_trace
                    })
                    return True, trace

        # Goal could not be proven
        trace.append({
            'type': 'goal_failure',
            'goal': goal,
            'status': 'unproven'
        })
        return False, trace

    def _can_fire_rule(self, rule: Rule) -> bool:
        """Check if all premises of a rule are satisfied"""
        all_known_facts = {**self.facts, **self.derived_facts}

        for premise in rule.premises:
            if not any(self._matches_fact(fact.content, premise)
                      for fact in all_known_facts.values()):
                return False
        return True

    def _fire_rule(self, rule: Rule) -> Optional[Fact]:
        """Apply a rule to derive a new fact"""
        # Calculate confidence of derived fact
        premise_confidences = []
        all_known_facts = {**self.facts, **self.derived_facts}

        for premise in rule.premises:
            for fact in all_known_facts.values():
                if self._matches_fact(fact.content, premise):
                    premise_confidences.append(fact.confidence)
                    break

        if premise_confidences:
            # Use minimum confidence of premises multiplied by rule confidence
            derived_confidence = min(premise_confidences) * rule.confidence

            fact_id = f"derived_{len(self.derived_facts)}"
            return Fact(
                id=fact_id,
                content=rule.conclusion,
                fact_type=FactType.ASSERTION,
                confidence=derived_confidence,
                source=f"rule_{rule.id}"
            )

        return None

    def _is_known_fact(self, statement: str) -> bool:
        """Check if a statement is a known fact"""
        all_known_facts = {**self.facts, **self.derived_facts}
        return any(self._matches_fact(fact.content, statement)
                  for fact in all_known_facts.values())

    def _matches_fact(self, fact_content: str, pattern: str) -> bool:
        """Check if a fact matches a pattern (simple string matching for now)"""
        return fact_content.strip().lower() == pattern.strip().lower()

    def _matches_conclusion(self, conclusion: str, goal: str) -> bool:
        """Check if a rule conclusion matches a goal"""
        return self._matches_fact(conclusion, goal)

    def get_explanation(self, fact_id: str) -> Dict:
        """Get explanation for how a fact was derived"""
        explanation = {
            'fact_id': fact_id,
            'reasoning_steps': []
        }

        # Find reasoning steps that led to this fact
        for step in self.reasoning_trace:
            if (step.get('type') == 'rule_application' and
                fact_id in self.derived_facts and
                self._matches_fact(step.get('conclusion', ''),
                                 self.derived_facts[fact_id].content)):
                explanation['reasoning_steps'].append(step)

        return explanation
```

### Motor de Raciocínio Probabilístico

```python
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import beta

class BayesianNetwork:
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str]] = []
        self.evidence: Dict[str, bool] = {}

    def add_node(self, name: str, states: List[str],
                 conditional_probabilities: Dict = None):
        """Add a node to the Bayesian network"""
        self.nodes[name] = {
            'states': states,
            'parents': [],
            'children': [],
            'cpt': conditional_probabilities or {}  # Conditional Probability Table
        }

    def add_edge(self, parent: str, child: str):
        """Add a directed edge (causal relationship)"""
        if parent in self.nodes and child in self.nodes:
            self.edges.append((parent, child))
            self.nodes[parent]['children'].append(child)
            self.nodes[child]['parents'].append(parent)

    def set_evidence(self, node: str, value: bool):
        """Set observed evidence for a node"""
        self.evidence[node] = value

    def probability_given_evidence(self, query_node: str,
                                  query_value: bool) -> float:
        """Calculate probability of query given evidence using enumeration"""
        # Simple implementation for binary variables
        # In practice, would use more sophisticated algorithms like variable elimination
        return self._enumerate_inference(query_node, query_value)

    def _enumerate_inference(self, query_node: str, query_value: bool) -> float:
        """Enumerate all possible worlds to calculate probability"""
        # Simplified implementation for demonstration
        total_probability = 0.0
        query_probability = 0.0

        # Generate all possible assignments to non-evidence variables
        non_evidence_nodes = [node for node in self.nodes if node not in self.evidence]

        for assignment in self._generate_assignments(non_evidence_nodes):
            full_assignment = {**self.evidence, **assignment}
            world_probability = self._calculate_world_probability(full_assignment)

            total_probability += world_probability

            if full_assignment.get(query_node) == query_value:
                query_probability += world_probability

        return query_probability / total_probability if total_probability > 0 else 0.0

    def _generate_assignments(self, nodes: List[str]):
        """Generate all possible truth assignments for nodes"""
        if not nodes:
            yield {}
            return

        node = nodes[0]
        remaining_nodes = nodes[1:]

        for value in [True, False]:
            for assignment in self._generate_assignments(remaining_nodes):
                yield {node: value, **assignment}

    def _calculate_world_probability(self, assignment: Dict[str, bool]) -> float:
        """Calculate probability of a complete world assignment"""
        probability = 1.0

        for node, value in assignment.items():
            node_prob = self._get_node_probability(node, value, assignment)
            probability *= node_prob

        return probability

    def _get_node_probability(self, node: str, value: bool,
                             assignment: Dict[str, bool]) -> float:
        """Get probability of node value given parent assignments"""
        parents = self.nodes[node]['parents']

        if not parents:
            # Root node - use prior probability
            return self.nodes[node]['cpt'].get(value, 0.5)

        # Get parent assignment
        parent_assignment = tuple(assignment[parent] for parent in parents)

        # Look up conditional probability
        cpt_key = (parent_assignment, value)
        return self.nodes[node]['cpt'].get(cpt_key, 0.5)

class ProbabilisticReasoningAgent:
    def __init__(self):
        self.bayesian_network = BayesianNetwork()
        self.belief_state: Dict[str, float] = {}
        self.uncertainty_threshold = 0.1

    def add_probabilistic_knowledge(self, hypothesis: str, evidence: List[str],
                                  probabilities: Dict):
        """Add probabilistic knowledge to the network"""
        # Add hypothesis node
        self.bayesian_network.add_node(hypothesis, ['true', 'false'], probabilities)

        # Add evidence nodes and connect them
        for evidence_item in evidence:
            if evidence_item not in self.bayesian_network.nodes:
                self.bayesian_network.add_node(evidence_item, ['true', 'false'])

            self.bayesian_network.add_edge(hypothesis, evidence_item)

    def update_beliefs(self, observations: Dict[str, bool]):
        """Update beliefs based on new observations"""
        # Set evidence
        for obs_node, obs_value in observations.items():
            self.bayesian_network.set_evidence(obs_node, obs_value)

        # Update belief state for all hypotheses
        for node in self.bayesian_network.nodes:
            if node not in observations:  # Don't update observed nodes
                prob_true = self.bayesian_network.probability_given_evidence(node, True)
                self.belief_state[node] = prob_true

    def get_most_likely_explanation(self, observations: Dict[str, bool]) -> Dict:
        """Find the most likely explanation for observations"""
        self.update_beliefs(observations)

        # Find hypothesis with highest probability
        explanations = []
        for hypothesis, probability in self.belief_state.items():
            if probability > 0.5:  # More likely than not
                explanations.append({
                    'hypothesis': hypothesis,
                    'probability': probability,
                    'confidence': 'high' if probability > 0.8 else 'medium'
                })

        return {
            'explanations': sorted(explanations,
                                 key=lambda x: x['probability'], reverse=True),
            'uncertainty': max(0.5 - max(self.belief_state.values(), default=0), 0)
        }
```

### Sistema de Raciocínio Analógico

```python
class AnalogicalReasoningEngine:
    def __init__(self):
        self.case_base: List[Dict] = []
        self.similarity_threshold = 0.7

    def add_case(self, case_id: str, problem_description: Dict,
                 solution: Dict, outcome: Dict):
        """Add a case to the case base"""
        case = {
            'id': case_id,
            'problem': problem_description,
            'solution': solution,
            'outcome': outcome,
            'features': self._extract_features(problem_description)
        }
        self.case_base.append(case)

    def find_analogous_cases(self, target_problem: Dict, top_k: int = 3) -> List[Dict]:
        """Find cases analogous to the target problem"""
        target_features = self._extract_features(target_problem)

        similarities = []
        for case in self.case_base:
            similarity = self._calculate_similarity(target_features, case['features'])
            if similarity >= self.similarity_threshold:
                similarities.append({
                    'case': case,
                    'similarity': similarity
                })

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def adapt_solution(self, analogous_case: Dict, target_problem: Dict) -> Dict:
        """Adapt solution from analogous case to target problem"""
        base_solution = analogous_case['solution']
        adaptations = []

        # Simple adaptation based on feature differences
        base_features = analogous_case['features']
        target_features = self._extract_features(target_problem)

        adapted_solution = base_solution.copy()

        for feature, target_value in target_features.items():
            if feature in base_features:
                base_value = base_features[feature]
                if target_value != base_value:
                    adaptation = self._adapt_for_feature_difference(
                        feature, base_value, target_value, base_solution
                    )
                    adapted_solution.update(adaptation)
                    adaptations.append({
                        'feature': feature,
                        'from': base_value,
                        'to': target_value,
                        'adaptation': adaptation
                    })

        return {
            'adapted_solution': adapted_solution,
            'adaptations': adaptations,
            'confidence': self._calculate_adaptation_confidence(adaptations)
        }

    def _extract_features(self, problem_description: Dict) -> Dict:
        """Extract relevant features from problem description"""
        features = {}

        # Extract various types of features
        for key, value in problem_description.items():
            if isinstance(value, (int, float)):
                features[f"numeric_{key}"] = value
            elif isinstance(value, str):
                features[f"categorical_{key}"] = value
            elif isinstance(value, list):
                features[f"list_size_{key}"] = len(value)
                if value and isinstance(value[0], str):
                    features[f"list_content_{key}"] = set(value)

        return features

    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two feature sets"""
        common_features = set(features1.keys()) & set(features2.keys())

        if not common_features:
            return 0.0

        total_similarity = 0.0
        for feature in common_features:
            value1 = features1[feature]
            value2 = features2[feature]

            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                # Numeric similarity
                max_val = max(abs(value1), abs(value2), 1)
                similarity = 1.0 - abs(value1 - value2) / max_val
            elif isinstance(value1, str) and isinstance(value2, str):
                # String similarity
                similarity = 1.0 if value1 == value2 else 0.0
            elif isinstance(value1, set) and isinstance(value2, set):
                # Set similarity (Jaccard index)
                intersection = len(value1 & value2)
                union = len(value1 | value2)
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 1.0 if value1 == value2 else 0.0

            total_similarity += similarity

        return total_similarity / len(common_features)

    def _adapt_for_feature_difference(self, feature: str, base_value: Any,
                                    target_value: Any, base_solution: Dict) -> Dict:
        """Adapt solution based on a specific feature difference"""
        adaptation = {}

        # Simple rule-based adaptations
        if feature.startswith('numeric_'):
            if isinstance(base_value, (int, float)) and isinstance(target_value, (int, float)):
                ratio = target_value / base_value if base_value != 0 else 1.0

                # Scale numeric solution parameters
                for sol_key, sol_value in base_solution.items():
                    if isinstance(sol_value, (int, float)):
                        adaptation[sol_key] = sol_value * ratio

        elif feature.startswith('categorical_'):
            # Category-specific adaptations
            adaptation_rules = {
                'size': {'small': 0.5, 'medium': 1.0, 'large': 1.5},
                'complexity': {'low': 0.7, 'medium': 1.0, 'high': 1.3}
            }

            for rule_type, multipliers in adaptation_rules.items():
                if rule_type in feature:
                    base_mult = multipliers.get(base_value, 1.0)
                    target_mult = multipliers.get(target_value, 1.0)
                    ratio = target_mult / base_mult

                    for sol_key, sol_value in base_solution.items():
                        if isinstance(sol_value, (int, float)):
                            adaptation[sol_key] = sol_value * ratio

        return adaptation

    def _calculate_adaptation_confidence(self, adaptations: List[Dict]) -> float:
        """Calculate confidence in the adapted solution"""
        if not adaptations:
            return 1.0

        # Confidence decreases with number and magnitude of adaptations
        confidence = 1.0
        for adaptation in adaptations:
            # Simple heuristic: each adaptation reduces confidence
            confidence *= 0.9

        return max(confidence, 0.1)  # Minimum confidence threshold

class ReasoningAgent:
    def __init__(self):
        self.deductive_engine = ReasoningEngine()
        self.probabilistic_engine = ProbabilisticReasoningAgent()
        self.analogical_engine = AnalogicalReasoningEngine()
        self.reasoning_history: List[Dict] = []

    async def reason_about_problem(self, problem: Dict, reasoning_type: str = "mixed") -> Dict:
        """Apply appropriate reasoning techniques to solve a problem"""
        result = {
            'problem': problem,
            'reasoning_type': reasoning_type,
            'conclusions': [],
            'confidence': 0.0,
            'explanation': []
        }

        if reasoning_type in ["deductive", "mixed"]:
            deductive_result = await self._apply_deductive_reasoning(problem)
            if deductive_result:
                result['conclusions'].extend(deductive_result['conclusions'])
                result['explanation'].extend(deductive_result['explanation'])

        if reasoning_type in ["probabilistic", "mixed"]:
            probabilistic_result = await self._apply_probabilistic_reasoning(problem)
            if probabilistic_result:
                result['conclusions'].extend(probabilistic_result['conclusions'])
                result['explanation'].extend(probabilistic_result['explanation'])

        if reasoning_type in ["analogical", "mixed"]:
            analogical_result = await self._apply_analogical_reasoning(problem)
            if analogical_result:
                result['conclusions'].extend(analogical_result['conclusions'])
                result['explanation'].extend(analogical_result['explanation'])

        # Calculate overall confidence
        if result['conclusions']:
            confidences = [c.get('confidence', 0.5) for c in result['conclusions']]
            result['confidence'] = sum(confidences) / len(confidences)

        # Record reasoning process
        self.reasoning_history.append(result)

        return result

    async def _apply_deductive_reasoning(self, problem: Dict) -> Optional[Dict]:
        """Apply deductive reasoning to the problem"""
        # Extract facts and goals from problem description
        facts = problem.get('facts', [])
        goal = problem.get('goal')

        if not goal:
            return None

        # Add facts to reasoning engine
        for i, fact in enumerate(facts):
            self.deductive_engine.add_fact(f"fact_{i}", fact)

        # Try to prove the goal
        proven, trace = self.deductive_engine.backward_chain(goal)

        if proven:
            return {
                'conclusions': [{
                    'type': 'deductive',
                    'conclusion': goal,
                    'proven': True,
                    'confidence': 1.0
                }],
                'explanation': [f"Deductive proof: {goal} can be proven from given facts"]
            }

        return None

    async def _apply_probabilistic_reasoning(self, problem: Dict) -> Optional[Dict]:
        """Apply probabilistic reasoning to the problem"""
        observations = problem.get('observations', {})
        hypotheses = problem.get('hypotheses', [])

        if not observations or not hypotheses:
            return None

        # Update beliefs based on observations
        self.probabilistic_engine.update_beliefs(observations)

        # Get most likely explanations
        explanations = self.probabilistic_engine.get_most_likely_explanation(observations)

        conclusions = []
        for explanation in explanations['explanations']:
            conclusions.append({
                'type': 'probabilistic',
                'conclusion': explanation['hypothesis'],
                'confidence': explanation['probability']
            })

        return {
            'conclusions': conclusions,
            'explanation': [f"Probabilistic analysis of observations: {observations}"]
        }

    async def _apply_analogical_reasoning(self, problem: Dict) -> Optional[Dict]:
        """Apply analogical reasoning to the problem"""
        problem_description = problem.get('description', {})

        if not problem_description:
            return None

        # Find analogous cases
        analogous_cases = self.analogical_engine.find_analogous_cases(problem_description)

        if not analogous_cases:
            return None

        # Adapt solution from most similar case
        best_case = analogous_cases[0]
        adapted_solution = self.analogical_engine.adapt_solution(
            best_case['case'], problem_description
        )

        return {
            'conclusions': [{
                'type': 'analogical',
                'conclusion': adapted_solution['adapted_solution'],
                'confidence': adapted_solution['confidence']
            }],
            'explanation': [
                f"Analogical reasoning based on similar case: {best_case['case']['id']}",
                f"Similarity: {best_case['similarity']:.2f}"
            ]
        }
```

## Exemplos de Código

### Raciocínio Temporal Avançado

```python
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class TemporalEvent:
    id: str
    description: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    event_type: str = "instant"
    confidence: float = 1.0

class TemporalReasoningEngine:
    def __init__(self):
        self.events: List[TemporalEvent] = []
        self.temporal_relations = [
            "before", "after", "during", "overlaps", "meets",
            "starts", "finishes", "equals"
        ]

    def add_event(self, event: TemporalEvent):
        """Add a temporal event"""
        self.events.append(event)

    def find_temporal_relations(self, event1_id: str, event2_id: str) -> List[str]:
        """Find temporal relations between two events"""
        event1 = self._get_event(event1_id)
        event2 = self._get_event(event2_id)

        if not event1 or not event2:
            return []

        relations = []

        # Allen's interval algebra relations
        if event1.end_time and event2.start_time:
            if event1.end_time < event2.start_time:
                relations.append("before")
            elif event1.start_time > event2.end_time:
                relations.append("after")
            elif (event1.start_time <= event2.start_time and
                  event1.end_time >= event2.end_time):
                relations.append("contains")
            elif (event2.start_time <= event1.start_time and
                  event2.end_time >= event1.end_time):
                relations.append("during")

        return relations

    def infer_causality(self, cause_event_id: str, effect_event_id: str) -> Dict:
        """Infer potential causal relationship between events"""
        cause_event = self._get_event(cause_event_id)
        effect_event = self._get_event(effect_event_id)

        if not cause_event or not effect_event:
            return {'causal': False, 'confidence': 0.0}

        # Temporal precedence is necessary for causality
        if cause_event.start_time >= effect_event.start_time:
            return {'causal': False, 'confidence': 0.0, 'reason': 'temporal_order'}

        # Calculate temporal proximity (closer in time = higher causal likelihood)
        time_diff = effect_event.start_time - cause_event.start_time
        proximity_score = max(0, 1.0 - time_diff.total_seconds() / (24 * 3600))  # 1 day max

        # Consider event types for causal plausibility
        causal_plausibility = self._assess_causal_plausibility(
            cause_event.event_type, effect_event.event_type
        )

        overall_confidence = (proximity_score + causal_plausibility) / 2

        return {
            'causal': overall_confidence > 0.5,
            'confidence': overall_confidence,
            'factors': {
                'temporal_proximity': proximity_score,
                'causal_plausibility': causal_plausibility
            }
        }

    def _get_event(self, event_id: str) -> Optional[TemporalEvent]:
        """Get event by ID"""
        for event in self.events:
            if event.id == event_id:
                return event
        return None

    def _assess_causal_plausibility(self, cause_type: str, effect_type: str) -> float:
        """Assess plausibility of causal relationship between event types"""
        # Simple rule-based causal plausibility
        causal_rules = {
            ('action', 'outcome'): 0.8,
            ('decision', 'action'): 0.7,
            ('external_event', 'reaction'): 0.6,
            ('communication', 'response'): 0.7
        }

        return causal_rules.get((cause_type, effect_type), 0.3)

# Example usage of advanced reasoning
async def reasoning_example():
    # Create reasoning agent
    agent = ReasoningAgent()

    # Add some knowledge to deductive engine
    agent.deductive_engine.add_rule(
        "rule1",
        ["bird(X)", "can_fly(X)"],
        "animal(X)",
        confidence=0.9
    )

    agent.deductive_engine.add_fact("fact1", "bird(tweety)")
    agent.deductive_engine.add_fact("fact2", "can_fly(tweety)")

    # Example problem for reasoning
    problem = {
        'description': {
            'domain': 'animal_classification',
            'complexity': 'medium',
            'data_size': 100
        },
        'facts': ['bird(tweety)', 'can_fly(tweety)'],
        'goal': 'animal(tweety)',
        'observations': {'tweety_flies': True, 'tweety_has_feathers': True},
        'hypotheses': ['bird', 'mammal', 'reptile']
    }

    # Apply mixed reasoning
    result = await agent.reason_about_problem(problem, "mixed")

    print("Reasoning Result:")
    print(f"Conclusions: {result['conclusions']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Explanation: {result['explanation']}")

# Run example
# asyncio.run(reasoning_example())
```

## Melhores Práticas

### Representação de Conhecimento
- **Formatos Estruturados**: Use linguagens e padrões formais de representação de conhecimento
- **Ontologias Consistentes**: Mantenha vocabulário consistente e frameworks conceituais
- **Conhecimento Modular**: Organize conhecimento em módulos reutilizáveis específicos de domínio
- **Controle de Versão**: Rastreie mudanças e evolução das bases de conhecimento

### Otimização de Inferência
- **Algoritmos Eficientes**: Use algoritmos de inferência otimizados para melhor performance
- **Estratégias de Poda**: Elimine caminhos de busca irrelevantes cedo no raciocínio
- **Cache de Resultados**: Armazene e reutilize inferências previamente computadas
- **Processamento Paralelo**: Distribua tarefas de raciocínio através de múltiplos processadores

### Gerenciamento de Incerteza
- **Rastreamento de Confiança**: Mantenha níveis de confiança ao longo das cadeias de raciocínio
- **Propagação de Incerteza**: Combine e propague adequadamente estimativas de incerteza
- **Análise de Sensibilidade**: Teste robustez das conclusões a entradas incertas
- **Múltiplas Hipóteses**: Considere explicações alternativas e suas probabilidades relativas

### Geração de Explicações
- **Raciocínio Transparente**: Forneça explicações claras dos processos de raciocínio
- **Múltiplos Níveis**: Ofereça explicações em diferentes níveis de detalhe
- **Exploração Interativa**: Permita que usuários explorem passos de raciocínio interativamente
- **Análise Contrafactual**: Mostre como conclusões mudariam com diferentes suposições

## Armadilhas Comuns

### Explosão Combinatorial
- **Problema**: Espaço de busca de raciocínio cresce exponencialmente com complexidade do problema
- **Solução**: Use heurísticas, poda e métodos de raciocínio aproximado

### Conhecimento Inconsistente
- **Problema**: Fatos ou regras contraditórios levando a conclusões inconsistentes
- **Solução**: Implemente verificação de consistência e mecanismos de revisão de crenças

### Overfitting a Exemplos
- **Problema**: Padrões de raciocínio que funcionam para casos de treinamento mas falham em novos problemas
- **Solução**: Use exemplos diversos e teste raciocínio em conjuntos de problemas reservados

### Fragilidade
- **Problema**: Sistemas de raciocínio falhando catastroficamente com entradas ligeiramente diferentes
- **Solução**: Implemente raciocínio robusto com degradação graceful e tratamento de incerteza

### Complexidade Computacional
- **Problema**: Raciocínio demorado demais para aplicações em tempo real
- **Solução**: Use algoritmos anytime, métodos de aproximação e orçamentos computacionais

### Gargalo de Aquisição de Conhecimento
- **Problema**: Dificuldade em adquirir e manter grandes bases de conhecimento
- **Solução**: Implemente extração automatizada de conhecimento e mecanismos de aprendizado

---

*Este capítulo aborda 24 páginas de conteúdo de "Agentic Design Patterns" por Antonio Gulli, focando em Técnicas de Raciocínio para construir agentes de IA inteligentes capazes de inferência lógica, raciocínio probabilístico e resolução estruturada de problemas.*

---

**Nota de Tradução**: Este documento foi traduzido do inglês para português brasileiro. O conteúdo técnico original foi preservado, mantendo termos técnicos estabelecidos em inglês quando apropriado.