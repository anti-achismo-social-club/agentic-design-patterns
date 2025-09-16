# Chapter 17: Reasoning Techniques

**Pattern Description:** Reasoning Techniques enable AI agents to perform logical inference, causal reasoning, and structured problem-solving through systematic approaches including deductive reasoning, inductive reasoning, abductive reasoning, and analogical reasoning.

## Introduction

Reasoning Techniques represent the cognitive capabilities that enable AI agents to draw conclusions, make inferences, and solve problems systematically. These patterns go beyond simple pattern matching or statistical correlations to implement structured logical processes that mirror human reasoning capabilities.

The importance of reasoning techniques in AI agents cannot be overstated. They enable agents to handle novel situations, make decisions under uncertainty, explain their conclusions, and adapt their behavior based on logical analysis rather than purely learned responses. This is particularly crucial for applications requiring explainable AI, safety-critical decisions, and complex problem-solving.

Modern AI systems increasingly require sophisticated reasoning capabilities to handle real-world complexity, maintain consistency in their conclusions, and provide transparent decision-making processes. These techniques bridge the gap between reactive AI systems and truly intelligent agents capable of deliberative thought.

## Key Concepts

### Types of Reasoning

#### Deductive Reasoning
- **Logical Inference**: Drawing specific conclusions from general principles or premises
- **Rule-Based Systems**: Applying if-then rules to derive new knowledge
- **Formal Logic**: Using propositional and predicate logic for systematic inference
- **Proof Systems**: Constructing formal proofs for mathematical and logical propositions

#### Inductive Reasoning
- **Pattern Recognition**: Identifying general patterns from specific observations
- **Hypothesis Formation**: Generating theories based on empirical evidence
- **Statistical Inference**: Using probabilistic methods to draw conclusions
- **Generalization**: Extending conclusions from samples to populations

#### Abductive Reasoning
- **Best Explanation**: Finding the most likely explanation for observed phenomena
- **Hypothesis Generation**: Creating plausible hypotheses to explain evidence
- **Diagnostic Reasoning**: Identifying causes from observed symptoms or effects
- **Creative Problem Solving**: Generating novel solutions through explanatory reasoning

#### Analogical Reasoning
- **Similarity Mapping**: Identifying structural similarities between different domains
- **Knowledge Transfer**: Applying solutions from familiar domains to new problems
- **Metaphorical Thinking**: Using metaphors and analogies for understanding
- **Case-Based Reasoning**: Solving new problems based on similar past cases

### Reasoning Frameworks

#### Symbolic Reasoning
- **Knowledge Representation**: Formal representation of facts, rules, and relationships
- **Logic Programming**: Using logical languages like Prolog for reasoning
- **Semantic Networks**: Graph-based representation of knowledge relationships
- **Ontological Reasoning**: Using formal ontologies for structured inference

#### Probabilistic Reasoning
- **Bayesian Networks**: Representing probabilistic dependencies between variables
- **Markov Models**: Reasoning about sequences and temporal dependencies
- **Uncertainty Quantification**: Managing and propagating uncertainty in reasoning
- **Evidence Combination**: Integrating multiple sources of uncertain evidence

#### Temporal Reasoning
- **Time Logic**: Reasoning about temporal relationships and sequences
- **Planning**: Generating sequences of actions to achieve goals
- **Causality**: Understanding cause-and-effect relationships over time
- **Event Recognition**: Identifying patterns in temporal event sequences

### Reasoning Architecture Components

#### Knowledge Base
- **Fact Storage**: Repository of known facts and observations
- **Rule Management**: Collection of inference rules and heuristics
- **Belief Maintenance**: Tracking confidence and consistency of beliefs
- **Knowledge Updates**: Mechanisms for incorporating new information

#### Inference Engine
- **Forward Chaining**: Reasoning from facts to conclusions
- **Backward Chaining**: Reasoning from goals to supporting evidence
- **Resolution Methods**: Systematic proof procedures for logical inference
- **Conflict Resolution**: Handling contradictory or competing inferences

#### Explanation System
- **Trace Generation**: Recording the reasoning process for transparency
- **Justification**: Providing explanations for conclusions and decisions
- **Confidence Assessment**: Quantifying certainty in reasoning outcomes
- **What-if Analysis**: Exploring alternative scenarios and their implications

## Implementation

### Basic Reasoning Engine

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

### Probabilistic Reasoning Engine

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

### Analogical Reasoning System

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

## Code Examples

### Advanced Temporal Reasoning

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

## Best Practices

### Knowledge Representation
- **Structured Formats**: Use formal knowledge representation languages and standards
- **Consistent Ontologies**: Maintain consistent vocabulary and conceptual frameworks
- **Modular Knowledge**: Organize knowledge into reusable, domain-specific modules
- **Version Control**: Track changes and evolution of knowledge bases

### Inference Optimization
- **Efficient Algorithms**: Use optimized inference algorithms for better performance
- **Pruning Strategies**: Eliminate irrelevant search paths early in reasoning
- **Caching Results**: Store and reuse previously computed inferences
- **Parallel Processing**: Distribute reasoning tasks across multiple processors

### Uncertainty Management
- **Confidence Tracking**: Maintain confidence levels throughout reasoning chains
- **Uncertainty Propagation**: Properly combine and propagate uncertainty estimates
- **Sensitivity Analysis**: Test robustness of conclusions to uncertain inputs
- **Multiple Hypotheses**: Consider alternative explanations and their relative likelihoods

### Explanation Generation
- **Transparent Reasoning**: Provide clear explanations of reasoning processes
- **Multiple Levels**: Offer explanations at different levels of detail
- **Interactive Exploration**: Allow users to explore reasoning steps interactively
- **Counterfactual Analysis**: Show how conclusions would change with different assumptions

## Common Pitfalls

### Combinatorial Explosion
- **Problem**: Reasoning search space grows exponentially with problem complexity
- **Solution**: Use heuristics, pruning, and approximate reasoning methods

### Inconsistent Knowledge
- **Problem**: Contradictory facts or rules leading to inconsistent conclusions
- **Solution**: Implement consistency checking and belief revision mechanisms

### Overfitting to Examples
- **Problem**: Reasoning patterns that work for training cases but fail on new problems
- **Solution**: Use diverse examples and test reasoning on held-out problem sets

### Brittleness
- **Problem**: Reasoning systems failing catastrophically with slightly different inputs
- **Solution**: Implement robust reasoning with graceful degradation and uncertainty handling

### Computational Complexity
- **Problem**: Reasoning taking too long for real-time applications
- **Solution**: Use anytime algorithms, approximation methods, and computational budgets

### Knowledge Acquisition Bottleneck
- **Problem**: Difficulty in acquiring and maintaining large knowledge bases
- **Solution**: Implement automated knowledge extraction and learning mechanisms

---

*This chapter covers 24 pages of content from "Agentic Design Patterns" by Antonio Gulli, focusing on Reasoning Techniques for building intelligent AI agents capable of logical inference, probabilistic reasoning, and structured problem-solving.*