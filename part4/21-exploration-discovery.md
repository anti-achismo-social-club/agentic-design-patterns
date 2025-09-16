# Chapter 21: Exploration and Discovery

**Pattern Description:** Exploration and Discovery patterns enable AI agents to systematically investigate unknown environments, learn from new experiences, and discover novel solutions through strategic exploration, experimentation, and knowledge acquisition.

## Introduction

Exploration and Discovery represent fundamental capabilities that enable AI agents to operate effectively in unknown or dynamic environments. These patterns go beyond exploitation of known strategies to include systematic investigation of new possibilities, learning from novel experiences, and discovering innovative solutions to problems.

The challenge of balancing exploration versus exploitation is central to intelligent agent behavior. Pure exploitation of known strategies may lead to local optima and missed opportunities, while excessive exploration can be inefficient and waste resources. Effective exploration patterns provide structured approaches for agents to navigate this trade-off while maintaining goal-directed behavior.

Modern AI agents often operate in environments where complete information is unavailable, conditions change over time, and novel situations arise frequently. Exploration and Discovery patterns equip agents with the capabilities to adapt, learn, and thrive in such dynamic contexts through systematic investigation and knowledge acquisition.

## Key Concepts

### Exploration Strategies

#### Random Exploration
- **Uniform Random Sampling**: Exploring the action/state space with equal probability
- **Weighted Random Sampling**: Biasing exploration toward promising regions
- **Entropy-Based Exploration**: Using information entropy to guide exploration decisions
- **Noise Injection**: Adding controlled randomness to deterministic policies

#### Systematic Exploration
- **Grid Search**: Systematically exploring parameter spaces in organized patterns
- **Breadth-First Exploration**: Exploring all options at each level before going deeper
- **Depth-First Exploration**: Following exploration paths to completion before trying alternatives
- **Spiral Search**: Exploring in expanding patterns from known good regions

#### Guided Exploration
- **Curiosity-Driven Exploration**: Exploring based on information gain and novelty
- **Uncertainty-Based Exploration**: Focusing exploration on areas with high uncertainty
- **Upper Confidence Bound**: Balancing exploration and exploitation using confidence intervals
- **Thompson Sampling**: Probabilistic exploration based on posterior distributions

#### Multi-Armed Bandit Approaches
- **Epsilon-Greedy**: Simple exploration with fixed probability
- **Decaying Epsilon**: Reducing exploration rate over time
- **UCB1**: Upper Confidence Bound algorithm for multi-armed bandits
- **Contextual Bandits**: Exploration considering contextual information

### Discovery Mechanisms

#### Pattern Recognition
- **Anomaly Detection**: Identifying unusual patterns or outliers in data
- **Clustering**: Grouping similar observations to discover structure
- **Association Rules**: Finding relationships between different variables or events
- **Sequential Pattern Mining**: Discovering temporal patterns in sequences

#### Hypothesis Generation
- **Abductive Reasoning**: Generating explanations for observed phenomena
- **Analogy-Based Discovery**: Using similarities to known problems for discovery
- **Combinatorial Exploration**: Systematically combining elements to find new solutions
- **Evolutionary Approaches**: Using genetic algorithms for solution discovery

#### Knowledge Acquisition
- **Active Learning**: Strategically selecting data points for learning
- **Transfer Learning**: Applying knowledge from related domains
- **Meta-Learning**: Learning how to learn more effectively
- **Continuous Learning**: Incrementally acquiring knowledge over time

### Exploration-Exploitation Balance

#### Dynamic Balancing
- **Adaptive Strategies**: Adjusting exploration rate based on performance and context
- **Multi-Objective Optimization**: Balancing multiple exploration and exploitation goals
- **Regret Minimization**: Minimizing the cost of exploration while maximizing learning
- **Budget-Constrained Exploration**: Managing exploration within resource constraints

#### Contextual Adaptation
- **Environment Sensing**: Adapting exploration based on environmental characteristics
- **Risk Assessment**: Modifying exploration based on potential risks and rewards
- **Time Horizon Consideration**: Adjusting strategy based on available time
- **Resource Availability**: Scaling exploration effort based on available resources

## Implementation

### Exploration Manager

```python
import random
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import time

class ExplorationStrategy(Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB1 = "ucb1"
    THOMPSON_SAMPLING = "thompson_sampling"
    CURIOSITY_DRIVEN = "curiosity_driven"
    SYSTEMATIC = "systematic"

@dataclass
class ExplorationAction:
    id: str
    parameters: Dict[str, Any]
    expected_reward: float = 0.0
    uncertainty: float = 1.0
    exploration_count: int = 0
    last_explored: float = 0.0

@dataclass
class ExplorationResult:
    action: ExplorationAction
    reward: float
    observations: Dict[str, Any]
    timestamp: float
    exploration_cost: float = 0.0

class ExplorationManager:
    def __init__(self, strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY):
        self.strategy = strategy
        self.actions: Dict[str, ExplorationAction] = {}
        self.exploration_history: List[ExplorationResult] = []
        self.total_explorations = 0

        # Strategy-specific parameters
        self.epsilon = 0.1  # For epsilon-greedy
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.c = 1.0  # For UCB1
        self.curiosity_threshold = 0.5

        # State tracking
        self.current_context: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, float] = {}

    def add_action(self, action: ExplorationAction):
        """Add an action to the exploration space"""
        self.actions[action.id] = action

    def select_action(self, context: Dict[str, Any] = None) -> ExplorationAction:
        """Select an action for exploration based on the current strategy"""
        self.current_context = context or {}

        if not self.actions:
            raise ValueError("No actions available for exploration")

        if self.strategy == ExplorationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy_selection()
        elif self.strategy == ExplorationStrategy.UCB1:
            return self._ucb1_selection()
        elif self.strategy == ExplorationStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling_selection()
        elif self.strategy == ExplorationStrategy.CURIOSITY_DRIVEN:
            return self._curiosity_driven_selection()
        elif self.strategy == ExplorationStrategy.SYSTEMATIC:
            return self._systematic_selection()
        else:
            return random.choice(list(self.actions.values()))

    def record_result(self, action_id: str, reward: float,
                     observations: Dict[str, Any] = None,
                     exploration_cost: float = 0.0):
        """Record the result of an exploration action"""
        if action_id not in self.actions:
            return

        action = self.actions[action_id]
        result = ExplorationResult(
            action=action,
            reward=reward,
            observations=observations or {},
            timestamp=time.time(),
            exploration_cost=exploration_cost
        )

        # Update action statistics
        action.exploration_count += 1
        action.last_explored = result.timestamp

        # Update expected reward using incremental average
        n = action.exploration_count
        action.expected_reward = ((n - 1) * action.expected_reward + reward) / n

        # Update uncertainty (decreases with more observations)
        action.uncertainty = max(0.1, 1.0 / math.sqrt(n))

        # Store result
        self.exploration_history.append(result)
        self.total_explorations += 1

        # Update strategy parameters
        self._update_strategy_parameters()

    def _epsilon_greedy_selection(self) -> ExplorationAction:
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(list(self.actions.values()))
        else:
            # Exploit: best known action
            return max(self.actions.values(), key=lambda a: a.expected_reward)

    def _ucb1_selection(self) -> ExplorationAction:
        """Upper Confidence Bound (UCB1) action selection"""
        if self.total_explorations == 0:
            return random.choice(list(self.actions.values()))

        best_action = None
        best_ucb_value = float('-inf')

        for action in self.actions.values():
            if action.exploration_count == 0:
                return action  # Prioritize unexplored actions

            # UCB1 formula
            confidence_radius = self.c * math.sqrt(
                math.log(self.total_explorations) / action.exploration_count
            )
            ucb_value = action.expected_reward + confidence_radius

            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_action = action

        return best_action or random.choice(list(self.actions.values()))

    def _thompson_sampling_selection(self) -> ExplorationAction:
        """Thompson Sampling action selection"""
        best_action = None
        best_sample = float('-inf')

        for action in self.actions.values():
            # Sample from beta distribution (assuming binary rewards)
            # In practice, you'd use appropriate distributions for your reward model
            alpha = max(1, action.exploration_count * action.expected_reward + 1)
            beta = max(1, action.exploration_count * (1 - action.expected_reward) + 1)

            sample = np.random.beta(alpha, beta)

            if sample > best_sample:
                best_sample = sample
                best_action = action

        return best_action or random.choice(list(self.actions.values()))

    def _curiosity_driven_selection(self) -> ExplorationAction:
        """Curiosity-driven action selection based on novelty and uncertainty"""
        best_action = None
        best_curiosity_score = float('-inf')

        current_time = time.time()

        for action in self.actions.values():
            # Novelty score (higher for less explored actions)
            novelty_score = 1.0 / (action.exploration_count + 1)

            # Recency score (higher for actions not explored recently)
            time_since_exploration = current_time - action.last_explored
            recency_score = min(1.0, time_since_exploration / 3600)  # Normalize by hour

            # Uncertainty score
            uncertainty_score = action.uncertainty

            # Combined curiosity score
            curiosity_score = (novelty_score + recency_score + uncertainty_score) / 3

            if curiosity_score > best_curiosity_score:
                best_curiosity_score = curiosity_score
                best_action = action

        return best_action or random.choice(list(self.actions.values()))

    def _systematic_selection(self) -> ExplorationAction:
        """Systematic exploration to ensure coverage"""
        # Find least explored action
        least_explored = min(self.actions.values(), key=lambda a: a.exploration_count)

        # If all actions have been explored equally, choose by uncertainty
        min_count = least_explored.exploration_count
        candidates = [a for a in self.actions.values()
                     if a.exploration_count == min_count]

        if len(candidates) > 1:
            return max(candidates, key=lambda a: a.uncertainty)
        else:
            return least_explored

    def _update_strategy_parameters(self):
        """Update strategy-specific parameters based on exploration history"""
        if self.strategy == ExplorationStrategy.EPSILON_GREEDY:
            # Decay epsilon over time
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get comprehensive exploration statistics"""
        if not self.exploration_history:
            return {'message': 'No exploration data available'}

        recent_history = self.exploration_history[-100:]  # Last 100 explorations
        total_reward = sum(r.reward for r in recent_history)
        avg_reward = total_reward / len(recent_history)

        action_stats = {}
        for action_id, action in self.actions.items():
            action_results = [r for r in recent_history if r.action.id == action_id]
            if action_results:
                action_reward = sum(r.reward for r in action_results) / len(action_results)
            else:
                action_reward = 0.0

            action_stats[action_id] = {
                'exploration_count': action.exploration_count,
                'expected_reward': action.expected_reward,
                'uncertainty': action.uncertainty,
                'recent_avg_reward': action_reward
            }

        return {
            'total_explorations': self.total_explorations,
            'recent_avg_reward': avg_reward,
            'current_epsilon': self.epsilon if self.strategy == ExplorationStrategy.EPSILON_GREEDY else None,
            'action_statistics': action_stats,
            'strategy': self.strategy.value
        }

class DiscoveryEngine:
    def __init__(self):
        self.patterns: Dict[str, Dict] = {}
        self.hypotheses: List[Dict] = []
        self.discovered_knowledge: List[Dict] = []
        self.observation_history: List[Dict] = []

    def add_observation(self, observation: Dict[str, Any]):
        """Add an observation for pattern discovery"""
        observation['timestamp'] = time.time()
        self.observation_history.append(observation)

        # Trigger pattern discovery
        self._discover_patterns()

    def _discover_patterns(self):
        """Discover patterns in observations"""
        if len(self.observation_history) < 10:
            return  # Need more data

        # Frequency pattern discovery
        self._discover_frequency_patterns()

        # Sequential pattern discovery
        self._discover_sequential_patterns()

        # Correlation discovery
        self._discover_correlations()

    def _discover_frequency_patterns(self):
        """Discover frequently occurring patterns"""
        # Simple frequency analysis
        feature_counts = defaultdict(int)

        for obs in self.observation_history[-100:]:  # Recent observations
            for key, value in obs.items():
                if isinstance(value, (str, int, bool)):
                    feature_counts[f"{key}={value}"] += 1

        # Find frequent patterns (threshold: appears in >20% of observations)
        threshold = len(self.observation_history[-100:]) * 0.2

        for pattern, count in feature_counts.items():
            if count > threshold:
                pattern_id = f"freq_{hash(pattern)}"
                if pattern_id not in self.patterns:
                    self.patterns[pattern_id] = {
                        'type': 'frequency',
                        'pattern': pattern,
                        'count': count,
                        'confidence': count / len(self.observation_history[-100:]),
                        'discovered_at': time.time()
                    }

    def _discover_sequential_patterns(self):
        """Discover sequential patterns in observations"""
        # Simple sequential pattern discovery
        if len(self.observation_history) < 5:
            return

        # Look for sequences of length 2 and 3
        for seq_length in [2, 3]:
            sequence_counts = defaultdict(int)

            for i in range(len(self.observation_history) - seq_length + 1):
                sequence = []
                for j in range(seq_length):
                    obs = self.observation_history[i + j]
                    # Simplified: use the most significant feature
                    key_feature = self._get_key_feature(obs)
                    if key_feature:
                        sequence.append(key_feature)

                if len(sequence) == seq_length:
                    seq_str = " -> ".join(sequence)
                    sequence_counts[seq_str] += 1

            # Find frequent sequences
            threshold = max(3, len(self.observation_history) * 0.05)

            for sequence, count in sequence_counts.items():
                if count >= threshold:
                    pattern_id = f"seq_{hash(sequence)}"
                    if pattern_id not in self.patterns:
                        self.patterns[pattern_id] = {
                            'type': 'sequential',
                            'pattern': sequence,
                            'count': count,
                            'confidence': count / len(self.observation_history),
                            'discovered_at': time.time()
                        }

    def _discover_correlations(self):
        """Discover correlations between different features"""
        if len(self.observation_history) < 20:
            return

        # Extract numeric features for correlation analysis
        numeric_features = defaultdict(list)

        for obs in self.observation_history[-50:]:
            for key, value in obs.items():
                if isinstance(value, (int, float)) and key != 'timestamp':
                    numeric_features[key].append(value)

        # Calculate correlations between feature pairs
        feature_names = list(numeric_features.keys())
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                feature1, feature2 = feature_names[i], feature_names[j]
                values1 = numeric_features[feature1]
                values2 = numeric_features[feature2]

                if len(values1) == len(values2) and len(values1) > 10:
                    correlation = np.corrcoef(values1, values2)[0, 1]

                    if abs(correlation) > 0.7:  # Strong correlation
                        pattern_id = f"corr_{hash(f'{feature1}_{feature2}')}"
                        if pattern_id not in self.patterns:
                            self.patterns[pattern_id] = {
                                'type': 'correlation',
                                'feature1': feature1,
                                'feature2': feature2,
                                'correlation': correlation,
                                'strength': 'strong' if abs(correlation) > 0.8 else 'moderate',
                                'discovered_at': time.time()
                            }

    def _get_key_feature(self, observation: Dict[str, Any]) -> Optional[str]:
        """Extract the most significant feature from an observation"""
        # Simple heuristic: return the first non-timestamp string or boolean feature
        for key, value in observation.items():
            if key != 'timestamp' and isinstance(value, (str, bool)):
                return f"{key}={value}"
        return None

    def generate_hypothesis(self, pattern_id: str) -> Dict[str, Any]:
        """Generate a hypothesis based on a discovered pattern"""
        if pattern_id not in self.patterns:
            return {}

        pattern = self.patterns[pattern_id]
        hypothesis = {
            'id': f"hyp_{len(self.hypotheses)}",
            'pattern_id': pattern_id,
            'hypothesis': self._formulate_hypothesis(pattern),
            'confidence': pattern.get('confidence', 0.5),
            'generated_at': time.time(),
            'tested': False,
            'validation_results': []
        }

        self.hypotheses.append(hypothesis)
        return hypothesis

    def _formulate_hypothesis(self, pattern: Dict[str, Any]) -> str:
        """Formulate a human-readable hypothesis from a pattern"""
        pattern_type = pattern['type']

        if pattern_type == 'frequency':
            return f"The condition '{pattern['pattern']}' occurs frequently (confidence: {pattern['confidence']:.2f})"

        elif pattern_type == 'sequential':
            return f"The sequence '{pattern['pattern']}' is a common occurrence pattern"

        elif pattern_type == 'correlation':
            direction = "positively" if pattern['correlation'] > 0 else "negatively"
            return f"Feature '{pattern['feature1']}' is {direction} correlated with '{pattern['feature2']}' (r={pattern['correlation']:.2f})"

        return "Unknown pattern type"

    def test_hypothesis(self, hypothesis_id: str, test_data: List[Dict]) -> Dict[str, Any]:
        """Test a hypothesis against new data"""
        hypothesis = next((h for h in self.hypotheses if h['id'] == hypothesis_id), None)
        if not hypothesis:
            return {'error': 'Hypothesis not found'}

        pattern_id = hypothesis['pattern_id']
        pattern = self.patterns.get(pattern_id)

        if not pattern:
            return {'error': 'Associated pattern not found'}

        # Perform hypothesis testing based on pattern type
        test_result = self._perform_hypothesis_test(pattern, test_data)

        # Update hypothesis with test results
        hypothesis['tested'] = True
        hypothesis['validation_results'].append({
            'test_data_size': len(test_data),
            'result': test_result,
            'tested_at': time.time()
        })

        return test_result

    def _perform_hypothesis_test(self, pattern: Dict, test_data: List[Dict]) -> Dict[str, Any]:
        """Perform statistical testing of a pattern against test data"""
        pattern_type = pattern['type']

        if pattern_type == 'frequency':
            return self._test_frequency_pattern(pattern, test_data)
        elif pattern_type == 'sequential':
            return self._test_sequential_pattern(pattern, test_data)
        elif pattern_type == 'correlation':
            return self._test_correlation_pattern(pattern, test_data)

        return {'supported': False, 'reason': 'Unknown pattern type'}

    def _test_frequency_pattern(self, pattern: Dict, test_data: List[Dict]) -> Dict[str, Any]:
        """Test frequency pattern against test data"""
        pattern_str = pattern['pattern']
        key, value = pattern_str.split('=')

        # Count occurrences in test data
        matches = sum(1 for obs in test_data if obs.get(key) == value)
        test_frequency = matches / len(test_data) if test_data else 0

        expected_frequency = pattern['confidence']
        threshold = 0.1  # Allow 10% deviation

        supported = abs(test_frequency - expected_frequency) <= threshold

        return {
            'supported': supported,
            'expected_frequency': expected_frequency,
            'observed_frequency': test_frequency,
            'deviation': abs(test_frequency - expected_frequency)
        }

    def _test_sequential_pattern(self, pattern: Dict, test_data: List[Dict]) -> Dict[str, Any]:
        """Test sequential pattern against test data"""
        # Simplified sequential pattern testing
        pattern_sequence = pattern['pattern'].split(' -> ')
        sequence_length = len(pattern_sequence)

        matches = 0
        total_possible = max(0, len(test_data) - sequence_length + 1)

        for i in range(total_possible):
            sequence_match = True
            for j, expected_feature in enumerate(pattern_sequence):
                obs = test_data[i + j]
                key_feature = self._get_key_feature(obs)
                if key_feature != expected_feature:
                    sequence_match = False
                    break

            if sequence_match:
                matches += 1

        observed_frequency = matches / total_possible if total_possible > 0 else 0
        expected_frequency = pattern['confidence']

        supported = observed_frequency >= expected_frequency * 0.5  # Allow 50% of expected

        return {
            'supported': supported,
            'expected_frequency': expected_frequency,
            'observed_frequency': observed_frequency,
            'matches_found': matches
        }

    def _test_correlation_pattern(self, pattern: Dict, test_data: List[Dict]) -> Dict[str, Any]:
        """Test correlation pattern against test data"""
        feature1 = pattern['feature1']
        feature2 = pattern['feature2']

        values1 = [obs.get(feature1) for obs in test_data
                  if isinstance(obs.get(feature1), (int, float))]
        values2 = [obs.get(feature2) for obs in test_data
                  if isinstance(obs.get(feature2), (int, float))]

        if len(values1) != len(values2) or len(values1) < 10:
            return {
                'supported': False,
                'reason': 'Insufficient numeric data for correlation test'
            }

        observed_correlation = np.corrcoef(values1, values2)[0, 1]
        expected_correlation = pattern['correlation']

        # Check if correlation is in the same direction and reasonably strong
        same_direction = (observed_correlation * expected_correlation) > 0
        sufficient_strength = abs(observed_correlation) > 0.3

        supported = same_direction and sufficient_strength

        return {
            'supported': supported,
            'expected_correlation': expected_correlation,
            'observed_correlation': observed_correlation,
            'same_direction': same_direction,
            'sufficient_strength': sufficient_strength
        }

    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of discovery activities"""
        return {
            'total_observations': len(self.observation_history),
            'patterns_discovered': len(self.patterns),
            'hypotheses_generated': len(self.hypotheses),
            'hypotheses_tested': sum(1 for h in self.hypotheses if h['tested']),
            'knowledge_items': len(self.discovered_knowledge),
            'pattern_types': {
                'frequency': len([p for p in self.patterns.values() if p['type'] == 'frequency']),
                'sequential': len([p for p in self.patterns.values() if p['type'] == 'sequential']),
                'correlation': len([p for p in self.patterns.values() if p['type'] == 'correlation'])
            }
        }
```

## Code Examples

### Curiosity-Driven Agent

```python
class CuriosityDrivenAgent:
    def __init__(self):
        self.exploration_manager = ExplorationManager(ExplorationStrategy.CURIOSITY_DRIVEN)
        self.discovery_engine = DiscoveryEngine()
        self.knowledge_base: Dict[str, Any] = {}
        self.curiosity_threshold = 0.7
        self.exploration_budget = 100
        self.exploration_spent = 0

    async def explore_environment(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Explore the environment driven by curiosity"""
        exploration_results = []

        while self.exploration_spent < self.exploration_budget:
            # Select action based on curiosity
            if not self.exploration_manager.actions:
                # Generate new actions if none exist
                await self._generate_exploration_actions(environment)

            action = self.exploration_manager.select_action(environment)

            # Execute exploration action
            result = await self._execute_exploration(action, environment)

            # Record result
            self.exploration_manager.record_result(
                action.id, result['reward'], result['observations']
            )

            # Add observation to discovery engine
            self.discovery_engine.add_observation(result['observations'])

            exploration_results.append(result)
            self.exploration_spent += result.get('cost', 1)

            # Check if we've discovered something interesting
            if self._is_discovery_interesting(result):
                await self._investigate_discovery(result)

        return {
            'exploration_results': exploration_results,
            'discoveries': self.discovery_engine.get_discovery_summary(),
            'knowledge_gained': len(self.knowledge_base)
        }

    async def _generate_exploration_actions(self, environment: Dict[str, Any]):
        """Generate new exploration actions based on current knowledge"""
        # Example: Generate actions for different parameter combinations
        possible_actions = [
            {'parameter': 'location', 'value': 'north'},
            {'parameter': 'location', 'value': 'south'},
            {'parameter': 'location', 'value': 'east'},
            {'parameter': 'location', 'value': 'west'},
            {'parameter': 'depth', 'value': 'shallow'},
            {'parameter': 'depth', 'value': 'deep'},
            {'parameter': 'tool', 'value': 'sensor_a'},
            {'parameter': 'tool', 'value': 'sensor_b'}
        ]

        for i, action_params in enumerate(possible_actions):
            action = ExplorationAction(
                id=f"explore_{i}",
                parameters=action_params,
                expected_reward=0.5,  # Neutral expectation
                uncertainty=1.0  # High uncertainty initially
            )
            self.exploration_manager.add_action(action)

    async def _execute_exploration(self, action: ExplorationAction,
                                 environment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an exploration action in the environment"""
        # Simulate exploration results
        import random

        # Generate observations based on action parameters
        observations = {
            'action_taken': action.parameters,
            'environment_state': environment.get('state', 'unknown'),
            'resource_discovered': random.choice([True, False]),
            'obstacle_encountered': random.choice([True, False]),
            'novelty_score': random.uniform(0, 1)
        }

        # Calculate reward based on novelty and resource discovery
        reward = 0.5  # Base reward
        if observations['resource_discovered']:
            reward += 0.3
        if observations['novelty_score'] > 0.7:
            reward += 0.2
        if observations['obstacle_encountered']:
            reward -= 0.1

        # Add some noise
        reward += random.uniform(-0.1, 0.1)

        return {
            'reward': max(0, min(1, reward)),
            'observations': observations,
            'cost': 1
        }

    def _is_discovery_interesting(self, result: Dict[str, Any]) -> bool:
        """Determine if a discovery is interesting enough to investigate further"""
        observations = result['observations']

        # High novelty is interesting
        if observations.get('novelty_score', 0) > self.curiosity_threshold:
            return True

        # Resource discovery is interesting
        if observations.get('resource_discovered', False):
            return True

        # Unexpected obstacles are interesting
        if observations.get('obstacle_encountered', False):
            return True

        return False

    async def _investigate_discovery(self, result: Dict[str, Any]):
        """Investigate an interesting discovery more deeply"""
        observations = result['observations']

        # Generate hypothesis about the discovery
        investigation_result = {
            'discovery_type': self._classify_discovery(observations),
            'investigation_depth': 'detailed',
            'follow_up_actions': self._generate_follow_up_actions(observations)
        }

        # Store in knowledge base
        discovery_id = f"discovery_{len(self.knowledge_base)}"
        self.knowledge_base[discovery_id] = {
            'original_result': result,
            'investigation': investigation_result,
            'timestamp': time.time()
        }

    def _classify_discovery(self, observations: Dict[str, Any]) -> str:
        """Classify the type of discovery"""
        if observations.get('resource_discovered'):
            return 'resource_discovery'
        elif observations.get('novelty_score', 0) > 0.8:
            return 'novel_phenomenon'
        elif observations.get('obstacle_encountered'):
            return 'environmental_hazard'
        else:
            return 'general_observation'

    def _generate_follow_up_actions(self, observations: Dict[str, Any]) -> List[str]:
        """Generate follow-up actions for deeper investigation"""
        actions = []

        if observations.get('resource_discovered'):
            actions.extend([
                'analyze_resource_composition',
                'estimate_resource_quantity',
                'assess_extraction_feasibility'
            ])

        if observations.get('obstacle_encountered'):
            actions.extend([
                'map_obstacle_boundaries',
                'analyze_obstacle_nature',
                'find_alternative_routes'
            ])

        if observations.get('novelty_score', 0) > 0.7:
            actions.extend([
                'collect_detailed_measurements',
                'compare_with_known_phenomena',
                'document_unique_characteristics'
            ])

        return actions

    def get_exploration_insights(self) -> Dict[str, Any]:
        """Get insights from exploration activities"""
        stats = self.exploration_manager.get_exploration_stats()
        discovery_summary = self.discovery_engine.get_discovery_summary()

        # Analyze knowledge base
        discovery_types = defaultdict(int)
        for knowledge in self.knowledge_base.values():
            disc_type = knowledge['investigation']['discovery_type']
            discovery_types[disc_type] += 1

        return {
            'exploration_statistics': stats,
            'discovery_summary': discovery_summary,
            'knowledge_base_size': len(self.knowledge_base),
            'discovery_types': dict(discovery_types),
            'exploration_efficiency': stats.get('recent_avg_reward', 0),
            'budget_utilization': self.exploration_spent / self.exploration_budget
        }

# Example usage
async def exploration_example():
    # Create curiosity-driven agent
    agent = CuriosityDrivenAgent()

    # Simulate environment
    environment = {
        'state': 'unexplored_region',
        'available_tools': ['sensor_a', 'sensor_b'],
        'constraints': ['time_limit', 'energy_budget']
    }

    # Run exploration
    exploration_result = await agent.explore_environment(environment)

    print("Exploration Results:")
    print(f"Total explorations: {len(exploration_result['exploration_results'])}")
    print(f"Discoveries made: {exploration_result['discoveries']}")
    print(f"Knowledge gained: {exploration_result['knowledge_gained']} items")

    # Get insights
    insights = agent.get_exploration_insights()
    print("\nExploration Insights:")
    print(f"Exploration efficiency: {insights['exploration_efficiency']:.2f}")
    print(f"Budget utilization: {insights['budget_utilization']:.2f}")
    print(f"Discovery types: {insights['discovery_types']}")

    # Test pattern discovery
    patterns = agent.discovery_engine.patterns
    print(f"\nPatterns discovered: {len(patterns)}")

    for pattern_id, pattern in patterns.items():
        print(f"- {pattern['type']}: {pattern.get('pattern', 'N/A')} (confidence: {pattern.get('confidence', 0):.2f})")

        # Generate and test hypothesis
        hypothesis = agent.discovery_engine.generate_hypothesis(pattern_id)
        if hypothesis:
            print(f"  Hypothesis: {hypothesis['hypothesis']}")

# Run example
# asyncio.run(exploration_example())
```

### Adaptive Exploration Strategy

```python
class AdaptiveExplorationStrategy:
    def __init__(self):
        self.strategy_performance: Dict[str, List[float]] = {
            strategy.value: [] for strategy in ExplorationStrategy
        }
        self.current_strategy = ExplorationStrategy.EPSILON_GREEDY
        self.strategy_switch_threshold = 10  # Number of actions before evaluation
        self.actions_since_switch = 0
        self.adaptation_enabled = True

    def select_strategy(self, context: Dict[str, Any] = None) -> ExplorationStrategy:
        """Select the best exploration strategy based on context and performance"""
        if not self.adaptation_enabled:
            return self.current_strategy

        context = context or {}

        # Check if it's time to evaluate strategy performance
        if self.actions_since_switch >= self.strategy_switch_threshold:
            self._evaluate_and_adapt_strategy(context)
            self.actions_since_switch = 0

        return self.current_strategy

    def record_strategy_performance(self, strategy: ExplorationStrategy, reward: float):
        """Record performance for a specific strategy"""
        self.strategy_performance[strategy.value].append(reward)
        self.actions_since_switch += 1

        # Keep only recent performance data
        max_history = 100
        if len(self.strategy_performance[strategy.value]) > max_history:
            self.strategy_performance[strategy.value] = \
                self.strategy_performance[strategy.value][-max_history:]

    def _evaluate_and_adapt_strategy(self, context: Dict[str, Any]):
        """Evaluate strategy performance and adapt if necessary"""
        # Calculate average performance for each strategy
        strategy_scores = {}

        for strategy_name, rewards in self.strategy_performance.items():
            if rewards:
                # Recent performance weighted more heavily
                recent_rewards = rewards[-20:] if len(rewards) >= 20 else rewards
                avg_reward = sum(recent_rewards) / len(recent_rewards)

                # Context-based adjustments
                context_bonus = self._calculate_context_bonus(strategy_name, context)
                strategy_scores[strategy_name] = avg_reward + context_bonus
            else:
                strategy_scores[strategy_name] = 0.0

        # Find best performing strategy
        if strategy_scores:
            best_strategy_name = max(strategy_scores, key=strategy_scores.get)
            best_strategy = ExplorationStrategy(best_strategy_name)

            # Switch if the best strategy is significantly better
            current_score = strategy_scores.get(self.current_strategy.value, 0)
            best_score = strategy_scores[best_strategy_name]

            if best_score > current_score + 0.1:  # 0.1 is the switching threshold
                print(f"Switching exploration strategy from {self.current_strategy.value} "
                      f"to {best_strategy_name} (score improvement: {best_score - current_score:.3f})")
                self.current_strategy = best_strategy

    def _calculate_context_bonus(self, strategy_name: str, context: Dict[str, Any]) -> float:
        """Calculate context-based bonus for strategy selection"""
        bonus = 0.0

        # Environment characteristics
        uncertainty_level = context.get('uncertainty_level', 0.5)
        time_pressure = context.get('time_pressure', 0.5)
        resource_constraints = context.get('resource_constraints', 0.5)

        if strategy_name == ExplorationStrategy.EPSILON_GREEDY.value:
            # Good for balanced exploration/exploitation
            bonus += (1 - abs(uncertainty_level - 0.5)) * 0.1

        elif strategy_name == ExplorationStrategy.UCB1.value:
            # Good for high uncertainty environments
            bonus += uncertainty_level * 0.15

        elif strategy_name == ExplorationStrategy.THOMPSON_SAMPLING.value:
            # Good for complex decision spaces
            bonus += context.get('decision_complexity', 0.5) * 0.1

        elif strategy_name == ExplorationStrategy.CURIOSITY_DRIVEN.value:
            # Good when resources are abundant and time pressure is low
            bonus += (1 - time_pressure) * (1 - resource_constraints) * 0.2

        elif strategy_name == ExplorationStrategy.SYSTEMATIC.value:
            # Good for comprehensive coverage requirements
            bonus += context.get('coverage_requirement', 0.5) * 0.15

        return bonus

    def get_strategy_comparison(self) -> Dict[str, Any]:
        """Get comparison of strategy performances"""
        comparison = {}

        for strategy_name, rewards in self.strategy_performance.items():
            if rewards:
                comparison[strategy_name] = {
                    'sample_count': len(rewards),
                    'average_reward': sum(rewards) / len(rewards),
                    'recent_average': sum(rewards[-10:]) / min(len(rewards), 10),
                    'best_reward': max(rewards),
                    'worst_reward': min(rewards)
                }
            else:
                comparison[strategy_name] = {
                    'sample_count': 0,
                    'average_reward': 0.0,
                    'recent_average': 0.0,
                    'best_reward': 0.0,
                    'worst_reward': 0.0
                }

        return {
            'current_strategy': self.current_strategy.value,
            'strategy_performance': comparison,
            'adaptation_enabled': self.adaptation_enabled
        }

class MetaExplorationAgent:
    def __init__(self):
        self.strategy_selector = AdaptiveExplorationStrategy()
        self.exploration_managers: Dict[str, ExplorationManager] = {}
        self.meta_learning_history: List[Dict] = []

        # Initialize exploration managers for each strategy
        for strategy in ExplorationStrategy:
            self.exploration_managers[strategy.value] = ExplorationManager(strategy)

    async def meta_explore(self, action_space: List[ExplorationAction],
                          environment_context: Dict[str, Any],
                          exploration_episodes: int = 50) -> Dict[str, Any]:
        """Perform meta-exploration using adaptive strategy selection"""
        # Add actions to all exploration managers
        for strategy_name, manager in self.exploration_managers.items():
            for action in action_space:
                manager.add_action(action)

        episode_results = []

        for episode in range(exploration_episodes):
            # Select strategy for this episode
            strategy = self.strategy_selector.select_strategy(environment_context)
            manager = self.exploration_managers[strategy.value]

            # Perform exploration episode
            episode_result = await self._run_exploration_episode(
                manager, strategy, environment_context
            )

            # Record strategy performance
            self.strategy_selector.record_strategy_performance(
                strategy, episode_result['average_reward']
            )

            episode_results.append(episode_result)

            # Record meta-learning data
            self.meta_learning_history.append({
                'episode': episode,
                'strategy_used': strategy.value,
                'context': environment_context.copy(),
                'performance': episode_result['average_reward'],
                'timestamp': time.time()
            })

        return {
            'total_episodes': exploration_episodes,
            'episode_results': episode_results,
            'strategy_comparison': self.strategy_selector.get_strategy_comparison(),
            'best_strategy': self.strategy_selector.current_strategy.value
        }

    async def _run_exploration_episode(self, manager: ExplorationManager,
                                     strategy: ExplorationStrategy,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single exploration episode with a specific strategy"""
        episode_rewards = []
        actions_taken = []

        # Perform multiple actions in this episode
        for step in range(10):  # 10 actions per episode
            action = manager.select_action(context)
            actions_taken.append(action.id)

            # Simulate action execution and reward
            reward = await self._simulate_action_execution(action, context)
            episode_rewards.append(reward)

            # Record result
            manager.record_result(action.id, reward)

        return {
            'strategy': strategy.value,
            'actions_taken': actions_taken,
            'rewards': episode_rewards,
            'average_reward': sum(episode_rewards) / len(episode_rewards),
            'total_reward': sum(episode_rewards)
        }

    async def _simulate_action_execution(self, action: ExplorationAction,
                                       context: Dict[str, Any]) -> float:
        """Simulate the execution of an exploration action"""
        # Simple simulation based on action parameters and context
        base_reward = 0.5

        # Action-specific rewards
        if action.parameters.get('parameter') == 'location':
            location_rewards = {'north': 0.7, 'south': 0.6, 'east': 0.8, 'west': 0.5}
            base_reward = location_rewards.get(action.parameters.get('value'), 0.5)

        elif action.parameters.get('parameter') == 'tool':
            tool_rewards = {'sensor_a': 0.6, 'sensor_b': 0.75}
            base_reward = tool_rewards.get(action.parameters.get('value'), 0.5)

        # Context-based modifiers
        uncertainty = context.get('uncertainty_level', 0.5)
        noise_level = context.get('noise_level', 0.1)

        # Add uncertainty and noise
        reward = base_reward + random.uniform(-noise_level, noise_level)
        reward = max(0.0, min(1.0, reward))

        return reward

# Example usage
async def meta_exploration_example():
    # Create meta-exploration agent
    meta_agent = MetaExplorationAgent()

    # Define action space
    action_space = [
        ExplorationAction(f"action_{i}", {'parameter': 'location', 'value': loc})
        for i, loc in enumerate(['north', 'south', 'east', 'west'])
    ]
    action_space.extend([
        ExplorationAction(f"tool_{i}", {'parameter': 'tool', 'value': tool})
        for i, tool in enumerate(['sensor_a', 'sensor_b'])
    ])

    # Define environment contexts
    contexts = [
        {
            'uncertainty_level': 0.3,
            'time_pressure': 0.2,
            'resource_constraints': 0.1,
            'noise_level': 0.05
        },
        {
            'uncertainty_level': 0.8,
            'time_pressure': 0.7,
            'resource_constraints': 0.6,
            'noise_level': 0.15
        }
    ]

    for i, context in enumerate(contexts):
        print(f"\n=== Context {i+1}: Uncertainty={context['uncertainty_level']}, "
              f"Time Pressure={context['time_pressure']} ===")

        # Run meta-exploration
        result = await meta_agent.meta_explore(action_space, context, exploration_episodes=30)

        print(f"Best strategy found: {result['best_strategy']}")
        print("Strategy comparison:")

        strategy_comparison = result['strategy_comparison']['strategy_performance']
        for strategy_name, stats in strategy_comparison.items():
            if stats['sample_count'] > 0:
                print(f"  {strategy_name}: avg_reward={stats['average_reward']:.3f}, "
                      f"samples={stats['sample_count']}")

# Run example
# asyncio.run(meta_exploration_example())
```

## Best Practices

### Exploration Strategy Design
- **Multi-Strategy Approach**: Implement multiple exploration strategies and adapt based on context
- **Balance Exploration/Exploitation**: Use principled methods to balance exploration with exploitation
- **Context Sensitivity**: Adapt exploration behavior based on environmental characteristics
- **Resource Management**: Consider exploration costs and budget constraints

### Discovery Process Optimization
- **Systematic Pattern Detection**: Implement robust methods for identifying meaningful patterns
- **Hypothesis Validation**: Ensure discovered patterns are validated against new data
- **Knowledge Integration**: Properly integrate new discoveries into existing knowledge base
- **Incremental Learning**: Support continuous learning and knowledge refinement

### Performance Measurement
- **Multi-Metric Evaluation**: Use diverse metrics to assess exploration effectiveness
- **Long-Term Tracking**: Monitor exploration performance over extended periods
- **Comparative Analysis**: Compare different exploration approaches systematically
- **Adaptation Metrics**: Track how well the system adapts to new environments

### Safety and Robustness
- **Safe Exploration**: Ensure exploration doesn't lead to harmful or dangerous states
- **Bounded Exploration**: Implement limits on exploration scope and resource usage
- **Graceful Degradation**: Handle exploration failures and unexpected situations
- **Recovery Mechanisms**: Provide ways to recover from poor exploration decisions

## Common Pitfalls

### Premature Convergence
- **Problem**: Stopping exploration too early and missing better solutions
- **Solution**: Implement proper exploration schedules and diversity maintenance

### Inefficient Exploration
- **Problem**: Random or unfocused exploration wasting resources
- **Solution**: Use guided exploration strategies and information-theoretic approaches

### Pattern Overfitting
- **Problem**: Detecting spurious patterns in noisy or limited data
- **Solution**: Use statistical validation and require sufficient evidence for pattern confirmation

### Context Ignorance
- **Problem**: Not adapting exploration strategy to different environments or conditions
- **Solution**: Implement context-aware exploration and adaptive strategy selection

### Resource Exhaustion
- **Problem**: Spending too many resources on exploration without sufficient return
- **Solution**: Implement exploration budgets and efficiency monitoring

### Knowledge Fragmentation
- **Problem**: Discoveries not being properly integrated or organized
- **Solution**: Use structured knowledge representation and integration mechanisms

---

*This chapter covers 13 pages of content from "Agentic Design Patterns" by Antonio Gulli, focusing on Exploration and Discovery patterns for enabling AI agents to systematically investigate unknown environments and discover novel solutions.*