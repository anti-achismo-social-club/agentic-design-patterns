# Chapter 20: Prioritization

**Pattern Description:** Prioritization patterns enable AI agents to effectively rank, order, and schedule tasks, resources, and decisions based on multiple criteria including urgency, importance, dependencies, and available resources to optimize overall system performance.

## Introduction

Prioritization is a fundamental capability that distinguishes intelligent agents from simple reactive systems. In complex environments with multiple competing demands, limited resources, and varying constraints, the ability to prioritize effectively determines the success of an AI agent system. Prioritization patterns provide systematic approaches for agents to make informed decisions about what to do first, what to defer, and what to abandon entirely.

Modern AI agents often operate in dynamic environments where priorities can shift rapidly based on changing conditions, new information, or evolving goals. Effective prioritization requires not only ranking mechanisms but also adaptive algorithms that can respond to changing circumstances while maintaining overall system coherence and goal alignment.

The challenge of prioritization in AI systems extends beyond simple ordering to include resource allocation, deadline management, dependency resolution, and optimization across multiple objectives. These patterns provide frameworks for handling these complex prioritization scenarios systematically and efficiently.

## Key Concepts

### Prioritization Dimensions

#### Urgency and Importance
- **Eisenhower Matrix**: Categorizing tasks by urgent/important dimensions
- **Deadline-Driven Prioritization**: Scheduling based on time constraints and deadlines
- **Critical Path Analysis**: Identifying tasks that directly impact overall timeline
- **Impact Assessment**: Evaluating the consequences of task completion or delay

#### Resource Requirements
- **Resource Availability**: Prioritizing based on currently available resources
- **Resource Efficiency**: Favoring tasks with better resource utilization ratios
- **Resource Contention**: Managing competing demands for limited resources
- **Opportunity Cost**: Considering what must be foregone when choosing priorities

#### Dependencies and Relationships
- **Prerequisite Ordering**: Ensuring dependencies are satisfied before task execution
- **Parallel Execution**: Identifying tasks that can be performed simultaneously
- **Blocking Relationships**: Prioritizing tasks that unblock other important work
- **Cascade Effects**: Considering how task completion affects other tasks

#### Strategic Alignment
- **Goal Contribution**: Prioritizing tasks that best advance overall objectives
- **Strategic Value**: Weighing long-term versus short-term benefits
- **Risk Mitigation**: Prioritizing tasks that reduce system vulnerabilities
- **Innovation Potential**: Balancing maintenance tasks with exploratory work

### Prioritization Algorithms

#### Simple Ranking Methods
- **Weighted Scoring**: Assigning scores based on multiple weighted criteria
- **Pairwise Comparison**: Comparing tasks against each other systematically
- **Fixed Priority Schemes**: Using predetermined priority levels or classes
- **FIFO/LIFO**: First-in-first-out or last-in-first-out ordering

#### Dynamic Prioritization
- **Real-Time Scoring**: Continuously updating priorities based on changing conditions
- **Adaptive Weights**: Adjusting prioritization criteria based on performance feedback
- **Context-Aware Prioritization**: Modifying priorities based on current system state
- **Learning-Based Prioritization**: Using machine learning to improve priority decisions

#### Multi-Objective Optimization
- **Pareto Optimization**: Finding optimal trade-offs between competing objectives
- **Utility Maximization**: Optimizing for overall system utility or value
- **Constraint Satisfaction**: Prioritizing within hard and soft constraints
- **Game-Theoretic Approaches**: Handling conflicting priorities from multiple stakeholders

### Prioritization Architectures

#### Centralized Prioritization
- **Global Priority Queue**: Single system-wide priority ordering
- **Central Scheduler**: Centralized component responsible for all prioritization decisions
- **Hierarchy-Based**: Nested priority systems with clear authority levels
- **Policy-Driven**: Rules-based prioritization with centralized policy management

#### Distributed Prioritization
- **Local Priority Queues**: Each agent or component maintains its own priorities
- **Negotiation-Based**: Agents negotiate priorities through communication protocols
- **Market-Based**: Using economic principles to allocate priorities and resources
- **Emergent Prioritization**: Priorities emerge from local agent interactions

## Implementation

### Priority Queue System

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

        # Prefer tasks with moderate resource usage (not too high, not too low)
        if utilization_ratio < 0.1:
            return 0.7  # Too low utilization
        elif utilization_ratio > 0.8:
            return 0.9  # Too high utilization
        else:
            return 0.3  # Good utilization

    def _calculate_dependency_impact(self, task: Task, context: Dict[str, Any]) -> float:
        """Calculate impact on other tasks through dependencies"""
        dependent_tasks = context.get('dependent_tasks', {})
        blocked_tasks = dependent_tasks.get(task.id, [])

        # Higher impact if many tasks depend on this one
        if len(blocked_tasks) > 5:
            return 0.1  # High impact = high priority
        elif len(blocked_tasks) > 2:
            return 0.3
        elif len(blocked_tasks) > 0:
            return 0.5
        else:
            return 0.7  # No dependents = lower priority

    def _calculate_strategic_value(self, task: Task, context: Dict[str, Any]) -> float:
        """Calculate strategic value based on business goals"""
        strategic_goals = context.get('strategic_goals', [])

        # Check alignment with strategic goals
        goal_alignment = 0
        for goal in strategic_goals:
            if any(keyword in task.description.lower() for keyword in goal.get('keywords', [])):
                goal_alignment += goal.get('weight', 1.0)

        # Normalize to 0-1 range
        if goal_alignment > 2:
            return 0.2  # High strategic value
        elif goal_alignment > 1:
            return 0.4
        elif goal_alignment > 0:
            return 0.6
        else:
            return 0.8  # Low strategic value

class PriorityQueue:
    def __init__(self, priority_calculator: PriorityCalculator):
        self.heap: List[Task] = []
        self.task_map: Dict[str, Task] = {}
        self.priority_calculator = priority_calculator
        self.context: Dict[str, Any] = {}

    def add_task(self, task: Task):
        """Add a task to the priority queue"""
        # Calculate initial priority
        task.priority_score = self.priority_calculator.calculate_priority(task, self.context)

        # Add to heap and map
        heapq.heappush(self.heap, task)
        self.task_map[task.id] = task

    def get_next_task(self) -> Optional[Task]:
        """Get the highest priority task"""
        while self.heap:
            task = heapq.heappop(self.heap)
            if task.id in self.task_map:  # Task hasn't been removed
                del self.task_map[task.id]
                return task

        return None

    def peek_next_task(self) -> Optional[Task]:
        """Peek at the highest priority task without removing it"""
        while self.heap:
            if self.heap[0].id in self.task_map:
                return self.heap[0]
            else:
                heapq.heappop(self.heap)  # Remove stale reference

        return None

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue"""
        if task_id in self.task_map:
            del self.task_map[task_id]
            return True
        return False

    def update_task_priority(self, task_id: str):
        """Recalculate and update task priority"""
        if task_id in self.task_map:
            task = self.task_map[task_id]
            old_score = task.priority_score
            new_score = self.priority_calculator.calculate_priority(task, self.context)

            # If priority changed significantly, re-add to heap
            if abs(old_score - new_score) > 0.1:
                task.priority_score = new_score
                heapq.heappush(self.heap, task)

    def update_context(self, new_context: Dict[str, Any]):
        """Update the context for priority calculations"""
        self.context.update(new_context)

        # Recalculate all priorities
        for task_id in list(self.task_map.keys()):
            self.update_task_priority(task_id)

    def get_tasks_by_priority(self, limit: int = None) -> List[Task]:
        """Get tasks ordered by priority"""
        # Create a copy of the heap to avoid modifying the original
        heap_copy = self.heap.copy()
        heapq.heapify(heap_copy)

        tasks = []
        count = 0

        while heap_copy and (limit is None or count < limit):
            task = heapq.heappop(heap_copy)
            if task.id in self.task_map:
                tasks.append(task)
                count += 1

        return tasks

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the priority queue"""
        active_tasks = len(self.task_map)

        if active_tasks == 0:
            return {
                'total_tasks': 0,
                'avg_priority': 0,
                'priority_distribution': {}
            }

        priorities = [task.priority_score for task in self.task_map.values()]

        # Priority distribution
        distribution = {
            'critical': len([p for p in priorities if p <= 0.2]),
            'high': len([p for p in priorities if 0.2 < p <= 0.4]),
            'medium': len([p for p in priorities if 0.4 < p <= 0.6]),
            'low': len([p for p in priorities if 0.6 < p <= 0.8]),
            'deferred': len([p for p in priorities if p > 0.8])
        }

        return {
            'total_tasks': active_tasks,
            'avg_priority': sum(priorities) / len(priorities),
            'min_priority': min(priorities),
            'max_priority': max(priorities),
            'priority_distribution': distribution
        }

class TaskScheduler:
    def __init__(self, priority_calculator: PriorityCalculator):
        self.priority_queue = PriorityQueue(priority_calculator)
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.max_concurrent_tasks = 5
        self.resource_manager = ResourceManager()

    async def add_task(self, task: Task) -> str:
        """Add a task to the scheduler"""
        self.priority_queue.add_task(task)
        await self._schedule_tasks()
        return task.id

    async def _schedule_tasks(self):
        """Schedule tasks based on priority and resource availability"""
        while (len(self.running_tasks) < self.max_concurrent_tasks and
               self.priority_queue.peek_next_task() is not None):

            next_task = self.priority_queue.peek_next_task()

            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(next_task):
                break

            # Check if resources are available
            if not self.resource_manager.can_allocate(next_task.resource_requirements):
                break

            # Remove task from queue and start execution
            task = self.priority_queue.get_next_task()
            await self._start_task(task)

    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied"""
        for dep_id in task.dependencies:
            # Check if dependency is completed
            if not any(t.id == dep_id for t in self.completed_tasks):
                # Check if dependency is currently running
                if dep_id not in self.running_tasks:
                    return False
        return True

    async def _start_task(self, task: Task):
        """Start executing a task"""
        # Allocate resources
        self.resource_manager.allocate(task.resource_requirements)

        # Mark as running
        self.running_tasks[task.id] = task

        # Start task execution (simulated)
        asyncio.create_task(self._execute_task(task))

    async def _execute_task(self, task: Task):
        """Execute a task (simulated)"""
        try:
            # Simulate task execution time
            await asyncio.sleep(task.estimated_duration)

            # Mark as completed
            await self._complete_task(task, success=True)

        except Exception as e:
            # Handle task failure
            await self._complete_task(task, success=False, error=str(e))

    async def _complete_task(self, task: Task, success: bool, error: str = None):
        """Mark a task as completed"""
        # Remove from running tasks
        if task.id in self.running_tasks:
            del self.running_tasks[task.id]

        # Release resources
        self.resource_manager.release(task.resource_requirements)

        # Add to completed tasks
        task.metadata['completed_at'] = time.time()
        task.metadata['success'] = success
        if error:
            task.metadata['error'] = error

        self.completed_tasks.append(task)

        # Try to schedule more tasks
        await self._schedule_tasks()

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            'queued_tasks': len(self.priority_queue.task_map),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'queue_stats': self.priority_queue.get_queue_stats(),
            'resource_utilization': self.resource_manager.get_utilization()
        }

class ResourceManager:
    def __init__(self):
        self.total_resources = {
            'cpu': 100.0,
            'memory': 100.0,
            'network': 100.0
        }
        self.allocated_resources = {
            'cpu': 0.0,
            'memory': 0.0,
            'network': 0.0
        }

    def can_allocate(self, requirements: Dict[str, float]) -> bool:
        """Check if resources can be allocated"""
        for resource, amount in requirements.items():
            available = self.total_resources.get(resource, 0) - self.allocated_resources.get(resource, 0)
            if amount > available:
                return False
        return True

    def allocate(self, requirements: Dict[str, float]) -> bool:
        """Allocate resources"""
        if self.can_allocate(requirements):
            for resource, amount in requirements.items():
                self.allocated_resources[resource] = self.allocated_resources.get(resource, 0) + amount
            return True
        return False

    def release(self, requirements: Dict[str, float]):
        """Release allocated resources"""
        for resource, amount in requirements.items():
            self.allocated_resources[resource] = max(0, self.allocated_resources.get(resource, 0) - amount)

    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        utilization = {}
        for resource in self.total_resources:
            total = self.total_resources[resource]
            allocated = self.allocated_resources.get(resource, 0)
            utilization[resource] = (allocated / total) * 100 if total > 0 else 0
        return utilization
```

### Multi-Criteria Decision Making

```python
import numpy as np
from typing import Dict, List, Tuple

class MultiCriteriaDecisionMaker:
    def __init__(self):
        self.criteria_weights: Dict[str, float] = {}
        self.normalization_method = "min_max"

    def set_criteria_weights(self, weights: Dict[str, float]):
        """Set weights for decision criteria"""
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        self.criteria_weights = {
            criterion: weight / total_weight
            for criterion, weight in weights.items()
        }

    def evaluate_alternatives(self, alternatives: List[Dict[str, Any]],
                            criteria: List[str]) -> List[Tuple[Dict, float]]:
        """Evaluate alternatives using multi-criteria decision analysis"""
        if not alternatives or not criteria:
            return []

        # Extract criteria values for all alternatives
        criteria_matrix = []
        for alternative in alternatives:
            row = []
            for criterion in criteria:
                value = alternative.get(criterion, 0)
                row.append(float(value))
            criteria_matrix.append(row)

        criteria_matrix = np.array(criteria_matrix)

        # Normalize the criteria matrix
        normalized_matrix = self._normalize_matrix(criteria_matrix)

        # Apply weights
        weighted_matrix = self._apply_weights(normalized_matrix, criteria)

        # Calculate final scores
        scores = np.sum(weighted_matrix, axis=1)

        # Pair alternatives with their scores
        results = list(zip(alternatives, scores))

        # Sort by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize the criteria matrix"""
        if self.normalization_method == "min_max":
            return self._min_max_normalize(matrix)
        elif self.normalization_method == "z_score":
            return self._z_score_normalize(matrix)
        else:
            return matrix

    def _min_max_normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Min-max normalization"""
        normalized = np.zeros_like(matrix)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            min_val, max_val = np.min(col), np.max(col)
            if max_val > min_val:
                normalized[:, j] = (col - min_val) / (max_val - min_val)
            else:
                normalized[:, j] = 1.0  # All values are the same
        return normalized

    def _z_score_normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        normalized = np.zeros_like(matrix)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            mean_val, std_val = np.mean(col), np.std(col)
            if std_val > 0:
                normalized[:, j] = (col - mean_val) / std_val
            else:
                normalized[:, j] = 0.0  # All values are the same
        return normalized

    def _apply_weights(self, matrix: np.ndarray, criteria: List[str]) -> np.ndarray:
        """Apply criteria weights to the normalized matrix"""
        weighted_matrix = np.zeros_like(matrix)
        for j, criterion in enumerate(criteria):
            weight = self.criteria_weights.get(criterion, 1.0 / len(criteria))
            weighted_matrix[:, j] = matrix[:, j] * weight
        return weighted_matrix

class AdaptivePrioritizer:
    def __init__(self):
        self.decision_maker = MultiCriteriaDecisionMaker()
        self.performance_history: List[Dict] = []
        self.learning_rate = 0.1
        self.criteria_weights = {
            'urgency': 0.3,
            'importance': 0.25,
            'resource_efficiency': 0.2,
            'success_probability': 0.15,
            'strategic_value': 0.1
        }
        self.decision_maker.set_criteria_weights(self.criteria_weights)

    async def prioritize_tasks(self, tasks: List[Task],
                             context: Dict[str, Any] = None) -> List[Task]:
        """Prioritize tasks using adaptive multi-criteria decision making"""
        if not tasks:
            return []

        context = context or {}

        # Convert tasks to alternatives for MCDM
        alternatives = []
        for task in tasks:
            alternative = {
                'task': task,
                'urgency': self._calculate_urgency_score(task),
                'importance': self._calculate_importance_score(task, context),
                'resource_efficiency': self._calculate_efficiency_score(task, context),
                'success_probability': self._estimate_success_probability(task, context),
                'strategic_value': self._calculate_strategic_score(task, context)
            }
            alternatives.append(alternative)

        # Evaluate alternatives
        criteria = ['urgency', 'importance', 'resource_efficiency',
                   'success_probability', 'strategic_value']

        ranked_alternatives = self.decision_maker.evaluate_alternatives(
            alternatives, criteria
        )

        # Extract prioritized tasks
        prioritized_tasks = [alt['task'] for alt, score in ranked_alternatives]

        return prioritized_tasks

    def _calculate_urgency_score(self, task: Task) -> float:
        """Calculate urgency score (0-1, higher is more urgent)"""
        if not task.deadline:
            return 0.5  # No deadline = medium urgency

        current_time = time.time()
        time_until_deadline = task.deadline - current_time

        if time_until_deadline <= 0:
            return 1.0  # Overdue
        elif time_until_deadline < 3600:  # < 1 hour
            return 0.9
        elif time_until_deadline < 86400:  # < 1 day
            return 0.7
        elif time_until_deadline < 604800:  # < 1 week
            return 0.5
        else:
            return 0.3

    def _calculate_importance_score(self, task: Task, context: Dict) -> float:
        """Calculate importance score (0-1, higher is more important)"""
        score = 0.5  # Base importance

        # Check tags
        if 'critical' in task.tags:
            score = 1.0
        elif 'high' in task.tags:
            score = 0.8
        elif 'medium' in task.tags:
            score = 0.5
        elif 'low' in task.tags:
            score = 0.2

        # Check business impact
        business_goals = context.get('business_goals', [])
        for goal in business_goals:
            if any(keyword in task.description.lower()
                  for keyword in goal.get('keywords', [])):
                score = min(1.0, score + goal.get('impact_multiplier', 0.1))

        return score

    def _calculate_efficiency_score(self, task: Task, context: Dict) -> float:
        """Calculate resource efficiency score (0-1, higher is more efficient)"""
        if not task.resource_requirements:
            return 0.8  # No specific requirements = efficient

        available_resources = context.get('available_resources', {})
        if not available_resources:
            return 0.5

        # Calculate resource utilization ratio
        total_required = sum(task.resource_requirements.values())
        total_available = sum(available_resources.values())

        if total_available == 0:
            return 0.0

        utilization_ratio = total_required / total_available

        # Optimal utilization is around 50-70%
        if 0.5 <= utilization_ratio <= 0.7:
            return 1.0
        elif utilization_ratio < 0.5:
            return 0.7 + (utilization_ratio / 0.5) * 0.3
        else:
            return max(0.0, 1.0 - (utilization_ratio - 0.7) / 0.3)

    def _estimate_success_probability(self, task: Task, context: Dict) -> float:
        """Estimate task success probability (0-1)"""
        # Base probability
        base_prob = 0.8

        # Adjust based on task complexity (estimated by duration)
        if task.estimated_duration > 3600:  # > 1 hour
            base_prob *= 0.9
        if task.estimated_duration > 86400:  # > 1 day
            base_prob *= 0.8

        # Adjust based on dependencies
        dependency_factor = max(0.5, 1.0 - (len(task.dependencies) * 0.1))
        base_prob *= dependency_factor

        # Adjust based on historical performance
        similar_tasks = self._find_similar_tasks(task)
        if similar_tasks:
            historical_success_rate = sum(
                t.get('success', 0) for t in similar_tasks
            ) / len(similar_tasks)
            base_prob = (base_prob + historical_success_rate) / 2

        return min(1.0, base_prob)

    def _calculate_strategic_score(self, task: Task, context: Dict) -> float:
        """Calculate strategic value score (0-1)"""
        strategic_goals = context.get('strategic_goals', [])
        if not strategic_goals:
            return 0.5

        score = 0.0
        for goal in strategic_goals:
            goal_weight = goal.get('weight', 1.0)
            keywords = goal.get('keywords', [])

            # Check if task aligns with strategic goal
            if any(keyword in task.description.lower() for keyword in keywords):
                score += goal_weight

        # Normalize score
        max_possible_score = sum(goal.get('weight', 1.0) for goal in strategic_goals)
        if max_possible_score > 0:
            score = min(1.0, score / max_possible_score)

        return score

    def _find_similar_tasks(self, task: Task) -> List[Dict]:
        """Find similar tasks in performance history"""
        similar_tasks = []

        for record in self.performance_history:
            # Simple similarity based on tags and description keywords
            similarity = 0.0

            # Tag similarity
            common_tags = set(task.tags) & set(record.get('tags', []))
            tag_similarity = len(common_tags) / max(len(task.tags), 1)
            similarity += tag_similarity * 0.5

            # Description similarity (simplified)
            task_words = set(task.description.lower().split())
            record_words = set(record.get('description', '').lower().split())
            word_similarity = len(task_words & record_words) / max(len(task_words), 1)
            similarity += word_similarity * 0.5

            if similarity > 0.3:  # Threshold for similarity
                similar_tasks.append(record)

        return similar_tasks

    async def update_performance(self, task: Task, success: bool,
                               actual_duration: float, context: Dict = None):
        """Update performance history with task results"""
        performance_record = {
            'task_id': task.id,
            'tags': task.tags,
            'description': task.description,
            'estimated_duration': task.estimated_duration,
            'actual_duration': actual_duration,
            'success': success,
            'timestamp': time.time(),
            'context': context or {}
        }

        self.performance_history.append(performance_record)

        # Adapt criteria weights based on performance
        await self._adapt_criteria_weights(performance_record)

    async def _adapt_criteria_weights(self, performance_record: Dict):
        """Adapt criteria weights based on performance feedback"""
        # Simple adaptation: if tasks with high urgency scores fail more often,
        # reduce urgency weight and increase other weights

        if len(self.performance_history) < 10:
            return  # Need more data

        # Analyze recent performance
        recent_records = self.performance_history[-20:]  # Last 20 tasks

        # Calculate success rates by dominant criteria
        criteria_performance = {
            'urgency': {'success': 0, 'total': 0},
            'importance': {'success': 0, 'total': 0},
            'resource_efficiency': {'success': 0, 'total': 0},
            'success_probability': {'success': 0, 'total': 0},
            'strategic_value': {'success': 0, 'total': 0}
        }

        for record in recent_records:
            # Determine which criterion was dominant (simplified heuristic)
            # In practice, you'd track which criterion contributed most to selection
            dominant_criterion = 'importance'  # Placeholder

            criteria_performance[dominant_criterion]['total'] += 1
            if record['success']:
                criteria_performance[dominant_criterion]['success'] += 1

        # Adjust weights based on performance
        for criterion, stats in criteria_performance.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total']
                current_weight = self.criteria_weights[criterion]

                # Increase weight if success rate is high, decrease if low
                adjustment = (success_rate - 0.8) * self.learning_rate
                new_weight = max(0.05, min(0.5, current_weight + adjustment))
                self.criteria_weights[criterion] = new_weight

        # Renormalize weights
        total_weight = sum(self.criteria_weights.values())
        self.criteria_weights = {
            k: v / total_weight for k, v in self.criteria_weights.items()
        }

        # Update decision maker
        self.decision_maker.set_criteria_weights(self.criteria_weights)
```

## Code Examples

### Dynamic Priority Adjustment

```python
class DynamicPriorityManager:
    def __init__(self):
        self.priority_adjusters: List[Callable] = []
        self.context_monitors: List[Callable] = []
        self.adjustment_history: List[Dict] = []

    def add_priority_adjuster(self, adjuster: Callable):
        """Add a function that can adjust priorities based on context"""
        self.priority_adjusters.append(adjuster)

    def add_context_monitor(self, monitor: Callable):
        """Add a function that monitors context changes"""
        self.context_monitors.append(monitor)

    async def adjust_priorities(self, priority_queue: PriorityQueue,
                              context_change: Dict[str, Any]):
        """Dynamically adjust priorities based on context changes"""
        adjustment_log = {
            'timestamp': time.time(),
            'context_change': context_change,
            'adjustments': []
        }

        # Run context monitors
        for monitor in self.context_monitors:
            try:
                monitor_result = await monitor(context_change)
                if monitor_result.get('requires_adjustment'):
                    adjustment_log['adjustments'].append(monitor_result)
            except Exception as e:
                print(f"Error in context monitor: {e}")

        # Apply priority adjusters
        tasks_to_adjust = []
        for adjuster in self.priority_adjusters:
            try:
                adjuster_result = await adjuster(
                    priority_queue.task_map.values(), context_change
                )
                tasks_to_adjust.extend(adjuster_result.get('tasks_to_adjust', []))
            except Exception as e:
                print(f"Error in priority adjuster: {e}")

        # Update task priorities
        for task_id in set(tasks_to_adjust):
            if task_id in priority_queue.task_map:
                old_priority = priority_queue.task_map[task_id].priority_score
                priority_queue.update_task_priority(task_id)
                new_priority = priority_queue.task_map[task_id].priority_score

                if abs(old_priority - new_priority) > 0.05:
                    adjustment_log['adjustments'].append({
                        'task_id': task_id,
                        'old_priority': old_priority,
                        'new_priority': new_priority,
                        'change': new_priority - old_priority
                    })

        self.adjustment_history.append(adjustment_log)
        return adjustment_log

# Example priority adjuster functions
async def deadline_pressure_adjuster(tasks: List[Task],
                                   context_change: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust priorities based on approaching deadlines"""
    current_time = time.time()
    tasks_to_adjust = []

    for task in tasks:
        if task.deadline:
            time_remaining = task.deadline - current_time
            if time_remaining < 3600 and time_remaining > 0:  # Less than 1 hour
                tasks_to_adjust.append(task.id)

    return {
        'adjuster': 'deadline_pressure',
        'tasks_to_adjust': tasks_to_adjust,
        'reason': 'Approaching deadlines detected'
    }

async def resource_availability_adjuster(tasks: List[Task],
                                       context_change: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust priorities based on resource availability changes"""
    resource_changes = context_change.get('resource_availability', {})
    tasks_to_adjust = []

    for task in tasks:
        # If resources for this task became more available, increase priority
        for resource, availability in resource_changes.items():
            if (resource in task.resource_requirements and
                availability > 0.8):  # High availability
                tasks_to_adjust.append(task.id)
                break

    return {
        'adjuster': 'resource_availability',
        'tasks_to_adjust': tasks_to_adjust,
        'reason': f'Resource availability changed: {resource_changes}'
    }

async def emergency_event_adjuster(tasks: List[Task],
                                 context_change: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust priorities based on emergency events"""
    emergency_tags = context_change.get('emergency_tags', [])
    tasks_to_adjust = []

    if emergency_tags:
        for task in tasks:
            # Boost priority of tasks related to emergency
            if any(tag in task.tags for tag in emergency_tags):
                tasks_to_adjust.append(task.id)

    return {
        'adjuster': 'emergency_event',
        'tasks_to_adjust': tasks_to_adjust,
        'reason': f'Emergency event detected: {emergency_tags}'
    }

# Example usage
async def prioritization_example():
    # Create priority calculator
    calculator = PriorityCalculator()

    # Create task scheduler
    scheduler = TaskScheduler(calculator)

    # Create adaptive prioritizer
    adaptive_prioritizer = AdaptivePrioritizer()

    # Create dynamic priority manager
    dynamic_manager = DynamicPriorityManager()
    dynamic_manager.add_priority_adjuster(deadline_pressure_adjuster)
    dynamic_manager.add_priority_adjuster(resource_availability_adjuster)
    dynamic_manager.add_priority_adjuster(emergency_event_adjuster)

    # Create sample tasks
    tasks = [
        Task(
            id="task_1",
            name="Data Processing",
            description="Process customer data for analysis",
            priority_score=0.0,  # Will be calculated
            created_at=time.time(),
            deadline=time.time() + 7200,  # 2 hours from now
            estimated_duration=1800,  # 30 minutes
            resource_requirements={'cpu': 20, 'memory': 30},
            tags=['high', 'data', 'customer']
        ),
        Task(
            id="task_2",
            name="Report Generation",
            description="Generate monthly financial report",
            priority_score=0.0,
            created_at=time.time() - 3600,  # 1 hour ago
            deadline=time.time() + 86400,  # 1 day from now
            estimated_duration=3600,  # 1 hour
            resource_requirements={'cpu': 10, 'memory': 20},
            tags=['medium', 'report', 'finance']
        ),
        Task(
            id="task_3",
            name="System Backup",
            description="Perform daily system backup",
            priority_score=0.0,
            created_at=time.time() - 1800,  # 30 minutes ago
            estimated_duration=2400,  # 40 minutes
            resource_requirements={'cpu': 5, 'disk': 50},
            tags=['low', 'maintenance', 'backup']
        )
    ]

    # Set up context
    context = {
        'available_resources': {'cpu': 100, 'memory': 100, 'disk': 100},
        'strategic_goals': [
            {
                'keywords': ['customer', 'data'],
                'weight': 2.0,
                'impact_multiplier': 0.2
            }
        ],
        'important_tags': ['high', 'critical']
    }

    # Add tasks to scheduler
    for task in tasks:
        await scheduler.add_task(task)

    # Update context
    scheduler.priority_queue.update_context(context)

    print("Initial scheduler status:")
    print(scheduler.get_status())

    # Use adaptive prioritizer
    prioritized_tasks = await adaptive_prioritizer.prioritize_tasks(tasks, context)
    print("\nAdaptive prioritization order:")
    for i, task in enumerate(prioritized_tasks):
        print(f"{i+1}. {task.name} (Priority: {task.priority_score:.3f})")

    # Simulate emergency event
    emergency_context = {
        'emergency_tags': ['customer'],
        'resource_availability': {'cpu': 0.9}
    }

    adjustment_log = await dynamic_manager.adjust_priorities(
        scheduler.priority_queue, emergency_context
    )

    print(f"\nDynamic adjustment log:")
    print(f"Adjustments made: {len(adjustment_log['adjustments'])}")

    # Show updated status
    print("\nUpdated scheduler status:")
    print(scheduler.get_status())

# Run example
# asyncio.run(prioritization_example())
```

## Best Practices

### Priority Calculation Design
- **Multi-Factor Scoring**: Use multiple criteria rather than single-factor prioritization
- **Normalization**: Ensure different criteria are properly normalized and weighted
- **Dynamic Adaptation**: Implement mechanisms to adjust priorities based on changing conditions
- **Context Awareness**: Consider environmental factors and system state in priority calculations

### Resource-Aware Prioritization
- **Resource Constraints**: Factor in resource availability when making priority decisions
- **Utilization Optimization**: Balance high-priority tasks with efficient resource usage
- **Dependency Management**: Ensure dependent tasks are scheduled appropriately
- **Deadlock Prevention**: Avoid priority inversions and resource deadlocks

### Performance Optimization
- **Efficient Data Structures**: Use appropriate data structures (heaps, balanced trees) for priority queues
- **Incremental Updates**: Update priorities incrementally rather than recalculating everything
- **Caching**: Cache priority calculations when inputs haven't changed
- **Parallel Processing**: Enable parallel execution of independent high-priority tasks

### Adaptability and Learning
- **Performance Feedback**: Use task completion results to improve future prioritization
- **Weight Adaptation**: Adjust criterion weights based on historical performance
- **Context Learning**: Learn from context patterns to improve priority predictions
- **Continuous Improvement**: Regularly review and refine prioritization algorithms

## Common Pitfalls

### Priority Inversion
- **Problem**: Low-priority tasks blocking high-priority tasks due to resource dependencies
- **Solution**: Implement priority inheritance and careful dependency management

### Starvation
- **Problem**: Low-priority tasks never getting executed due to constant high-priority work
- **Solution**: Implement aging mechanisms and fairness guarantees

### Thrashing
- **Problem**: Constantly changing priorities causing inefficient context switching
- **Solution**: Use hysteresis and minimum duration thresholds for priority changes

### Over-Optimization
- **Problem**: Spending more time on prioritization than actual task execution
- **Solution**: Balance prioritization complexity with execution efficiency

### Context Ignorance
- **Problem**: Priorities not adapting to changing environmental conditions
- **Solution**: Implement comprehensive context monitoring and adaptive mechanisms

### Short-Term Focus
- **Problem**: Always prioritizing urgent tasks over important long-term work
- **Solution**: Balance immediate needs with strategic objectives through proper weighting

---

*This chapter covers 10 pages of content from "Agentic Design Patterns" by Antonio Gulli, focusing on Prioritization patterns for effective task ordering and resource allocation in AI agent systems.*