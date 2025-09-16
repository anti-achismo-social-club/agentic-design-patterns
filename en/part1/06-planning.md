# Chapter 6: Planning

*Original content: 13 pages - by Antonio Gulli*

## Brief Description

Planning is an agentic design pattern where AI systems break down complex tasks into structured sequences of actions, creating detailed roadmaps for achieving specific goals. This pattern enables systematic problem-solving by organizing tasks hierarchically, managing dependencies, and adapting plans based on execution outcomes and changing conditions.

## Introduction

Planning represents one of the most sophisticated cognitive capabilities in agentic AI systems, mirroring human strategic thinking and project management skills. Unlike reactive systems that respond to immediate inputs, planning agents proactively structure their approach to complex problems, considering multiple steps, dependencies, and potential contingencies.

The planning pattern is essential for handling multi-step tasks that require coordination, resource allocation, and temporal reasoning. It enables AI systems to work toward long-term objectives while managing intermediate goals, handling uncertainties, and adapting to changing circumstances.

Modern AI planning goes beyond simple sequential task lists. It encompasses hierarchical decomposition, parallel execution paths, contingency planning, resource optimization, and dynamic replanning based on real-world feedback. This makes it particularly valuable for complex workflows, project management, strategic decision-making, and autonomous system control.

## Key Concepts

### Task Decomposition
- Breaking complex goals into manageable sub-tasks
- Hierarchical planning with multiple levels of abstraction
- Identifying atomic actions and compound activities
- Balancing granularity with practical execution needs

### Dependency Management
- Identifying prerequisites and sequencing constraints
- Managing resource dependencies and availability
- Handling temporal dependencies and scheduling
- Optimizing critical path execution

### Plan Representation
- Structured formats for storing and communicating plans
- Graph-based representations for complex dependencies
- Timeline-based planning for temporal constraints
- State-space planning for condition-dependent actions

### Dynamic Planning
- Adapting plans based on execution feedback
- Replanning when assumptions change or constraints evolve
- Handling unexpected obstacles and opportunities
- Maintaining plan coherence while enabling flexibility

## Implementation

### Basic Planning Framework
```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import uuid
from datetime import datetime, timedelta

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    id: str
    name: str
    description: str
    dependencies: List[str]
    estimated_duration: timedelta
    required_resources: List[str]
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Plan:
    id: str
    goal: str
    tasks: Dict[str, Task]
    created_at: datetime
    updated_at: datetime
    status: str = "active"

class BasicPlanner:
    """Basic planning system for task decomposition and sequencing"""

    def __init__(self):
        self.plans: Dict[str, Plan] = {}

    def create_plan(self, goal: str, requirements: Dict[str, Any]) -> Plan:
        """Create a new plan for achieving a goal"""
        plan_id = str(uuid.uuid4())

        # Decompose goal into tasks
        tasks = self._decompose_goal(goal, requirements)

        # Create plan object
        plan = Plan(
            id=plan_id,
            goal=goal,
            tasks={task.id: task for task in tasks},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        self.plans[plan_id] = plan
        return plan

    def _decompose_goal(self, goal: str, requirements: Dict[str, Any]) -> List[Task]:
        """Decompose a goal into executable tasks"""
        # This would use AI/ML to intelligently break down the goal
        # For demonstration, using a simplified approach

        decomposition_prompt = f"""
        Goal: {goal}
        Requirements: {requirements}

        Break this goal into specific, actionable tasks with dependencies.
        Each task should be concrete and measurable.
        """

        # Simulate AI-generated task decomposition
        task_descriptions = self._generate_task_decomposition(decomposition_prompt)

        tasks = []
        for i, task_desc in enumerate(task_descriptions):
            task = Task(
                id=str(uuid.uuid4()),
                name=task_desc['name'],
                description=task_desc['description'],
                dependencies=task_desc.get('dependencies', []),
                estimated_duration=timedelta(hours=task_desc.get('hours', 1)),
                required_resources=task_desc.get('resources', []),
                priority=task_desc.get('priority', 1)
            )
            tasks.append(task)

        return tasks

    def get_executable_tasks(self, plan_id: str) -> List[Task]:
        """Get tasks that can be executed now (dependencies met)"""
        plan = self.plans[plan_id]
        executable = []

        for task in plan.tasks.values():
            if task.status == TaskStatus.PENDING:
                if self._dependencies_satisfied(task, plan):
                    executable.append(task)

        return sorted(executable, key=lambda t: t.priority, reverse=True)

    def _dependencies_satisfied(self, task: Task, plan: Plan) -> bool:
        """Check if all task dependencies are completed"""
        for dep_id in task.dependencies:
            if dep_id in plan.tasks:
                dep_task = plan.tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
        return True

    def update_task_status(self, plan_id: str, task_id: str, status: TaskStatus):
        """Update task status and trigger replanning if needed"""
        plan = self.plans[plan_id]
        if task_id in plan.tasks:
            plan.tasks[task_id].status = status
            plan.updated_at = datetime.now()

            # Trigger replanning for certain status changes
            if status == TaskStatus.FAILED:
                self._handle_task_failure(plan, task_id)
```

### Advanced Planning System
```python
class AdvancedPlanner:
    """Advanced planning system with hierarchical planning and adaptation"""

    def __init__(self):
        self.strategy_generator = PlanningStrategyGenerator()
        self.resource_manager = ResourceManager()
        self.risk_assessor = RiskAssessor()
        self.plan_optimizer = PlanOptimizer()

    async def create_comprehensive_plan(self, objective: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive plan with multiple strategies and contingencies"""

        # Generate multiple planning strategies
        strategies = await self.strategy_generator.generate_strategies(objective, context)

        # Evaluate and select best strategy
        selected_strategy = await self._evaluate_strategies(strategies, context)

        # Create detailed plan based on selected strategy
        detailed_plan = await self._create_detailed_plan(selected_strategy, context)

        # Generate contingency plans
        contingencies = await self._generate_contingency_plans(detailed_plan, context)

        # Optimize plan for resources and timeline
        optimized_plan = await self.plan_optimizer.optimize(detailed_plan, context)

        return {
            'primary_plan': optimized_plan,
            'contingency_plans': contingencies,
            'strategy': selected_strategy,
            'risk_assessment': await self.risk_assessor.assess_plan(optimized_plan),
            'resource_requirements': await self.resource_manager.analyze_requirements(optimized_plan)
        }

    async def _evaluate_strategies(self, strategies: List[Dict], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate planning strategies and select the best one"""
        evaluations = []

        for strategy in strategies:
            evaluation = await self._evaluate_single_strategy(strategy, context)
            evaluations.append({
                'strategy': strategy,
                'evaluation': evaluation
            })

        # Select strategy with best overall score
        best_strategy = max(evaluations, key=lambda x: x['evaluation']['overall_score'])
        return best_strategy['strategy']

    async def _create_detailed_plan(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed execution plan from high-level strategy"""

        plan = {
            'id': str(uuid.uuid4()),
            'strategy': strategy,
            'phases': [],
            'tasks': {},
            'timeline': {},
            'resources': {},
            'milestones': []
        }

        # Generate phases from strategy
        for phase_desc in strategy['phases']:
            phase = await self._generate_phase_plan(phase_desc, context)
            plan['phases'].append(phase)

            # Add phase tasks to overall task collection
            for task in phase['tasks']:
                plan['tasks'][task['id']] = task

        # Generate timeline and resource allocation
        plan['timeline'] = await self._generate_timeline(plan['tasks'])
        plan['resources'] = await self.resource_manager.allocate_resources(plan['tasks'])

        return plan

    async def replan_dynamically(self, plan_id: str, execution_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically replan based on execution feedback"""

        current_plan = self._get_plan(plan_id)

        # Analyze feedback and identify needed changes
        change_analysis = await self._analyze_execution_feedback(execution_feedback, current_plan)

        if change_analysis['requires_replanning']:
            # Generate updated plan
            updated_plan = await self._update_plan(current_plan, change_analysis)

            # Validate updated plan
            validation_result = await self._validate_plan(updated_plan)

            if validation_result['valid']:
                return {
                    'updated_plan': updated_plan,
                    'changes': change_analysis['changes'],
                    'rationale': change_analysis['rationale']
                }
            else:
                # Fallback to contingency plan
                return await self._activate_contingency_plan(plan_id, change_analysis)

        return {'status': 'no_changes_needed', 'current_plan': current_plan}
```

## Code Examples

### Example 1: Project Management Planner
```python
class ProjectManagementPlanner:
    """Specialized planner for project management tasks"""

    def __init__(self):
        self.base_planner = AdvancedPlanner()
        self.project_templates = ProjectTemplateLibrary()

    async def plan_software_project(self, project_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive plan for a software development project"""

        # Extract project requirements
        requirements = self._extract_requirements(project_spec)

        # Select appropriate project template
        template = await self.project_templates.find_best_match(requirements)

        # Customize template for specific project
        customized_phases = await self._customize_project_phases(template, requirements)

        # Create detailed plan
        project_plan = await self.base_planner.create_comprehensive_plan(
            objective=f"Develop {project_spec['name']} software project",
            context={
                'requirements': requirements,
                'template': template,
                'phases': customized_phases,
                'team_size': project_spec.get('team_size', 5),
                'timeline': project_spec.get('timeline', '6 months'),
                'budget': project_spec.get('budget')
            }
        )

        # Add project-specific elements
        project_plan['development_phases'] = await self._plan_development_phases(customized_phases)
        project_plan['quality_gates'] = await self._define_quality_gates(requirements)
        project_plan['risk_mitigation'] = await self._plan_risk_mitigation(project_plan)

        return project_plan

    async def _plan_development_phases(self, phases: List[Dict]) -> List[Dict]:
        """Plan detailed development phases"""
        detailed_phases = []

        for phase in phases:
            if phase['type'] == 'development':
                # Break down development into sprints
                sprints = await self._plan_development_sprints(phase)
                detailed_phases.extend(sprints)
            elif phase['type'] == 'testing':
                # Plan testing activities
                testing_plan = await self._plan_testing_phase(phase)
                detailed_phases.append(testing_plan)
            elif phase['type'] == 'deployment':
                # Plan deployment activities
                deployment_plan = await self._plan_deployment_phase(phase)
                detailed_phases.append(deployment_plan)

        return detailed_phases

    async def _plan_development_sprints(self, development_phase: Dict) -> List[Dict]:
        """Plan individual development sprints"""
        features = development_phase['features']
        team_velocity = development_phase.get('team_velocity', 20)  # story points per sprint

        sprints = []
        current_features = []
        current_points = 0

        for feature in features:
            feature_points = feature.get('story_points', 8)

            if current_points + feature_points > team_velocity and current_features:
                # Create sprint with current features
                sprint = await self._create_sprint_plan(current_features, len(sprints) + 1)
                sprints.append(sprint)

                # Start new sprint
                current_features = [feature]
                current_points = feature_points
            else:
                current_features.append(feature)
                current_points += feature_points

        # Add final sprint if there are remaining features
        if current_features:
            sprint = await self._create_sprint_plan(current_features, len(sprints) + 1)
            sprints.append(sprint)

        return sprints
```

### Example 2: Research Planning Agent
```python
class ResearchPlanningAgent:
    """Agent specialized in planning research activities"""

    def __init__(self):
        self.knowledge_mapper = KnowledgeMapper()
        self.methodology_selector = MethodologySelector()
        self.resource_estimator = ResourceEstimator()

    async def plan_research_project(self, research_question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a comprehensive research project"""

        # Map existing knowledge and identify gaps
        knowledge_map = await self.knowledge_mapper.map_domain(research_question)

        # Select appropriate research methodologies
        methodologies = await self.methodology_selector.select_methods(
            research_question, knowledge_map, context
        )

        # Create research phases
        research_phases = await self._plan_research_phases(research_question, methodologies, knowledge_map)

        # Estimate resources and timeline
        resource_estimate = await self.resource_estimator.estimate_research_resources(research_phases)

        # Create comprehensive research plan
        research_plan = {
            'research_question': research_question,
            'knowledge_map': knowledge_map,
            'methodologies': methodologies,
            'phases': research_phases,
            'resource_requirements': resource_estimate,
            'timeline': self._create_research_timeline(research_phases),
            'deliverables': self._identify_deliverables(research_phases),
            'validation_criteria': self._define_validation_criteria(research_question)
        }

        return research_plan

    async def _plan_research_phases(self, question: str, methodologies: List[Dict], knowledge_map: Dict) -> List[Dict]:
        """Plan detailed research phases"""
        phases = []

        # Literature review phase
        literature_phase = await self._plan_literature_review(question, knowledge_map)
        phases.append(literature_phase)

        # Methodology-specific phases
        for methodology in methodologies:
            if methodology['type'] == 'experimental':
                experimental_phase = await self._plan_experimental_research(methodology, question)
                phases.append(experimental_phase)
            elif methodology['type'] == 'analytical':
                analytical_phase = await self._plan_analytical_research(methodology, question)
                phases.append(analytical_phase)
            elif methodology['type'] == 'empirical':
                empirical_phase = await self._plan_empirical_research(methodology, question)
                phases.append(empirical_phase)

        # Analysis and synthesis phase
        analysis_phase = await self._plan_analysis_phase(methodologies, question)
        phases.append(analysis_phase)

        # Documentation and publication phase
        documentation_phase = await self._plan_documentation_phase(question, phases)
        phases.append(documentation_phase)

        return phases

    async def _plan_literature_review(self, question: str, knowledge_map: Dict) -> Dict[str, Any]:
        """Plan comprehensive literature review"""
        return {
            'name': 'Literature Review',
            'type': 'literature_review',
            'objectives': [
                'Map existing knowledge in the domain',
                'Identify research gaps',
                'Establish theoretical foundation'
            ],
            'activities': [
                {
                    'name': 'Database Search',
                    'description': 'Search academic databases for relevant literature',
                    'databases': ['PubMed', 'IEEE Xplore', 'Google Scholar', 'ArXiv'],
                    'search_terms': self._generate_search_terms(question),
                    'estimated_duration': timedelta(weeks=2)
                },
                {
                    'name': 'Paper Analysis',
                    'description': 'Analyze and categorize selected papers',
                    'methods': ['systematic review', 'thematic analysis'],
                    'estimated_duration': timedelta(weeks=3)
                },
                {
                    'name': 'Knowledge Synthesis',
                    'description': 'Synthesize findings and identify gaps',
                    'deliverables': ['literature review document', 'knowledge gap analysis'],
                    'estimated_duration': timedelta(weeks=1)
                }
            ],
            'success_criteria': [
                'Comprehensive coverage of relevant literature',
                'Clear identification of research gaps',
                'Strong theoretical foundation established'
            ]
        }
```

### Example 3: Business Strategy Planner
```python
class BusinessStrategyPlanner:
    """Planner for business strategy development and execution"""

    def __init__(self):
        self.market_analyzer = MarketAnalyzer()
        self.competitive_analyzer = CompetitiveAnalyzer()
        self.scenario_planner = ScenarioPlanner()

    async def develop_business_strategy(self, business_context: Dict[str, Any]) -> Dict[str, Any]:
        """Develop comprehensive business strategy with execution plan"""

        # Analyze market and competitive landscape
        market_analysis = await self.market_analyzer.analyze_market(business_context)
        competitive_analysis = await self.competitive_analyzer.analyze_competitors(business_context)

        # Generate strategic options
        strategic_options = await self._generate_strategic_options(
            business_context, market_analysis, competitive_analysis
        )

        # Evaluate options using scenario planning
        option_evaluations = await self.scenario_planner.evaluate_options(
            strategic_options, market_analysis
        )

        # Select optimal strategy
        selected_strategy = await self._select_optimal_strategy(option_evaluations)

        # Create execution plan
        execution_plan = await self._create_strategy_execution_plan(
            selected_strategy, business_context
        )

        return {
            'strategy': selected_strategy,
            'execution_plan': execution_plan,
            'market_analysis': market_analysis,
            'competitive_analysis': competitive_analysis,
            'alternative_strategies': [opt for opt in strategic_options if opt != selected_strategy],
            'success_metrics': self._define_success_metrics(selected_strategy),
            'risk_assessment': await self._assess_strategy_risks(selected_strategy, execution_plan)
        }

    async def _create_strategy_execution_plan(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed execution plan for selected strategy"""

        # Break strategy into strategic initiatives
        initiatives = await self._decompose_strategy_into_initiatives(strategy)

        # Plan implementation phases
        implementation_phases = []
        for initiative in initiatives:
            phase_plan = await self._plan_initiative_implementation(initiative, context)
            implementation_phases.append(phase_plan)

        # Create integrated timeline
        integrated_timeline = await self._create_integrated_timeline(implementation_phases)

        # Plan resource allocation
        resource_plan = await self._plan_resource_allocation(implementation_phases, context)

        # Define governance structure
        governance_plan = await self._plan_strategy_governance(strategy, implementation_phases)

        return {
            'initiatives': initiatives,
            'implementation_phases': implementation_phases,
            'timeline': integrated_timeline,
            'resource_plan': resource_plan,
            'governance': governance_plan,
            'monitoring_framework': self._create_monitoring_framework(strategy, initiatives)
        }
```

## Best Practices

### Planning Design Principles
- **Goal Clarity**: Ensure objectives are specific, measurable, and achievable
- **Hierarchical Structure**: Use multiple levels of abstraction for complex plans
- **Dependency Management**: Explicitly model and track task dependencies
- **Flexibility**: Design plans that can adapt to changing circumstances

### Plan Quality Assurance
- **Validation**: Verify plan feasibility and resource requirements
- **Risk Assessment**: Identify potential risks and mitigation strategies
- **Contingency Planning**: Prepare alternative approaches for critical paths
- **Stakeholder Alignment**: Ensure plans align with stakeholder expectations

### Execution Monitoring
- **Progress Tracking**: Implement mechanisms to monitor plan execution
- **Feedback Integration**: Use execution feedback to improve planning
- **Adaptive Replanning**: Adjust plans based on real-world outcomes
- **Success Metrics**: Define clear criteria for measuring plan success

## Common Pitfalls

### Over-Planning
- **Problem**: Creating overly detailed plans that become rigid and hard to adapt
- **Solution**: Balance planning detail with execution flexibility
- **Mitigation**: Use progressive planning with increasing detail closer to execution

### Unrealistic Assumptions
- **Problem**: Plans based on overly optimistic or unrealistic assumptions
- **Solution**: Use historical data and expert judgment for realistic planning
- **Mitigation**: Build buffers and contingencies into plans

### Insufficient Stakeholder Input
- **Problem**: Plans that don't consider all relevant stakeholders and constraints
- **Solution**: Implement comprehensive stakeholder consultation processes
- **Mitigation**: Regular stakeholder reviews and feedback incorporation

### Poor Dependency Management
- **Problem**: Inadequate modeling of task dependencies leading to execution issues
- **Solution**: Use sophisticated dependency modeling and critical path analysis
- **Mitigation**: Regular dependency review and validation

### Static Planning
- **Problem**: Plans that don't adapt to changing circumstances
- **Solution**: Implement dynamic replanning capabilities
- **Mitigation**: Regular plan reviews and updates based on feedback

## Advanced Concepts

### Multi-Agent Planning
- Coordinated planning across multiple AI agents
- Distributed planning for large-scale systems
- Collaborative plan development and execution

### Hierarchical Planning
- Multiple levels of planning abstraction
- Strategic, tactical, and operational planning integration
- Recursive planning decomposition

### Probabilistic Planning
- Planning under uncertainty with probabilistic outcomes
- Risk-aware planning with multiple scenarios
- Adaptive planning based on probability updates

## Conclusion

Planning is a fundamental pattern that enables AI systems to tackle complex, multi-step challenges through systematic decomposition and structured execution. By implementing sophisticated planning capabilities, AI agents can work toward long-term objectives while managing dependencies, resources, and uncertainties. Success with planning requires balancing detail with flexibility, incorporating stakeholder input, and maintaining adaptability in the face of changing circumstances.