# Chapter 19: Evaluation and Monitoring

**Pattern Description:** Evaluation and Monitoring patterns provide comprehensive frameworks for assessing AI agent performance, tracking system health, measuring quality metrics, and ensuring continuous improvement through systematic observation and analysis.

## Introduction

Evaluation and Monitoring represent critical components in the lifecycle of AI agent systems, enabling organizations to assess performance, ensure reliability, and drive continuous improvement. These patterns encompass both real-time monitoring of operational systems and comprehensive evaluation frameworks for assessing agent capabilities, quality, and effectiveness.

The complexity of modern AI agents requires sophisticated monitoring approaches that go beyond traditional system metrics to include behavioral analysis, quality assessment, user satisfaction measurement, and ethical compliance tracking. Effective evaluation and monitoring enable early detection of issues, performance optimization, and evidence-based decision-making about system improvements.

As AI agents become more autonomous and handle increasingly critical tasks, the importance of robust evaluation and monitoring cannot be overstated. These patterns provide the foundation for trustworthy AI systems by ensuring transparency, accountability, and continuous quality assurance throughout the agent lifecycle.

## Key Concepts

### Performance Monitoring Framework

#### System-Level Metrics
- **Resource Utilization**: CPU, memory, disk, and network usage tracking
- **Response Time**: Latency measurement across different operations and conditions
- **Throughput**: Request processing capacity and scalability metrics
- **Availability**: Uptime, downtime, and service reliability measurements

#### Agent-Level Metrics
- **Task Success Rate**: Percentage of successfully completed tasks
- **Quality Scores**: Accuracy, precision, recall, and F1 scores for agent outputs
- **User Satisfaction**: Feedback scores and user experience metrics
- **Error Rates**: Classification and tracking of different error types

#### Business-Level Metrics
- **Goal Achievement**: Measurement of how well agents achieve business objectives
- **Cost Efficiency**: Resource costs versus value delivered
- **ROI Metrics**: Return on investment from agent deployment
- **SLA Compliance**: Service level agreement adherence tracking

### Evaluation Methodologies

#### Automated Evaluation
- **Unit Testing**: Testing individual agent components and functions
- **Integration Testing**: Testing agent interactions with other systems
- **Regression Testing**: Ensuring new changes don't break existing functionality
- **Performance Benchmarking**: Standardized performance comparisons

#### Human Evaluation
- **Expert Review**: Domain expert assessment of agent outputs and decisions
- **User Studies**: Real-world user interaction and feedback collection
- **Blind Testing**: Unbiased evaluation without knowledge of system details
- **Comparative Analysis**: Side-by-side comparison with alternative solutions

#### Continuous Evaluation
- **A/B Testing**: Comparing different agent versions or configurations
- **Canary Deployment**: Gradual rollout with continuous monitoring
- **Shadow Mode**: Running new versions alongside production without affecting users
- **Champion-Challenger**: Ongoing comparison between current and candidate systems

### Quality Assurance Framework

#### Output Quality Assessment
- **Accuracy Measurement**: Correctness of agent responses and decisions
- **Consistency Tracking**: Uniformity of outputs across similar inputs
- **Completeness Evaluation**: Assessment of response thoroughness
- **Relevance Scoring**: Measurement of output relevance to input queries

#### Behavioral Quality Assessment
- **Bias Detection**: Identification of unfair or discriminatory behaviors
- **Safety Compliance**: Adherence to safety guidelines and constraints
- **Ethical Behavior**: Assessment of ethical decision-making
- **Robustness Testing**: Performance under adversarial or edge case conditions

## Implementation

### Comprehensive Monitoring System

```python
import asyncio
import time
import json
import logging
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricValue:
    timestamp: float
    value: Union[int, float]
    labels: Dict[str, str] = None

@dataclass
class Alert:
    id: str
    timestamp: float
    level: AlertLevel
    metric_name: str
    threshold_value: Union[int, float]
    actual_value: Union[int, float]
    message: str
    resolved: bool = False
    resolution_timestamp: Optional[float] = None

class Metric:
    def __init__(self, name: str, metric_type: MetricType, description: str = ""):
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self.values: deque = deque(maxlen=10000)  # Keep last 10k values
        self.thresholds: Dict[AlertLevel, float] = {}

    def record(self, value: Union[int, float], labels: Dict[str, str] = None):
        """Record a metric value"""
        metric_value = MetricValue(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        self.values.append(metric_value)

    def get_current_value(self) -> Optional[Union[int, float]]:
        """Get the most recent metric value"""
        if self.values:
            return self.values[-1].value
        return None

    def get_statistics(self, window_seconds: int = 300) -> Dict[str, float]:
        """Get statistical summary over a time window"""
        cutoff_time = time.time() - window_seconds
        recent_values = [
            mv.value for mv in self.values
            if mv.timestamp >= cutoff_time
        ]

        if not recent_values:
            return {}

        return {
            'count': len(recent_values),
            'mean': statistics.mean(recent_values),
            'median': statistics.median(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'std_dev': statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        }

    def set_threshold(self, level: AlertLevel, value: float):
        """Set alert threshold for a specific level"""
        self.thresholds[level] = value

class MonitoringSystem:
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self.evaluation_results: List[Dict] = []

        # Monitoring configuration
        self.monitoring_interval = 10.0  # seconds
        self.alert_cooldown = 300.0  # 5 minutes
        self.last_alerts: Dict[str, float] = {}

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

    def create_metric(self, name: str, metric_type: MetricType,
                     description: str = "") -> Metric:
        """Create a new metric"""
        metric = Metric(name, metric_type, description)
        self.metrics[name] = metric
        return metric

    def record_metric(self, name: str, value: Union[int, float],
                     labels: Dict[str, str] = None):
        """Record a value for a metric"""
        if name in self.metrics:
            self.metrics[name].record(value, labels)
        else:
            # Auto-create gauge metric if it doesn't exist
            metric = self.create_metric(name, MetricType.GAUGE)
            metric.record(value, labels)

    def set_alert_threshold(self, metric_name: str, level: AlertLevel, threshold: float):
        """Set alert threshold for a metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].set_threshold(level, threshold)

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._check_thresholds()
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")

    async def _check_thresholds(self):
        """Check metric thresholds and generate alerts"""
        current_time = time.time()

        for metric_name, metric in self.metrics.items():
            current_value = metric.get_current_value()
            if current_value is None:
                continue

            for level, threshold in metric.thresholds.items():
                # Check if threshold is exceeded
                threshold_exceeded = False
                if level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]:
                    threshold_exceeded = current_value > threshold
                else:  # INFO level
                    threshold_exceeded = current_value < threshold

                if threshold_exceeded:
                    # Check cooldown to avoid alert spam
                    last_alert_key = f"{metric_name}_{level.value}"
                    last_alert_time = self.last_alerts.get(last_alert_key, 0)

                    if current_time - last_alert_time > self.alert_cooldown:
                        alert = Alert(
                            id=f"alert_{len(self.alerts)}_{int(current_time)}",
                            timestamp=current_time,
                            level=level,
                            metric_name=metric_name,
                            threshold_value=threshold,
                            actual_value=current_value,
                            message=f"{metric_name} threshold exceeded: {current_value} > {threshold}"
                        )

                        self.alerts.append(alert)
                        self.last_alerts[last_alert_key] = current_time

                        # Notify callbacks
                        for callback in self.alert_callbacks:
                            try:
                                await callback(alert)
                            except Exception as e:
                                logging.error(f"Error in alert callback: {e}")

    def get_metric_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive metric dashboard data"""
        dashboard_data = {
            'timestamp': time.time(),
            'metrics': {},
            'alerts': {
                'active': len([a for a in self.alerts if not a.resolved]),
                'total': len(self.alerts),
                'by_level': defaultdict(int)
            }
        }

        # Compile metric data
        for name, metric in self.metrics.items():
            stats = metric.get_statistics()
            dashboard_data['metrics'][name] = {
                'type': metric.metric_type.value,
                'current_value': metric.get_current_value(),
                'statistics': stats,
                'thresholds': {level.value: threshold for level, threshold in metric.thresholds.items()}
            }

        # Compile alert statistics
        for alert in self.alerts:
            dashboard_data['alerts']['by_level'][alert.level.value] += 1

        return dashboard_data

class AgentEvaluator:
    def __init__(self, monitoring_system: MonitoringSystem):
        self.monitoring = monitoring_system
        self.evaluation_metrics = {}
        self.ground_truth_data: Dict[str, Any] = {}
        self.evaluation_history: List[Dict] = []

    def add_evaluation_metric(self, name: str, evaluator_func: Callable):
        """Add an evaluation metric with custom evaluator function"""
        self.evaluation_metrics[name] = evaluator_func

    def set_ground_truth(self, task_id: str, expected_result: Any):
        """Set ground truth data for evaluation"""
        self.ground_truth_data[task_id] = expected_result

    async def evaluate_agent_output(self, agent_id: str, task_id: str,
                                  actual_result: Any, context: Dict = None) -> Dict[str, Any]:
        """Evaluate agent output against ground truth and metrics"""
        evaluation_result = {
            'agent_id': agent_id,
            'task_id': task_id,
            'timestamp': time.time(),
            'scores': {},
            'overall_score': 0.0,
            'context': context or {}
        }

        # Get ground truth if available
        expected_result = self.ground_truth_data.get(task_id)

        # Apply evaluation metrics
        total_score = 0.0
        metric_count = 0

        for metric_name, evaluator_func in self.evaluation_metrics.items():
            try:
                score = await evaluator_func(actual_result, expected_result, context)
                evaluation_result['scores'][metric_name] = score
                total_score += score
                metric_count += 1

                # Record metric for monitoring
                self.monitoring.record_metric(f"evaluation_{metric_name}", score, {
                    'agent_id': agent_id,
                    'task_type': context.get('task_type', 'unknown')
                })

            except Exception as e:
                logging.error(f"Error evaluating metric {metric_name}: {e}")
                evaluation_result['scores'][metric_name] = 0.0

        # Calculate overall score
        if metric_count > 0:
            evaluation_result['overall_score'] = total_score / metric_count

        # Record overall evaluation score
        self.monitoring.record_metric("evaluation_overall", evaluation_result['overall_score'], {
            'agent_id': agent_id
        })

        # Store evaluation result
        self.evaluation_history.append(evaluation_result)

        return evaluation_result

    async def batch_evaluate(self, evaluation_data: List[Dict]) -> Dict[str, Any]:
        """Perform batch evaluation on multiple agent outputs"""
        batch_results = []

        for item in evaluation_data:
            result = await self.evaluate_agent_output(
                item['agent_id'],
                item['task_id'],
                item['actual_result'],
                item.get('context')
            )
            batch_results.append(result)

        # Calculate batch statistics
        if batch_results:
            overall_scores = [r['overall_score'] for r in batch_results]
            batch_summary = {
                'timestamp': time.time(),
                'total_evaluations': len(batch_results),
                'mean_score': statistics.mean(overall_scores),
                'median_score': statistics.median(overall_scores),
                'min_score': min(overall_scores),
                'max_score': max(overall_scores),
                'std_dev': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                'results': batch_results
            }
        else:
            batch_summary = {
                'timestamp': time.time(),
                'total_evaluations': 0,
                'results': []
            }

        return batch_summary

    def get_agent_performance_report(self, agent_id: str,
                                   time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate performance report for a specific agent"""
        cutoff_time = time.time() - (time_window_hours * 3600)

        # Filter evaluations for this agent and time window
        agent_evaluations = [
            eval_result for eval_result in self.evaluation_history
            if (eval_result['agent_id'] == agent_id and
                eval_result['timestamp'] >= cutoff_time)
        ]

        if not agent_evaluations:
            return {
                'agent_id': agent_id,
                'time_window_hours': time_window_hours,
                'total_evaluations': 0,
                'message': 'No evaluations found for this agent in the specified time window'
            }

        # Calculate performance metrics
        overall_scores = [e['overall_score'] for e in agent_evaluations]
        metric_scores = defaultdict(list)

        for evaluation in agent_evaluations:
            for metric_name, score in evaluation['scores'].items():
                metric_scores[metric_name].append(score)

        report = {
            'agent_id': agent_id,
            'time_window_hours': time_window_hours,
            'total_evaluations': len(agent_evaluations),
            'overall_performance': {
                'mean_score': statistics.mean(overall_scores),
                'median_score': statistics.median(overall_scores),
                'min_score': min(overall_scores),
                'max_score': max(overall_scores),
                'std_dev': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0
            },
            'metric_performance': {}
        }

        # Calculate per-metric performance
        for metric_name, scores in metric_scores.items():
            report['metric_performance'][metric_name] = {
                'mean_score': statistics.mean(scores),
                'median_score': statistics.median(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0
            }

        return report

# Standard evaluation metrics
async def accuracy_metric(actual: Any, expected: Any, context: Dict = None) -> float:
    """Calculate accuracy metric"""
    if expected is None:
        return 1.0  # No ground truth available

    if isinstance(actual, str) and isinstance(expected, str):
        return 1.0 if actual.strip().lower() == expected.strip().lower() else 0.0
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        tolerance = context.get('tolerance', 0.01) if context else 0.01
        return 1.0 if abs(actual - expected) <= tolerance else 0.0
    else:
        return 1.0 if actual == expected else 0.0

async def completeness_metric(actual: Any, expected: Any, context: Dict = None) -> float:
    """Calculate completeness metric based on expected elements"""
    if expected is None:
        return 1.0

    if isinstance(actual, str) and isinstance(expected, str):
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        if len(expected_words) == 0:
            return 1.0
        return len(actual_words & expected_words) / len(expected_words)

    elif isinstance(actual, list) and isinstance(expected, list):
        expected_set = set(expected)
        actual_set = set(actual)
        if len(expected_set) == 0:
            return 1.0
        return len(actual_set & expected_set) / len(expected_set)

    return 1.0 if actual == expected else 0.0

async def relevance_metric(actual: Any, expected: Any, context: Dict = None) -> float:
    """Calculate relevance metric based on context"""
    if context is None:
        return 1.0

    query_keywords = context.get('query_keywords', [])
    if not query_keywords:
        return 1.0

    if isinstance(actual, str):
        actual_words = set(actual.lower().split())
        keyword_matches = sum(1 for keyword in query_keywords if keyword.lower() in actual_words)
        return keyword_matches / len(query_keywords) if query_keywords else 1.0

    return 1.0
```

### Real-time Performance Monitor

```python
import psutil
import threading
from datetime import datetime, timedelta

class PerformanceMonitor:
    def __init__(self, monitoring_system: MonitoringSystem):
        self.monitoring = monitoring_system
        self.agent_metrics: Dict[str, Dict] = defaultdict(dict)
        self.system_metrics_enabled = True
        self.agent_metrics_enabled = True

        # Initialize system metrics
        self._init_system_metrics()

        # Start monitoring threads
        if self.system_metrics_enabled:
            self.system_thread = threading.Thread(target=self._monitor_system_metrics)
            self.system_thread.daemon = True
            self.system_thread.start()

    def _init_system_metrics(self):
        """Initialize system-level metrics"""
        self.monitoring.create_metric("system_cpu_percent", MetricType.GAUGE, "CPU utilization percentage")
        self.monitoring.create_metric("system_memory_percent", MetricType.GAUGE, "Memory utilization percentage")
        self.monitoring.create_metric("system_disk_usage_percent", MetricType.GAUGE, "Disk usage percentage")
        self.monitoring.create_metric("system_network_bytes_sent", MetricType.COUNTER, "Network bytes sent")
        self.monitoring.create_metric("system_network_bytes_recv", MetricType.COUNTER, "Network bytes received")

        # Set default thresholds
        self.monitoring.set_alert_threshold("system_cpu_percent", AlertLevel.WARNING, 80.0)
        self.monitoring.set_alert_threshold("system_cpu_percent", AlertLevel.CRITICAL, 95.0)
        self.monitoring.set_alert_threshold("system_memory_percent", AlertLevel.WARNING, 85.0)
        self.monitoring.set_alert_threshold("system_memory_percent", AlertLevel.CRITICAL, 95.0)

    def _monitor_system_metrics(self):
        """Monitor system-level metrics in a separate thread"""
        while self.system_metrics_enabled:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.monitoring.record_metric("system_cpu_percent", cpu_percent)

                # Memory metrics
                memory = psutil.virtual_memory()
                self.monitoring.record_metric("system_memory_percent", memory.percent)

                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.monitoring.record_metric("system_disk_usage_percent", disk_percent)

                # Network metrics
                network = psutil.net_io_counters()
                if network:
                    self.monitoring.record_metric("system_network_bytes_sent", network.bytes_sent)
                    self.monitoring.record_metric("system_network_bytes_recv", network.bytes_recv)

                time.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                logging.error(f"Error monitoring system metrics: {e}")
                time.sleep(10)

    def record_agent_task_start(self, agent_id: str, task_id: str, task_type: str = "unknown"):
        """Record the start of an agent task"""
        task_key = f"{agent_id}_{task_id}"
        self.agent_metrics[task_key] = {
            'agent_id': agent_id,
            'task_id': task_id,
            'task_type': task_type,
            'start_time': time.time(),
            'end_time': None,
            'success': None,
            'error': None
        }

    def record_agent_task_end(self, agent_id: str, task_id: str, success: bool = True, error: str = None):
        """Record the end of an agent task"""
        task_key = f"{agent_id}_{task_id}"
        if task_key in self.agent_metrics:
            task_data = self.agent_metrics[task_key]
            task_data['end_time'] = time.time()
            task_data['success'] = success
            task_data['error'] = error

            # Calculate and record metrics
            duration = task_data['end_time'] - task_data['start_time']

            # Record task duration
            self.monitoring.record_metric("agent_task_duration", duration, {
                'agent_id': agent_id,
                'task_type': task_data['task_type']
            })

            # Record success/failure
            self.monitoring.record_metric("agent_task_success", 1.0 if success else 0.0, {
                'agent_id': agent_id,
                'task_type': task_data['task_type']
            })

            # Record task completion
            self.monitoring.record_metric("agent_tasks_completed", 1, {
                'agent_id': agent_id,
                'task_type': task_data['task_type']
            })

            # Clean up completed task
            del self.agent_metrics[task_key]

    def get_agent_performance_summary(self, agent_id: str, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for an agent"""
        cutoff_time = time.time() - (hours * 3600)

        # Get relevant metrics from monitoring system
        task_duration_metric = self.monitoring.metrics.get("agent_task_duration")
        task_success_metric = self.monitoring.metrics.get("agent_task_success")

        if not task_duration_metric or not task_success_metric:
            return {
                'agent_id': agent_id,
                'message': 'No performance data available'
            }

        # Filter metrics for this agent and time window
        duration_values = [
            mv.value for mv in task_duration_metric.values
            if (mv.timestamp >= cutoff_time and
                mv.labels.get('agent_id') == agent_id)
        ]

        success_values = [
            mv.value for mv in task_success_metric.values
            if (mv.timestamp >= cutoff_time and
                mv.labels.get('agent_id') == agent_id)
        ]

        if not duration_values or not success_values:
            return {
                'agent_id': agent_id,
                'message': f'No performance data found for the last {hours} hours'
            }

        # Calculate summary statistics
        summary = {
            'agent_id': agent_id,
            'time_window_hours': hours,
            'total_tasks': len(duration_values),
            'success_rate': sum(success_values) / len(success_values) * 100,
            'average_duration': statistics.mean(duration_values),
            'median_duration': statistics.median(duration_values),
            'min_duration': min(duration_values),
            'max_duration': max(duration_values),
            'tasks_per_hour': len(duration_values) / hours
        }

        return summary

class QualityAssuranceMonitor:
    def __init__(self, monitoring_system: MonitoringSystem):
        self.monitoring = monitoring_system
        self.quality_checks: List[Callable] = []
        self.quality_history: List[Dict] = []

    def add_quality_check(self, check_func: Callable):
        """Add a quality assurance check function"""
        self.quality_checks.append(check_func)

    async def run_quality_checks(self, agent_id: str, output_data: Any,
                               context: Dict = None) -> Dict[str, Any]:
        """Run all quality checks on agent output"""
        check_results = {
            'agent_id': agent_id,
            'timestamp': time.time(),
            'checks': {},
            'overall_quality_score': 0.0,
            'quality_grade': 'F'
        }

        total_score = 0.0
        check_count = 0

        for i, check_func in enumerate(self.quality_checks):
            try:
                check_name = getattr(check_func, '__name__', f'check_{i}')
                result = await check_func(output_data, context)

                check_results['checks'][check_name] = result
                total_score += result.get('score', 0.0)
                check_count += 1

                # Record individual check metric
                self.monitoring.record_metric(f"quality_check_{check_name}", result.get('score', 0.0), {
                    'agent_id': agent_id
                })

            except Exception as e:
                logging.error(f"Error running quality check {i}: {e}")
                check_results['checks'][f'check_{i}'] = {
                    'score': 0.0,
                    'error': str(e)
                }

        # Calculate overall quality score
        if check_count > 0:
            check_results['overall_quality_score'] = total_score / check_count
            check_results['quality_grade'] = self._score_to_grade(check_results['overall_quality_score'])

        # Record overall quality score
        self.monitoring.record_metric("overall_quality_score", check_results['overall_quality_score'], {
            'agent_id': agent_id
        })

        # Store quality check result
        self.quality_history.append(check_results)

        return check_results

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'

    def get_quality_trend(self, agent_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get quality trend for an agent over time"""
        cutoff_time = time.time() - (hours * 3600)

        relevant_checks = [
            check for check in self.quality_history
            if (check['agent_id'] == agent_id and
                check['timestamp'] >= cutoff_time)
        ]

        if not relevant_checks:
            return {
                'agent_id': agent_id,
                'message': f'No quality data found for the last {hours} hours'
            }

        scores = [check['overall_quality_score'] for check in relevant_checks]
        timestamps = [check['timestamp'] for check in relevant_checks]

        # Calculate trend (simple linear regression slope)
        if len(scores) > 1:
            x = np.array(timestamps)
            y = np.array(scores)
            slope = np.polyfit(x, y, 1)[0]
            trend_direction = "improving" if slope > 0.001 else "declining" if slope < -0.001 else "stable"
        else:
            slope = 0.0
            trend_direction = "insufficient_data"

        return {
            'agent_id': agent_id,
            'time_window_hours': hours,
            'total_quality_checks': len(relevant_checks),
            'average_quality_score': statistics.mean(scores),
            'latest_quality_score': scores[-1] if scores else 0.0,
            'trend_slope': slope,
            'trend_direction': trend_direction,
            'quality_variance': statistics.variance(scores) if len(scores) > 1 else 0.0
        }

# Example quality check functions
async def output_completeness_check(output_data: Any, context: Dict = None) -> Dict[str, Any]:
    """Check if output is complete and comprehensive"""
    if isinstance(output_data, str):
        word_count = len(output_data.split())
        min_words = context.get('min_word_count', 10) if context else 10

        score = min(word_count / min_words, 1.0)

        return {
            'score': score,
            'word_count': word_count,
            'min_required': min_words,
            'passed': score >= 1.0
        }

    return {'score': 1.0, 'passed': True}

async def output_coherence_check(output_data: Any, context: Dict = None) -> Dict[str, Any]:
    """Check output coherence and logical flow"""
    if isinstance(output_data, str):
        sentences = output_data.split('.')
        sentence_count = len([s for s in sentences if s.strip()])

        # Simple coherence heuristics
        avg_sentence_length = len(output_data.split()) / max(sentence_count, 1)
        coherence_score = 1.0

        # Penalize very short or very long sentences
        if avg_sentence_length < 5:
            coherence_score *= 0.7
        elif avg_sentence_length > 50:
            coherence_score *= 0.8

        # Check for repetitive patterns
        words = output_data.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / max(len(words), 1)

        if repetition_ratio < 0.5:
            coherence_score *= 0.6

        return {
            'score': coherence_score,
            'avg_sentence_length': avg_sentence_length,
            'repetition_ratio': repetition_ratio,
            'passed': coherence_score >= 0.7
        }

    return {'score': 1.0, 'passed': True}

async def output_safety_check(output_data: Any, context: Dict = None) -> Dict[str, Any]:
    """Check output for safety and appropriateness"""
    if isinstance(output_data, str):
        # Simple safety check - look for concerning patterns
        concerning_patterns = [
            'harmful', 'dangerous', 'illegal', 'unethical'
        ]

        safety_violations = []
        for pattern in concerning_patterns:
            if pattern in output_data.lower():
                safety_violations.append(pattern)

        safety_score = max(0.0, 1.0 - (len(safety_violations) * 0.3))

        return {
            'score': safety_score,
            'violations': safety_violations,
            'passed': len(safety_violations) == 0
        }

    return {'score': 1.0, 'passed': True}
```

## Code Examples

### Advanced Analytics Dashboard

```python
class AnalyticsDashboard:
    def __init__(self, monitoring_system: MonitoringSystem,
                 evaluator: AgentEvaluator,
                 performance_monitor: PerformanceMonitor):
        self.monitoring = monitoring_system
        self.evaluator = evaluator
        self.performance_monitor = performance_monitor

    def generate_comprehensive_report(self, agent_ids: List[str] = None,
                                    time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        cutoff_time = time.time() - (time_window_hours * 3600)

        report = {
            'report_timestamp': time.time(),
            'time_window_hours': time_window_hours,
            'system_overview': self._get_system_overview(),
            'agent_performance': {},
            'quality_analysis': {},
            'alert_summary': self._get_alert_summary(cutoff_time),
            'recommendations': []
        }

        # Filter agents if specified
        if agent_ids is None:
            agent_ids = self._get_active_agent_ids(cutoff_time)

        # Generate per-agent analysis
        for agent_id in agent_ids:
            report['agent_performance'][agent_id] = self._analyze_agent_performance(
                agent_id, time_window_hours
            )
            report['quality_analysis'][agent_id] = self._analyze_agent_quality(
                agent_id, time_window_hours
            )

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)

        return report

    def _get_system_overview(self) -> Dict[str, Any]:
        """Get system-level overview"""
        dashboard_data = self.monitoring.get_metric_dashboard()

        return {
            'total_metrics': len(dashboard_data['metrics']),
            'active_alerts': dashboard_data['alerts']['active'],
            'system_health': self._calculate_system_health(dashboard_data['metrics']),
            'key_metrics': {
                'cpu_usage': dashboard_data['metrics'].get('system_cpu_percent', {}).get('current_value'),
                'memory_usage': dashboard_data['metrics'].get('system_memory_percent', {}).get('current_value'),
                'disk_usage': dashboard_data['metrics'].get('system_disk_usage_percent', {}).get('current_value')
            }
        }

    def _calculate_system_health(self, metrics: Dict[str, Any]) -> str:
        """Calculate overall system health status"""
        health_indicators = []

        # Check CPU usage
        cpu_metric = metrics.get('system_cpu_percent', {})
        cpu_value = cpu_metric.get('current_value', 0)
        if cpu_value > 90:
            health_indicators.append('cpu_critical')
        elif cpu_value > 80:
            health_indicators.append('cpu_warning')

        # Check memory usage
        memory_metric = metrics.get('system_memory_percent', {})
        memory_value = memory_metric.get('current_value', 0)
        if memory_value > 95:
            health_indicators.append('memory_critical')
        elif memory_value > 85:
            health_indicators.append('memory_warning')

        # Determine overall health
        if any('critical' in indicator for indicator in health_indicators):
            return 'critical'
        elif any('warning' in indicator for indicator in health_indicators):
            return 'warning'
        else:
            return 'healthy'

    def _get_active_agent_ids(self, cutoff_time: float) -> List[str]:
        """Get list of active agent IDs in the time window"""
        agent_ids = set()

        # Check evaluation history
        for evaluation in self.evaluator.evaluation_history:
            if evaluation['timestamp'] >= cutoff_time:
                agent_ids.add(evaluation['agent_id'])

        return list(agent_ids)

    def _analyze_agent_performance(self, agent_id: str, hours: int) -> Dict[str, Any]:
        """Analyze performance for a specific agent"""
        performance_summary = self.performance_monitor.get_agent_performance_summary(
            agent_id, hours
        )

        evaluation_report = self.evaluator.get_agent_performance_report(
            agent_id, hours
        )

        return {
            'task_performance': performance_summary,
            'quality_performance': evaluation_report,
            'performance_grade': self._calculate_performance_grade(
                performance_summary, evaluation_report
            )
        }

    def _analyze_agent_quality(self, agent_id: str, hours: int) -> Dict[str, Any]:
        """Analyze quality metrics for a specific agent"""
        # This would integrate with QualityAssuranceMonitor if available
        return {
            'quality_score': 0.85,  # Placeholder
            'trend': 'stable',
            'improvement_areas': []
        }

    def _get_alert_summary(self, cutoff_time: float) -> Dict[str, Any]:
        """Get summary of alerts in the time window"""
        recent_alerts = [
            alert for alert in self.monitoring.alerts
            if alert.timestamp >= cutoff_time
        ]

        alert_summary = {
            'total_alerts': len(recent_alerts),
            'by_level': defaultdict(int),
            'by_metric': defaultdict(int),
            'resolution_rate': 0.0
        }

        resolved_count = 0
        for alert in recent_alerts:
            alert_summary['by_level'][alert.level.value] += 1
            alert_summary['by_metric'][alert.metric_name] += 1
            if alert.resolved:
                resolved_count += 1

        if recent_alerts:
            alert_summary['resolution_rate'] = resolved_count / len(recent_alerts)

        return alert_summary

    def _calculate_performance_grade(self, task_perf: Dict, quality_perf: Dict) -> str:
        """Calculate overall performance grade"""
        # Simple grading algorithm
        success_rate = task_perf.get('success_rate', 0)
        avg_score = quality_perf.get('overall_performance', {}).get('mean_score', 0)

        combined_score = (success_rate / 100 + avg_score) / 2

        if combined_score >= 0.9:
            return 'A'
        elif combined_score >= 0.8:
            return 'B'
        elif combined_score >= 0.7:
            return 'C'
        elif combined_score >= 0.6:
            return 'D'
        else:
            return 'F'

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on report data"""
        recommendations = []

        # System-level recommendations
        system_health = report['system_overview']['system_health']
        if system_health == 'critical':
            recommendations.append("URGENT: System resources are critically low. Consider scaling up infrastructure.")
        elif system_health == 'warning':
            recommendations.append("WARNING: System resources are under pressure. Monitor closely and consider optimization.")

        # Agent-level recommendations
        for agent_id, performance in report['agent_performance'].items():
            grade = performance.get('performance_grade', 'F')
            if grade in ['D', 'F']:
                recommendations.append(f"Agent {agent_id} performance is below acceptable levels. Review and retrain.")

            success_rate = performance.get('task_performance', {}).get('success_rate', 0)
            if success_rate < 80:
                recommendations.append(f"Agent {agent_id} has low success rate ({success_rate:.1f}%). Investigate common failure modes.")

        # Alert-based recommendations
        alert_summary = report['alert_summary']
        if alert_summary['total_alerts'] > 50:
            recommendations.append("High alert volume detected. Review alert thresholds and implement noise reduction.")

        return recommendations

# Example usage
async def monitoring_example():
    # Create monitoring system
    monitoring = MonitoringSystem()

    # Create evaluator
    evaluator = AgentEvaluator(monitoring)

    # Add evaluation metrics
    evaluator.add_evaluation_metric('accuracy', accuracy_metric)
    evaluator.add_evaluation_metric('completeness', completeness_metric)
    evaluator.add_evaluation_metric('relevance', relevance_metric)

    # Create performance monitor
    performance_monitor = PerformanceMonitor(monitoring)

    # Create quality monitor
    quality_monitor = QualityAssuranceMonitor(monitoring)
    quality_monitor.add_quality_check(output_completeness_check)
    quality_monitor.add_quality_check(output_coherence_check)
    quality_monitor.add_quality_check(output_safety_check)

    # Create analytics dashboard
    dashboard = AnalyticsDashboard(monitoring, evaluator, performance_monitor)

    # Simulate some agent activity
    agent_id = "test_agent"

    # Start a task
    task_id = "task_001"
    performance_monitor.record_agent_task_start(agent_id, task_id, "text_generation")

    # Simulate processing time
    await asyncio.sleep(2)

    # End the task
    performance_monitor.record_agent_task_end(agent_id, task_id, success=True)

    # Evaluate agent output
    test_output = "This is a comprehensive response to the user's query about monitoring systems."
    evaluation_result = await evaluator.evaluate_agent_output(
        agent_id, task_id, test_output,
        context={'query_keywords': ['monitoring', 'systems']}
    )

    # Run quality checks
    quality_result = await quality_monitor.run_quality_checks(
        agent_id, test_output,
        context={'min_word_count': 10}
    )

    # Generate comprehensive report
    report = dashboard.generate_comprehensive_report([agent_id])

    print("Monitoring Example Results:")
    print(f"Evaluation: {evaluation_result}")
    print(f"Quality: {quality_result}")
    print(f"Report: {json.dumps(report, indent=2, default=str)}")

# Run example
# asyncio.run(monitoring_example())
```

## Best Practices

### Comprehensive Monitoring Strategy
- **Multi-Level Monitoring**: Implement monitoring at system, application, and business levels
- **Real-Time and Batch Monitoring**: Combine real-time alerting with periodic batch analysis
- **Baseline Establishment**: Establish performance baselines and track deviations
- **Contextual Monitoring**: Consider context when interpreting metrics and alerts

### Evaluation Framework Design
- **Multiple Evaluation Methods**: Use both automated and human evaluation approaches
- **Diverse Metrics**: Implement comprehensive metrics covering accuracy, quality, safety, and user satisfaction
- **Ground Truth Management**: Maintain high-quality ground truth data for reliable evaluation
- **Continuous Evaluation**: Implement ongoing evaluation rather than point-in-time assessments

### Alert Management
- **Intelligent Alerting**: Use smart thresholds and avoid alert fatigue
- **Alert Prioritization**: Implement clear severity levels and escalation procedures
- **Root Cause Analysis**: Focus on identifying and addressing underlying causes
- **Alert Documentation**: Maintain clear documentation for alert resolution procedures

### Data Quality and Governance
- **Data Validation**: Implement robust data validation for monitoring and evaluation data
- **Retention Policies**: Define appropriate data retention policies for different types of metrics
- **Privacy Protection**: Ensure monitoring and evaluation comply with privacy requirements
- **Audit Trails**: Maintain comprehensive audit trails for compliance and debugging

## Common Pitfalls

### Metric Overload
- **Problem**: Too many metrics leading to information overload and reduced actionability
- **Solution**: Focus on key performance indicators and implement metric hierarchies

### False Positives
- **Problem**: Excessive false alarms reducing trust in the monitoring system
- **Solution**: Tune alert thresholds carefully and implement smart alerting logic

### Evaluation Bias
- **Problem**: Biased evaluation data or methods leading to incorrect performance assessments
- **Solution**: Use diverse evaluation datasets and multiple evaluation perspectives

### Performance Impact
- **Problem**: Monitoring and evaluation overhead affecting system performance
- **Solution**: Optimize monitoring code and use sampling for high-frequency operations

### Data Quality Issues
- **Problem**: Poor quality monitoring data leading to incorrect conclusions
- **Solution**: Implement comprehensive data validation and quality checks

### Alert Fatigue
- **Problem**: Too many alerts causing important ones to be ignored
- **Solution**: Implement intelligent alerting, proper prioritization, and noise reduction

---

*This chapter covers 18 pages of content from "Agentic Design Patterns" by Antonio Gulli, focusing on Evaluation and Monitoring patterns for comprehensive assessment and continuous improvement of AI agent systems.*