"""
Django management command for ML analytics and performance monitoring.

Usage:
    python manage.py ml_analytics dashboard
    python manage.py ml_analytics model-performance
    python manage.py ml_analytics feature-analysis
    python manage.py ml_analytics user-satisfaction
    python manage.py ml_analytics export-metrics --output report.json
"""

import json
import os
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.db.models import Avg, Count, Q
from django.contrib.auth.models import User

from analyzer.models import (
    Query, QueryFeedback, UserQueryHistory,
    MLModel, TrainingData, LearningMetrics, FeedbackLearning
)


class Command(BaseCommand):
    help = 'ML analytics and performance monitoring'

    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest='action', help='Available analytics actions')

        # Dashboard command
        dashboard_parser = subparsers.add_parser('dashboard', help='Show ML system dashboard')

        # Model performance command
        performance_parser = subparsers.add_parser('model-performance', help='Analyze model performance')
        performance_parser.add_argument(
            '--model-version',
            type=str,
            help='Analyze specific model version'
        )
        performance_parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Analysis period in days (default: 30)'
        )

        # Feature analysis command
        feature_parser = subparsers.add_parser('feature-analysis', help='Analyze feature importance and distribution')

        # User satisfaction command
        satisfaction_parser = subparsers.add_parser('user-satisfaction', help='Analyze user satisfaction metrics')
        satisfaction_parser.add_argument(
            '--by-query-type',
            action='store_true',
            help='Break down satisfaction by query type'
        )

        # Export metrics command
        export_parser = subparsers.add_parser('export-metrics', help='Export metrics to file')
        export_parser.add_argument(
            '--output',
            type=str,
            required=True,
            help='Output file path'
        )
        export_parser.add_argument(
            '--format',
            choices=['json', 'csv'],
            default='json',
            help='Export format'
        )
        export_parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Period to export (default: 30 days)'
        )

        # Model comparison command
        comparison_parser = subparsers.add_parser('model-comparison', help='Compare model versions')
        comparison_parser.add_argument(
            'versions',
            nargs='+',
            help='Model versions to compare'
        )

    def handle(self, *args, **options):
        """Handle the analytics command."""
        action = options.get('action')

        if not action:
            self.print_help()
            return

        try:
            if action == 'dashboard':
                self.show_dashboard(options)
            elif action == 'model-performance':
                self.analyze_model_performance(options)
            elif action == 'feature-analysis':
                self.analyze_features(options)
            elif action == 'user-satisfaction':
                self.analyze_user_satisfaction(options)
            elif action == 'export-metrics':
                self.export_metrics(options)
            elif action == 'model-comparison':
                self.compare_models(options)
            else:
                raise CommandError(f'Unknown action: {action}')

        except Exception as e:
            raise CommandError(f'Analytics command failed: {str(e)}')

    def show_dashboard(self, options):
        """Show ML system dashboard."""
        self.stdout.write(self.style.SUCCESS('QueryGrade ML Analytics Dashboard'))
        self.stdout.write('=' * 50)

        # System overview
        self._show_system_overview()

        # Model status
        self._show_model_status()

        # Recent activity
        self._show_recent_activity()

        # Performance summary
        self._show_performance_summary()

        # Alerts and recommendations
        self._show_alerts_and_recommendations()

    def _show_system_overview(self):
        """Show system overview metrics."""
        self.stdout.write('\n📊 System Overview')
        self.stdout.write('-' * 20)

        total_queries = Query.objects.count()
        total_feedback = QueryFeedback.objects.count()
        total_training_data = TrainingData.objects.count()
        total_users = User.objects.filter(queryfeedback__isnull=False).distinct().count()

        self.stdout.write(f'Total queries analyzed: {total_queries:,}')
        self.stdout.write(f'Total feedback items: {total_feedback:,}')
        self.stdout.write(f'Training data samples: {total_training_data:,}')
        self.stdout.write(f'Active users providing feedback: {total_users:,}')

        # Feedback rate
        feedback_rate = (total_feedback / total_queries * 100) if total_queries > 0 else 0
        self.stdout.write(f'Feedback rate: {feedback_rate:.1f}%')

    def _show_model_status(self):
        """Show model status information."""
        self.stdout.write('\n🤖 Model Status')
        self.stdout.write('-' * 15)

        active_models = MLModel.objects.filter(status='ACTIVE')
        total_models = MLModel.objects.count()

        self.stdout.write(f'Total models: {total_models}')
        self.stdout.write(f'Active models: {active_models.count()}')

        for model in active_models:
            accuracy = model.validation_accuracy or 0
            age = (timezone.now() - model.created_at).days
            self.stdout.write(f'  {model.model_type}: {model.version}')
            self.stdout.write(f'    Accuracy: {accuracy:.1%}')
            self.stdout.write(f'    Age: {age} days')

    def _show_recent_activity(self):
        """Show recent activity metrics."""
        self.stdout.write('\n📈 Recent Activity (Last 7 Days)')
        self.stdout.write('-' * 35)

        seven_days_ago = timezone.now() - timedelta(days=7)

        recent_queries = Query.objects.filter(created_at__gte=seven_days_ago).count()
        recent_feedback = QueryFeedback.objects.filter(created_at__gte=seven_days_ago).count()
        recent_models = MLModel.objects.filter(created_at__gte=seven_days_ago).count()

        self.stdout.write(f'New queries: {recent_queries}')
        self.stdout.write(f'New feedback: {recent_feedback}')
        self.stdout.write(f'Models created: {recent_models}')

        # Daily breakdown
        for i in range(7):
            day = timezone.now() - timedelta(days=i)
            day_queries = Query.objects.filter(
                created_at__date=day.date()
            ).count()
            day_feedback = QueryFeedback.objects.filter(
                created_at__date=day.date()
            ).count()

            if day_queries > 0 or day_feedback > 0:
                self.stdout.write(f'  {day.strftime("%Y-%m-%d")}: {day_queries} queries, {day_feedback} feedback')

    def _show_performance_summary(self):
        """Show performance summary."""
        self.stdout.write('\n⚡ Performance Summary')
        self.stdout.write('-' * 20)

        # Latest metrics
        latest_metrics = LearningMetrics.objects.order_by('-created_at').first()
        if latest_metrics:
            self.stdout.write(f'Latest model metrics ({latest_metrics.model.version}):')
            self.stdout.write(f'  Accuracy: {latest_metrics.accuracy:.1%}')
            self.stdout.write(f'  F1 score: {latest_metrics.f1_score:.3f}')
            self.stdout.write(f'  User agreement rate: {latest_metrics.user_agreement_rate:.3f}')
            self.stdout.write(f'  Avg user rating: {latest_metrics.avg_user_rating:.1f}/5.0')

        # User satisfaction trend
        avg_satisfaction = QueryFeedback.objects.aggregate(
            avg=Avg('score_agreement')
        )['avg'] or 0
        self.stdout.write(f'Overall user satisfaction: {avg_satisfaction:.1f}/5.0')

    def _show_alerts_and_recommendations(self):
        """Show alerts and recommendations."""
        self.stdout.write('\n🚨 Alerts & Recommendations')
        self.stdout.write('-' * 30)

        alerts = []
        recommendations = []

        # Check for low accuracy models
        for model in MLModel.objects.filter(status='ACTIVE'):
            accuracy = model.validation_accuracy or 0
            if accuracy < 0.7:
                alerts.append(f'Model {model.version} has low accuracy: {accuracy:.1%}')

        # Check for old models
        for model in MLModel.objects.filter(status='ACTIVE'):
            age = (timezone.now() - model.created_at).days
            if age > 30:
                recommendations.append(f'Consider retraining {model.version} (age: {age} days)')

        # Check training data sufficiency
        training_data_count = TrainingData.objects.count()
        if training_data_count < 100:
            recommendations.append(f'Low training data: {training_data_count} samples (recommended: 100+)')

        # Check recent feedback volume
        recent_feedback = QueryFeedback.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=7)
        ).count()
        if recent_feedback < 10:
            recommendations.append(f'Low recent feedback volume: {recent_feedback} items this week')

        # Display alerts and recommendations
        if alerts:
            for alert in alerts:
                self.stdout.write(self.style.ERROR(f'⚠️  {alert}'))

        if recommendations:
            for rec in recommendations:
                self.stdout.write(self.style.WARNING(f'💡 {rec}'))

        if not alerts and not recommendations:
            self.stdout.write(self.style.SUCCESS('✅ No alerts or recommendations'))

    def analyze_model_performance(self, options):
        """Analyze model performance in detail."""
        self.stdout.write(self.style.SUCCESS('Model Performance Analysis'))
        self.stdout.write('=' * 40)

        model_version = options.get('model_version')
        days = options['days']

        if model_version:
            models = MLModel.objects.filter(version=model_version)
        else:
            models = MLModel.objects.filter(status='ACTIVE')

        for model in models:
            self._analyze_single_model(model, days)

    def _analyze_single_model(self, model, days):
        """Analyze performance of a single model."""
        self.stdout.write(f'\n🔍 Analyzing {model.version}')
        self.stdout.write('-' * (12 + len(model.version)))

        # Basic metrics
        self.stdout.write(f'Training accuracy: {model.training_accuracy if model.training_accuracy is not None else "N/A"}')
        self.stdout.write(f'Validation accuracy: {model.validation_accuracy if model.validation_accuracy is not None else "N/A"}')

        # Recent performance
        cutoff_date = timezone.now() - timedelta(days=days)
        recent_feedback = FeedbackLearning.objects.filter(
            created_at__gte=cutoff_date
        )

        if recent_feedback.exists():
            avg_agreement = recent_feedback.aggregate(avg=Avg('user_feedback_score'))['avg']
            self.stdout.write(f'\nRecent user agreement (last {days} days): {avg_agreement:.1f}/5.0')

            # Agreement distribution
            for level in ['HIGH', 'MEDIUM', 'LOW']:
                count = recent_feedback.filter(agreement_level=level).count()
                percentage = count / recent_feedback.count() * 100
                self.stdout.write(f'  {level}: {count} ({percentage:.1f}%)')

    def analyze_features(self, options):
        """Analyze feature importance and distribution."""
        self.stdout.write(self.style.SUCCESS('Feature Analysis'))
        self.stdout.write('=' * 20)

        # Feature importance is no longer persisted on MLModel; surface a clear message.
        active_model = MLModel.objects.filter(status='ACTIVE').first()
        if not active_model:
            self.stdout.write(self.style.WARNING('No active model found'))
            return

        self.stdout.write(self.style.WARNING(
            'Feature importance is not persisted in the current MLModel schema; '
            'retrain to populate (or load directly from the .pkl bundle).'
        ))
        return

        self.stdout.write('Feature importance ranking:')
        for i, (feature, importance) in enumerate(sorted_features, 1):
            bar_length = int(importance * 50)  # Scale to 50 chars max
            bar = '█' * bar_length + '░' * (50 - bar_length)
            self.stdout.write(f'{i:2d}. {feature:<30} {bar} {importance:.3f}')

        # Feature categories analysis
        self._analyze_feature_categories(sorted_features)

    def _analyze_feature_categories(self, sorted_features):
        """Analyze features by category."""
        categories = {
            'Basic Structure': ['query_length', 'token_count', 'keyword_count', 'identifier_count'],
            'Complexity': ['table_count', 'join_count', 'where_conditions', 'subquery_count'],
            'Query Type': ['is_select', 'is_insert', 'is_update', 'is_delete'],
            'Performance': ['select_star_present', 'missing_where_clause', 'cartesian_product_risk'],
            'Advanced': ['nesting_level', 'function_call_count', 'union_count']
        }

        self.stdout.write('\n📊 Feature importance by category:')

        for category, features in categories.items():
            category_importance = sum(
                importance for feature, importance in sorted_features
                if feature in features
            )
            self.stdout.write(f'{category}: {category_importance:.3f}')

    def analyze_user_satisfaction(self, options):
        """Analyze user satisfaction metrics."""
        self.stdout.write(self.style.SUCCESS('User Satisfaction Analysis'))
        self.stdout.write('=' * 35)

        # Overall satisfaction
        overall_satisfaction = QueryFeedback.objects.aggregate(
            avg=Avg('score_agreement')
        )['avg'] or 0

        self.stdout.write(f'Overall satisfaction: {overall_satisfaction:.2f}/5.0')

        # Satisfaction distribution
        self.stdout.write('\nSatisfaction score distribution:')
        for score in range(1, 6):
            count = QueryFeedback.objects.filter(score_agreement=score).count()
            total = QueryFeedback.objects.count()
            percentage = (count / total * 100) if total > 0 else 0
            self.stdout.write(f'  Score {score}: {count} ({percentage:.1f}%)')

        # Satisfaction by query type
        if options['by_query_type']:
            self.stdout.write('\nSatisfaction by query type:')
            query_types = Query.objects.values_list('query_type', flat=True).distinct()

            for query_type in query_types:
                avg_satisfaction = QueryFeedback.objects.filter(
                    query__query_type=query_type
                ).aggregate(avg=Avg('score_agreement'))['avg']

                if avg_satisfaction:
                    self.stdout.write(f'  {query_type}: {avg_satisfaction:.2f}/5.0')

        # User engagement analysis
        self._analyze_user_engagement()

    def _analyze_user_engagement(self):
        """Analyze user engagement patterns."""
        self.stdout.write('\n👥 User engagement:')

        # Active users
        active_users = User.objects.filter(
            queryfeedback__created_at__gte=timezone.now() - timedelta(days=30)
        ).distinct().count()

        total_users = User.objects.filter(queryfeedback__isnull=False).distinct().count()

        self.stdout.write(f'Active users (last 30 days): {active_users}/{total_users}')

        # Top contributors
        top_contributors = User.objects.filter(
            queryfeedback__isnull=False
        ).annotate(
            feedback_count=Count('queryfeedback')
        ).order_by('-feedback_count')[:5]

        self.stdout.write('\nTop feedback contributors:')
        for user in top_contributors:
            self.stdout.write(f'  {user.username}: {user.feedback_count} items')

    def export_metrics(self, options):
        """Export metrics to file."""
        output_file = options['output']
        format_type = options['format']
        days = options['days']

        cutoff_date = timezone.now() - timedelta(days=days)

        # Collect metrics
        metrics = {
            'export_date': timezone.now().isoformat(),
            'period_days': days,
            'system_overview': self._get_system_metrics(),
            'model_performance': self._get_model_metrics(),
            'user_satisfaction': self._get_satisfaction_metrics(cutoff_date),
            'activity_metrics': self._get_activity_metrics(cutoff_date)
        }

        # Export to file
        if format_type == 'json':
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        elif format_type == 'csv':
            self._export_metrics_csv(metrics, output_file)

        self.stdout.write(
            self.style.SUCCESS(f'Metrics exported to {output_file}')
        )

    def _get_system_metrics(self):
        """Get system overview metrics."""
        return {
            'total_queries': Query.objects.count(),
            'total_feedback': QueryFeedback.objects.count(),
            'total_training_data': TrainingData.objects.count(),
            'total_models': MLModel.objects.count(),
            'active_models': MLModel.objects.filter(status='ACTIVE').count()
        }

    def _get_model_metrics(self):
        """Get model performance metrics."""
        models = []
        for model in MLModel.objects.all():
            models.append({
                'version': model.version,
                'type': model.model_type,
                'status': model.status,
                'training_accuracy': model.training_accuracy,
                'validation_accuracy': model.validation_accuracy,
                'created_at': model.created_at.isoformat()
            })
        return models

    def _get_satisfaction_metrics(self, cutoff_date):
        """Get user satisfaction metrics."""
        return {
            'overall_average': QueryFeedback.objects.aggregate(avg=Avg('score_agreement'))['avg'],
            'recent_average': QueryFeedback.objects.filter(
                created_at__gte=cutoff_date
            ).aggregate(avg=Avg('score_agreement'))['avg'],
            'score_distribution': {
                str(score): QueryFeedback.objects.filter(score_agreement=score).count()
                for score in range(1, 6)
            }
        }

    def _get_activity_metrics(self, cutoff_date):
        """Get activity metrics."""
        return {
            'recent_queries': Query.objects.filter(created_at__gte=cutoff_date).count(),
            'recent_feedback': QueryFeedback.objects.filter(created_at__gte=cutoff_date).count(),
            'active_users': User.objects.filter(
                queryfeedback__created_at__gte=cutoff_date
            ).distinct().count()
        }

    def compare_models(self, options):
        """Compare model versions."""
        versions = options['versions']

        self.stdout.write(self.style.SUCCESS('Model Comparison'))
        self.stdout.write('=' * 20)

        models = []
        for version in versions:
            try:
                model = MLModel.objects.get(version=version)
                models.append(model)
            except MLModel.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'Model not found: {version}'))

        if len(models) < 2:
            raise CommandError('Need at least 2 models to compare')

        # Compare metrics
        self.stdout.write(f'{"Metric":<25} {" | ".join(f"{m.version[:15]:<15}" for m in models)}')
        self.stdout.write('-' * (25 + len(models) * 18))

        metrics_to_compare = ['training_accuracy', 'validation_accuracy']

        for metric in metrics_to_compare:
            values = []
            for model in models:
                value = getattr(model, metric, None)
                if isinstance(value, float):
                    values.append(f'{value:.3f}')
                else:
                    values.append(str(value) if value is not None else 'N/A')

            self.stdout.write(f'{metric:<25} {" | ".join(f"{v:<15}" for v in values)}')

    def print_help(self):
        """Print help message."""
        self.stdout.write('Available analytics actions:')
        self.stdout.write('  dashboard         - Show ML system dashboard')
        self.stdout.write('  model-performance - Analyze model performance')
        self.stdout.write('  feature-analysis  - Analyze feature importance')
        self.stdout.write('  user-satisfaction - Analyze user satisfaction')
        self.stdout.write('  export-metrics    - Export metrics to file')
        self.stdout.write('  model-comparison  - Compare model versions')
        self.stdout.write('')
        self.stdout.write('Use --help with any action for more details.')