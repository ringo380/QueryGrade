"""
Django management command to train ML models for QueryGrade.

Usage:
    python manage.py train_ml_model
    python manage.py train_ml_model --force
    python manage.py train_ml_model --algorithm random_forest
    python manage.py train_ml_model --config /path/to/config.json
"""

import json
import os
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from analyzer.ml.core.training_pipeline import TrainingPipelineManager, TrainingConfig


class Command(BaseCommand):
    help = 'Train ML models for query analysis'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force retraining even if recent model exists'
        )

        parser.add_argument(
            '--algorithm',
            type=str,
            choices=['random_forest', 'gradient_boosting'],
            default='random_forest',
            help='ML algorithm to use for training'
        )

        parser.add_argument(
            '--model-type',
            type=str,
            default='QUERY_GRADER',
            help='Type of model to train'
        )

        parser.add_argument(
            '--min-samples',
            type=int,
            default=50,
            help='Minimum number of training samples required'
        )

        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Proportion of data to use for testing (0.0-1.0)'
        )

        parser.add_argument(
            '--config',
            type=str,
            help='Path to JSON configuration file'
        )

        parser.add_argument(
            '--deploy',
            action='store_true',
            help='Automatically deploy model if it meets performance threshold'
        )

        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without actually training'
        )

        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )

    def handle(self, *args, **options):
        """Handle the training command."""
        if options['verbose']:
            self.stdout.write(self.style.SUCCESS('Starting ML model training...'))

        try:
            # Load configuration
            config = self._load_config(options)

            # Create pipeline manager
            pipeline_manager = TrainingPipelineManager(config)

            if options['dry_run']:
                self._show_dry_run_info(pipeline_manager, options)
                return

            # Check training status
            status = pipeline_manager.get_training_status()
            self._display_training_status(status)

            if not status['training_data']['ready_for_training']:
                raise CommandError(
                    f"Insufficient training data: {status['training_data']['total_samples']} samples "
                    f"(minimum required: {config.min_training_samples})"
                )

            # Run training pipeline
            self.stdout.write('Running training pipeline...')
            result = pipeline_manager.run_training_pipeline(force_retrain=options['force'])

            # Display results
            self._display_training_results(result, options)

            if result.success:
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Model training completed successfully! Version: {result.model_version}'
                    )
                )

                if options['deploy'] and result.validation_accuracy >= config.performance_threshold:
                    self.stdout.write('Deploying model...')
                    pipeline_manager._deploy_model(result.model_version)
                    self.stdout.write(self.style.SUCCESS('Model deployed successfully!'))

            else:
                raise CommandError(f'Training failed: {result.error_message}')

        except Exception as e:
            if options['verbose']:
                import traceback
                self.stdout.write(self.style.ERROR(traceback.format_exc()))

            raise CommandError(f'Training command failed: {str(e)}')

    def _load_config(self, options):
        """Load training configuration from options and config file."""
        # Start with default config
        config = TrainingConfig()

        # Load from config file if provided
        if options['config']:
            if not os.path.exists(options['config']):
                raise CommandError(f"Config file not found: {options['config']}")

            with open(options['config'], 'r') as f:
                config_data = json.load(f)

            # Update config with file values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Override with command line options
        if options['algorithm']:
            config.algorithm = options['algorithm']

        if options['model_type']:
            config.model_type = options['model_type']

        if options['min_samples']:
            config.min_training_samples = options['min_samples']

        if options['test_size']:
            config.test_size = options['test_size']

        if options['deploy']:
            config.auto_deployment = True

        return config

    def _show_dry_run_info(self, pipeline_manager, options):
        """Show information about what would be done in a dry run."""
        self.stdout.write(self.style.WARNING('DRY RUN MODE - No actual training will occur'))
        self.stdout.write('')

        status = pipeline_manager.get_training_status()
        self._display_training_status(status)

        self.stdout.write('Training configuration:')
        config = pipeline_manager.config
        self.stdout.write(f'  Algorithm: {config.algorithm}')
        self.stdout.write(f'  Model type: {config.model_type}')
        self.stdout.write(f'  Min samples: {config.min_training_samples}')
        self.stdout.write(f'  Test size: {config.test_size}')
        self.stdout.write(f'  Auto deployment: {config.auto_deployment}')
        self.stdout.write('')

        if status['training_data']['ready_for_training']:
            self.stdout.write(self.style.SUCCESS('✓ Ready for training'))
        else:
            self.stdout.write(self.style.ERROR('✗ Not ready for training'))

    def _display_training_status(self, status):
        """Display current training status."""
        self.stdout.write('Current training status:')

        # Latest model info
        latest = status['latest_model']
        if latest['version']:
            self.stdout.write(f'  Latest model: {latest["version"]}')
            self.stdout.write(f'  Status: {latest["status"]}')
            if latest['performance']:
                accuracy = latest['performance'].get('validation_accuracy', 'N/A')
                self.stdout.write(f'  Validation accuracy: {accuracy}')
        else:
            self.stdout.write('  No existing models found')

        # Training data info
        data_info = status['training_data']
        self.stdout.write(f'  Training samples: {data_info["total_samples"]}')
        self.stdout.write(f'  Ready for training: {data_info["ready_for_training"]}')

        # Recent activity
        activity = status['recent_activity']
        self.stdout.write(f'  Recent feedback: {activity["feedback_last_week"]} (last 7 days)')
        self.stdout.write('')

    def _display_training_results(self, result, options):
        """Display training results."""
        if not result.success:
            self.stdout.write(self.style.ERROR(f'Training failed: {result.error_message}'))
            return

        self.stdout.write(self.style.SUCCESS('Training Results:'))
        self.stdout.write(f'  Model version: {result.model_version}')
        self.stdout.write(f'  Training accuracy: {result.training_accuracy:.3f}')
        self.stdout.write(f'  Validation accuracy: {result.validation_accuracy:.3f}')
        self.stdout.write(f'  Test accuracy: {result.test_accuracy:.3f}')
        self.stdout.write(f'  Training time: {result.training_time:.2f}s')
        self.stdout.write(f'  Model saved to: {result.model_path}')

        if options['verbose'] and result.metrics:
            self.stdout.write('')
            self.stdout.write('Detailed metrics:')
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    self.stdout.write(f'  {key}: {value:.3f}')
                else:
                    self.stdout.write(f'  {key}: {value}')

        if options['verbose'] and result.feature_importance:
            self.stdout.write('')
            self.stdout.write('Top 10 most important features:')
            sorted_features = sorted(
                result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            for feature, importance in sorted_features:
                self.stdout.write(f'  {feature}: {importance:.3f}')

    def _create_sample_config(self):
        """Create a sample configuration file."""
        sample_config = {
            "model_type": "HYBRID_SCORER",
            "model_name": "query_grader",
            "algorithm": "random_forest",
            "test_size": 0.2,
            "validation_size": 0.2,
            "cross_validation_folds": 5,
            "min_training_samples": 50,
            "max_training_samples": 10000,
            "feature_scaling": True,
            "hyperparameter_tuning": True,
            "model_versioning": True,
            "auto_deployment": False,
            "performance_threshold": 0.7
        }

        config_path = os.path.join(settings.BASE_DIR, 'ml_training_config.json')
        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)

        self.stdout.write(f'Sample config created at: {config_path}')