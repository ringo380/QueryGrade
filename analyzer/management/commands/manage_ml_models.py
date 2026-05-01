"""
Django management command to manage ML models for QueryGrade.

Usage:
    python manage.py manage_ml_models list
    python manage.py manage_ml_models activate <model_version>
    python manage.py manage_ml_models deactivate <model_version>
    python manage.py manage_ml_models delete <model_version>
    python manage.py manage_ml_models status
    python manage.py manage_ml_models cleanup --keep 5
"""

import os
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from analyzer.models import MLModel, LearningMetrics, TrainingData


class Command(BaseCommand):
    help = 'Manage ML models (list, activate, deactivate, delete, cleanup)'

    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest='action', help='Available actions')

        # List command
        list_parser = subparsers.add_parser('list', help='List all ML models')
        list_parser.add_argument(
            '--active-only',
            action='store_true',
            help='Show only active models'
        )
        list_parser.add_argument(
            '--model-type',
            type=str,
            help='Filter by model type'
        )

        # Status command
        status_parser = subparsers.add_parser('status', help='Show ML system status')

        # Activate command
        activate_parser = subparsers.add_parser('activate', help='Activate a model')
        activate_parser.add_argument('model_version', help='Model version to activate')
        activate_parser.add_argument(
            '--force',
            action='store_true',
            help='Force activation even if model file is missing'
        )

        # Deactivate command
        deactivate_parser = subparsers.add_parser('deactivate', help='Deactivate a model')
        deactivate_parser.add_argument('model_version', help='Model version to deactivate')

        # Delete command
        delete_parser = subparsers.add_parser('delete', help='Delete a model')
        delete_parser.add_argument('model_version', help='Model version to delete')
        delete_parser.add_argument(
            '--confirm',
            action='store_true',
            help='Confirm deletion without interactive prompt'
        )

        # Cleanup command
        cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old models')
        cleanup_parser.add_argument(
            '--keep',
            type=int,
            default=5,
            help='Number of models to keep per type (default: 5)'
        )
        cleanup_parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting'
        )

        # Export command
        export_parser = subparsers.add_parser('export', help='Export model information')
        export_parser.add_argument('output_file', help='Output file path')
        export_parser.add_argument(
            '--format',
            choices=['json', 'csv'],
            default='json',
            help='Export format'
        )

    def handle(self, *args, **options):
        """Handle the model management command."""
        action = options.get('action')

        if not action:
            self.print_help()
            return

        try:
            if action == 'list':
                self.handle_list(options)
            elif action == 'status':
                self.handle_status(options)
            elif action == 'activate':
                self.handle_activate(options)
            elif action == 'deactivate':
                self.handle_deactivate(options)
            elif action == 'delete':
                self.handle_delete(options)
            elif action == 'cleanup':
                self.handle_cleanup(options)
            elif action == 'export':
                self.handle_export(options)
            else:
                raise CommandError(f'Unknown action: {action}')

        except Exception as e:
            raise CommandError(f'Command failed: {str(e)}')

    def handle_list(self, options):
        """List ML models."""
        queryset = MLModel.objects.all()

        if options['active_only']:
            queryset = queryset.filter(status='ACTIVE')

        if options['model_type']:
            queryset = queryset.filter(model_type=options['model_type'])

        models = queryset.order_by('-created_at')

        if not models.exists():
            self.stdout.write(self.style.WARNING('No models found'))
            return

        self.stdout.write(f'Found {models.count()} models:')
        self.stdout.write('')

        # Table header
        self.stdout.write(f'{"Version":<30} {"Type":<15} {"Status":<12} {"Accuracy":<10} {"Created":<20}')
        self.stdout.write('-' * 89)

        # Table rows
        for model in models:
            version = model.version[:28] + '...' if len(model.version) > 30 else model.version
            accuracy = f"{model.validation_accuracy:.3f}" if model.validation_accuracy is not None else 'N/A'
            created = model.created_at.strftime('%Y-%m-%d %H:%M')

            self.stdout.write(f'{version:<30} {model.model_type:<15} {model.status:<12} {accuracy:<10} {created:<20}')

    def handle_status(self, options):
        """Show ML system status."""
        self.stdout.write(self.style.SUCCESS('ML System Status'))
        self.stdout.write('=' * 40)

        # Model counts by type
        model_types = MLModel.objects.values_list('model_type', flat=True).distinct()
        for model_type in model_types:
            total = MLModel.objects.filter(model_type=model_type).count()
            active = MLModel.objects.filter(model_type=model_type, status='ACTIVE').count()
            self.stdout.write(f'{model_type}: {total} total, {active} active')

        # Training data status
        training_data_count = TrainingData.objects.count()
        self.stdout.write(f'Training data samples: {training_data_count}')

        # Recent activity
        recent_models = MLModel.objects.filter(
            created_at__gte=timezone.now() - timezone.timedelta(days=7)
        ).count()
        self.stdout.write(f'Models created last 7 days: {recent_models}')

        # Active models details
        active_models = MLModel.objects.filter(status='ACTIVE').order_by('model_type')
        if active_models.exists():
            self.stdout.write('')
            self.stdout.write('Active Models:')
            for model in active_models:
                accuracy = model.validation_accuracy if model.validation_accuracy is not None else 'N/A'
                self.stdout.write(f'  {model.model_type}: {model.version} (accuracy: {accuracy})')

        # Storage usage
        total_size = 0
        for model in MLModel.objects.all():
            if model.file_path and os.path.exists(model.file_path):
                total_size += os.path.getsize(model.file_path)

        self.stdout.write(f'Storage used: {total_size / (1024*1024):.2f} MB')

    def handle_activate(self, options):
        """Activate a model."""
        model_version = options['model_version']

        try:
            model = MLModel.objects.get(version=model_version)
        except MLModel.DoesNotExist:
            raise CommandError(f'Model not found: {model_version}')

        # Check if model file exists
        if not options['force'] and not os.path.exists(model.file_path):
            raise CommandError(f'Model file not found: {model.file_path}')

        with transaction.atomic():
            # Deactivate all other models of the same type
            MLModel.objects.filter(
                model_type=model.model_type,
                status='ACTIVE'
            ).update(status='DEPRECATED')

            # Activate the target model
            model.status = 'ACTIVE'
            model.deployed_at = timezone.now()
            model.save(update_fields=['status', 'deployed_at'])

        self.stdout.write(
            self.style.SUCCESS(f'Model {model_version} activated successfully')
        )

    def handle_deactivate(self, options):
        """Deactivate a model."""
        model_version = options['model_version']

        try:
            model = MLModel.objects.get(version=model_version)
        except MLModel.DoesNotExist:
            raise CommandError(f'Model not found: {model_version}')

        model.status = 'DEPRECATED'
        model.save(update_fields=['status'])

        self.stdout.write(
            self.style.SUCCESS(f'Model {model_version} deactivated successfully')
        )

    def handle_delete(self, options):
        """Delete a model."""
        model_version = options['model_version']

        try:
            model = MLModel.objects.get(version=model_version)
        except MLModel.DoesNotExist:
            raise CommandError(f'Model not found: {model_version}')

        if model.status == 'ACTIVE' and not options['confirm']:
            if not self._confirm_action(f'Model {model_version} is currently active. Delete anyway?'):
                self.stdout.write('Deletion cancelled')
                return

        # Confirm deletion
        if not options['confirm']:
            if not self._confirm_action(f'Delete model {model_version}? This cannot be undone.'):
                self.stdout.write('Deletion cancelled')
                return

        # Delete model file
        if model.file_path and os.path.exists(model.file_path):
            try:
                os.remove(model.file_path)
                self.stdout.write(f'Deleted model file: {model.file_path}')
            except OSError as e:
                self.stdout.write(
                    self.style.WARNING(f'Could not delete model file: {e}')
                )

        # Delete database record
        model.delete()

        self.stdout.write(
            self.style.SUCCESS(f'Model {model_version} deleted successfully')
        )

    def handle_cleanup(self, options):
        """Clean up old models."""
        keep_count = options['keep']
        dry_run = options['dry_run']

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No actual deletion'))

        model_types = MLModel.objects.values_list('model_type', flat=True).distinct()
        total_deleted = 0

        for model_type in model_types:
            # Get old models (keep the most recent ones)
            old_models = MLModel.objects.filter(
                model_type=model_type,
            ).exclude(status='ACTIVE').order_by('-created_at')[keep_count:]

            if not old_models:
                continue

            self.stdout.write(f'Cleaning up {model_type} models (keeping {keep_count} most recent)')

            for model in old_models:
                if dry_run:
                    self.stdout.write(f'  Would delete: {model.version}')
                else:
                    # Delete model file
                    if model.file_path and os.path.exists(model.file_path):
                        try:
                            os.remove(model.file_path)
                        except OSError:
                            pass

                    # Delete database record
                    model.delete()
                    self.stdout.write(f'  Deleted: {model.version}')

                total_deleted += 1

        if dry_run:
            self.stdout.write(f'Would delete {total_deleted} models')
        else:
            self.stdout.write(
                self.style.SUCCESS(f'Cleanup complete. Deleted {total_deleted} models.')
            )

    def handle_export(self, options):
        """Export model information."""
        output_file = options['output_file']
        format_type = options['format']

        models = MLModel.objects.all().order_by('-created_at')

        if format_type == 'json':
            self._export_json(models, output_file)
        elif format_type == 'csv':
            self._export_csv(models, output_file)

        self.stdout.write(
            self.style.SUCCESS(f'Exported {models.count()} models to {output_file}')
        )

    def _export_json(self, models, output_file):
        """Export models to JSON format."""
        import json

        data = []
        for model in models:
            data.append({
                'version': model.version,
                'name': model.name,
                'model_type': model.model_type,
                'status': model.status,
                'training_accuracy': model.training_accuracy,
                'validation_accuracy': model.validation_accuracy,
                'created_at': model.created_at.isoformat(),
                'file_size_bytes': model.file_size_bytes,
                'training_samples': model.training_samples,
            })

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_csv(self, models, output_file):
        """Export models to CSV format."""
        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Version', 'Name', 'Type', 'Status', 'Validation Accuracy',
                'Training Accuracy', 'Created', 'File Size', 'Training Samples'
            ])

            # Data rows
            for model in models:
                writer.writerow([
                    model.version,
                    model.name,
                    model.model_type,
                    model.status,
                    model.validation_accuracy if model.validation_accuracy is not None else '',
                    model.training_accuracy if model.training_accuracy is not None else '',
                    model.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    model.file_size_bytes or '',
                    model.training_samples or ''
                ])

    def _confirm_action(self, message):
        """Ask for user confirmation."""
        try:
            response = input(f'{message} (y/N): ')
            return response.lower().startswith('y')
        except (EOFError, KeyboardInterrupt):
            return False

    def print_help(self):
        """Print help message."""
        self.stdout.write('Available actions:')
        self.stdout.write('  list      - List all ML models')
        self.stdout.write('  status    - Show ML system status')
        self.stdout.write('  activate  - Activate a model')
        self.stdout.write('  deactivate - Deactivate a model')
        self.stdout.write('  delete    - Delete a model')
        self.stdout.write('  cleanup   - Clean up old models')
        self.stdout.write('  export    - Export model information')
        self.stdout.write('')
        self.stdout.write('Use --help with any action for more details.')