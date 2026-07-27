# Contributing to QueryGrade

Thank you for considering contributing to QueryGrade! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct (see CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- Clear, descriptive title
- Detailed steps to reproduce
- Expected vs. actual behavior
- SQL query that triggered the issue (if applicable)
- Environment details (OS, browser, database type)
- Screenshots if relevant

Use the bug report template when creating issues.

### Suggesting Features

Feature suggestions are welcome! Please:

- Use the feature request template
- Explain the problem you're trying to solve
- Describe your proposed solution
- Include example use cases
- Indicate which area of the system it affects

### ML Model Improvements

If you're reporting ML model inaccuracies or suggesting improvements:

- Use the ML performance issue template
- Include the query and analysis results
- Explain why you disagree with the assessment
- Provide actual query performance metrics if available
- Suggest specific improvements

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Write/update tests**
5. **Run the test suite** (`python manage.py test`)
6. **Update documentation** if needed
7. **Commit your changes** with clear messages
8. **Push to your fork**
9. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/QueryGrade.git
cd QueryGrade

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run tests
python manage.py test

# Start development server
python manage.py runserver
```

## Coding Standards

### Python Style Guide

- Follow PEP 8
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Maximum line length: 100 characters

### Code Quality

- Write unit tests for new features
- Maintain or improve test coverage
- Use type hints where applicable
- Handle errors gracefully
- Add logging for important operations

### Commit Messages

Write clear, descriptive commit messages:

```
feat: Add semantic query analysis to ML pipeline

- Implement transformer-based query embeddings
- Add intent detection for SELECT/UPDATE/DELETE
- Integrate with unified analyzer
```

**Format:** `type: Brief description`

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

## Testing

### Running Tests

```bash
# All tests
python manage.py test

# Specific app
python manage.py test analyzer

# Specific test file
python manage.py test analyzer.test_query_grader

# ML integration test
python test_ml_integration.py
```

### Writing Tests

- Test both success and failure cases
- Use descriptive test names
- Mock external dependencies
- Test edge cases
- Include integration tests for complex features

## ML Development Guidelines

### Adding ML Features

1. Implement in `analyzer/ml/` directory
2. Create corresponding tests
3. Update ML documentation in README.md
4. Add performance metrics tracking
5. Validate against benchmark queries

### Model Training

- Document training data requirements
- Include validation metrics
- Use cross-validation
- Track model versions
- Compare against baseline

### Feature Engineering

- Document new features in code comments
- Validate feature importance
- Consider computational cost
- Test on diverse query types

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Use Google-style docstrings
- Include parameter types and return types
- Add usage examples for complex features

### User Documentation

- Update README.md for user-facing changes
- Add examples for new features
- Update API documentation
- Include screenshots for UI changes

## Review Process

All submissions require review. We use GitHub pull requests for this purpose:

1. **Automated checks** must pass (tests, linting)
2. **Code review** by maintainers
3. **Discussion** of design decisions if needed
4. **Approval** and merge

## Project Structure

```
QueryGrade/
├── analyzer/              # Main Django app
│   ├── ml/               # ML components
│   ├── templates/        # HTML templates
│   ├── views.py          # Web views
│   ├── models.py         # Database models
│   └── tests/            # Test files
├── querygrade/           # Django project settings
├── requirements.txt      # Python dependencies
└── docs/                # Design specs
```

## Getting Help

- Check existing issues and documentation
- Ask questions in issue comments
- Reach out to maintainers

## Recognition

Contributors will be recognized in:
- Release notes
- Contributors section
- Git commit history

Thank you for contributing to QueryGrade! 🎉
