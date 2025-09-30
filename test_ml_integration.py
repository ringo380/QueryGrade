#!/usr/bin/env python
"""
Simple test script to validate ML integration works without Django admin issues
"""

import os
import sys
import django
from django.conf import settings

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'querygrade.settings')
django.setup()

# Now we can import Django components
from analyzer.ml.unified_query_analyzer import UnifiedQueryAnalyzer, AnalysisRequest
import asyncio
import json

async def test_ml_integration():
    """Test the ML integration with a simple query"""
    print("Testing ML Integration...")

    # Create test query
    test_query = """
    SELECT u.id, u.username, p.title, COUNT(c.id) as comment_count
    FROM users u
    LEFT JOIN posts p ON u.id = p.user_id
    LEFT JOIN comments c ON p.id = c.post_id
    WHERE u.created_at > '2023-01-01'
    GROUP BY u.id, u.username, p.title
    ORDER BY comment_count DESC
    LIMIT 10
    """

    # Create analysis request
    request = AnalysisRequest(
        query=test_query,
        user_id="test_user",
        context={
            "use_case": "User engagement analysis",
            "expected_rows": 1000,
            "database_type": "mysql",
            "database_version": "8.0"
        },
        analysis_level="comprehensive",
        personalize=True,
        include_rewrite=True
    )

    # Initialize unified analyzer
    analyzer = UnifiedQueryAnalyzer()

    try:
        # Run analysis
        print("Running unified ML analysis...")
        result = await analyzer.analyze_query(request)

        # Print results
        print("\n=== ML Analysis Results ===")
        print(f"Request ID: {result.request_id}")
        print(f"Overall Grade: {result.overall_grade}")
        print(f"Overall Score: {result.overall_score:.1f}/100")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Analysis Time: {result.analysis_time_ms:.1f}ms")
        print(f"Components Used: {', '.join(result.components_used)}")

        print("\n=== Semantic Analysis ===")
        if result.semantic_analysis:
            print(json.dumps(result.semantic_analysis, indent=2, default=str))

        print("\n=== Pattern Analysis ===")
        if result.pattern_analysis:
            print(json.dumps(result.pattern_analysis, indent=2, default=str))

        print("\n=== Anti-Pattern Analysis ===")
        if result.anti_pattern_analysis:
            print(json.dumps(result.anti_pattern_analysis, indent=2, default=str))

        print("\n=== Performance Prediction ===")
        if result.performance_prediction:
            print(json.dumps(result.performance_prediction, indent=2, default=str))

        print("\n=== Natural Language Feedback ===")
        if result.natural_language_feedback:
            print(json.dumps(result.natural_language_feedback, indent=2, default=str))

        print("\n=== Personalized Recommendations ===")
        if result.personalized_recommendations:
            print(json.dumps(result.personalized_recommendations, indent=2, default=str))

        print("\n=== Query Rewrite ===")
        if result.query_rewrite:
            print(json.dumps(result.query_rewrite, indent=2, default=str))

        if result.warnings:
            print(f"\n=== Warnings ===")
            for warning in result.warnings:
                print(f"⚠️  {warning}")

        print("\n=== Performance Metrics ===")
        print(f"Component execution times:")
        for component, time_ms in result.execution_time_by_component.items():
            print(f"  - {component}: {time_ms:.1f}ms")

        print("\n✅ ML Integration test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ ML Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("QueryGrade ML Integration Test")
    print("=" * 50)

    # Run the async test
    success = asyncio.run(test_ml_integration())

    if success:
        print("\n🎉 All tests passed! ML integration is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed! Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()