"""
Feedback Collection Pipeline for QueryGrade ML System

This module handles the collection, aggregation, and preprocessing of user feedback
to create training data for machine learning models.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone

from ...models import (
    FeedbackLearning,
    MLModel,
    Query,
    QueryAnalysis,
    QueryFeedback,
    TrainingData,
    UserQueryHistory,
)

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects and processes user feedback for machine learning training."""

    def __init__(self):
        self.feedback_weight_threshold = 0.1  # Minimum weight for feedback to be useful
        self.min_feedback_count = (
            3  # Minimum feedback instances before creating training data
        )
        self.user_reliability_decay = 0.9  # Decay factor for user reliability over time

    def collect_feedback_for_query(self, query_id: int) -> Optional[TrainingData]:
        """
        Collect and aggregate all feedback for a specific query.

        Args:
            query_id: ID of the query to collect feedback for

        Returns:
            TrainingData object if sufficient feedback exists, None otherwise
        """
        try:
            query = Query.objects.get(id=query_id)
            analysis = query.analysis

            # Get all user histories for this query
            user_histories = UserQueryHistory.objects.filter(query=query)

            if not user_histories.exists():
                logger.debug(f"No user histories found for query {query_id}")
                return None

            # Collect feedback from all users
            feedback_data = []
            for history in user_histories:
                feedback = self._extract_feedback_from_history(history)
                if feedback:
                    feedback_data.append(feedback)

            if len(feedback_data) < self.min_feedback_count:
                logger.debug(
                    f"Insufficient feedback for query {query_id}: {len(feedback_data)} < {self.min_feedback_count}"
                )
                return None

            # Aggregate feedback
            aggregated = self._aggregate_feedback(feedback_data)

            # Create or update training data
            training_data = self._create_training_data(query, analysis, aggregated)

            # Create individual learning records
            self._create_learning_records(user_histories, analysis, aggregated)

            logger.info(f"Successfully collected feedback for query {query_id}")
            return training_data

        except Query.DoesNotExist:
            logger.error(f"Query {query_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error collecting feedback for query {query_id}: {str(e)}")
            return None

    def _extract_feedback_from_history(
        self, history: UserQueryHistory
    ) -> Optional[Dict]:
        """Extract feedback data from a user history record."""
        try:
            # Check for detailed feedback
            if hasattr(history, "detailed_feedback"):
                detailed = history.detailed_feedback
                user_grade = self._convert_ratings_to_grade(
                    detailed.accuracy_rating,
                    detailed.usefulness_rating,
                    detailed.clarity_rating,
                )
                confidence = self._calculate_feedback_confidence(detailed)
                return {
                    "user_grade": user_grade,
                    "confidence": confidence,
                    "accuracy_rating": detailed.accuracy_rating,
                    "usefulness_rating": detailed.usefulness_rating,
                    "clarity_rating": detailed.clarity_rating,
                    "user": history.user,
                    "timestamp": detailed.created_at,
                    "would_recommend": detailed.would_recommend,
                }

            # Check for simple thumbs up/down feedback
            elif history.was_helpful is not None:
                user_grade = (
                    4.0 if history.was_helpful else 2.0
                )  # Convert boolean to 1-5 scale
                confidence = 0.7  # Lower confidence for simple feedback
                return {
                    "user_grade": user_grade,
                    "confidence": confidence,
                    "accuracy_rating": None,
                    "usefulness_rating": user_grade,
                    "clarity_rating": None,
                    "user": history.user,
                    "timestamp": history.submitted_at,
                    "would_recommend": history.was_helpful,
                }

            return None

        except Exception as e:
            logger.warning(
                f"Error extracting feedback from history {history.id}: {str(e)}"
            )
            return None

    def _convert_ratings_to_grade(
        self, accuracy: Optional[int], usefulness: Optional[int], clarity: Optional[int]
    ) -> float:
        """Convert 1-5 ratings to a single grade value."""
        ratings = [r for r in [accuracy, usefulness, clarity] if r is not None]
        if not ratings:
            return 3.0  # Default neutral grade

        # Weight usefulness highest, then accuracy, then clarity
        weights = []
        values = []

        if usefulness is not None:
            weights.append(0.5)
            values.append(usefulness)
        if accuracy is not None:
            weights.append(0.3)
            values.append(accuracy)
        if clarity is not None:
            weights.append(0.2)
            values.append(clarity)

        # Calculate weighted average
        weighted_sum = sum(w * v for w, v in zip(weights, values))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 3.0

    def _calculate_feedback_confidence(self, feedback: QueryFeedback) -> float:
        """Calculate confidence score for feedback based on completeness and consistency."""
        confidence = 0.5  # Base confidence

        # Higher confidence for complete feedback
        rating_count = sum(
            1
            for r in [
                feedback.accuracy_rating,
                feedback.usefulness_rating,
                feedback.clarity_rating,
            ]
            if r is not None
        )
        confidence += (rating_count / 3) * 0.3

        # Higher confidence for consistent ratings
        ratings = [
            r
            for r in [
                feedback.accuracy_rating,
                feedback.usefulness_rating,
                feedback.clarity_rating,
            ]
            if r is not None
        ]
        if len(ratings) > 1:
            rating_variance = sum(
                (r - sum(ratings) / len(ratings)) ** 2 for r in ratings
            ) / len(ratings)
            consistency_bonus = (
                max(0, (2.0 - rating_variance) / 2.0) * 0.2
            )  # Lower variance = higher consistency
            confidence += consistency_bonus

        # Bonus for written suggestions
        if feedback.suggestions and len(feedback.suggestions.strip()) > 10:
            confidence += 0.1

        return min(1.0, confidence)

    def _aggregate_feedback(self, feedback_data: List[Dict]) -> Dict:
        """Aggregate multiple feedback instances into summary statistics."""
        if not feedback_data:
            return {}

        # Calculate user reliability weights
        weighted_grades = []
        total_weight = 0

        accuracy_ratings = []
        usefulness_ratings = []
        clarity_ratings = []

        for feedback in feedback_data:
            user_reliability = self._get_user_reliability(feedback["user"])
            feedback_confidence = feedback["confidence"]

            # Combined weight from user reliability and feedback confidence
            weight = user_reliability * feedback_confidence

            if weight >= self.feedback_weight_threshold:
                weighted_grades.append(feedback["user_grade"] * weight)
                total_weight += weight

                # Collect individual ratings
                if feedback["accuracy_rating"]:
                    accuracy_ratings.append(feedback["accuracy_rating"])
                if feedback["usefulness_rating"]:
                    usefulness_ratings.append(feedback["usefulness_rating"])
                if feedback["clarity_rating"]:
                    clarity_ratings.append(feedback["clarity_rating"])

        if total_weight == 0:
            logger.warning("No feedback meets minimum weight threshold")
            return {}

        # Calculate weighted average grade
        avg_grade = sum(weighted_grades) / total_weight

        # Calculate standard deviation
        variance = (
            sum(
                ((feedback["user_grade"] - avg_grade) ** 2)
                * (
                    self._get_user_reliability(feedback["user"])
                    * feedback["confidence"]
                )
                for feedback in feedback_data
            )
            / total_weight
        )
        stddev = variance**0.5

        return {
            "user_grade_avg": avg_grade,
            "user_grade_count": len(feedback_data),
            "user_grade_stddev": stddev,
            "accuracy_rating_avg": (
                sum(accuracy_ratings) / len(accuracy_ratings)
                if accuracy_ratings
                else None
            ),
            "usefulness_rating_avg": (
                sum(usefulness_ratings) / len(usefulness_ratings)
                if usefulness_ratings
                else None
            ),
            "clarity_rating_avg": (
                sum(clarity_ratings) / len(clarity_ratings) if clarity_ratings else None
            ),
            "total_weight": total_weight,
        }

    def _get_user_reliability(self, user: User) -> float:
        """Calculate reliability score for a user based on their feedback history."""
        # This is a simplified implementation - in production you'd want more sophisticated scoring
        try:
            # Count user's feedback instances
            feedback_count = QueryFeedback.objects.filter(
                user_history__user=user
            ).count()

            # Users with more feedback are generally more reliable (up to a point)
            base_reliability = min(0.9, 0.5 + (feedback_count * 0.05))

            # Could add factors like:
            # - Agreement with expert ratings
            # - Consistency over time
            # - Account age
            # - Verification status

            return base_reliability

        except Exception:
            return 0.5  # Default reliability for new/unknown users

    @transaction.atomic
    def _create_training_data(
        self, query: Query, analysis: QueryAnalysis, aggregated: Dict
    ) -> TrainingData:
        """Create or update training data record."""
        try:
            training_data, created = TrainingData.objects.get_or_create(
                query=query,
                defaults={
                    "user_grade_avg": aggregated["user_grade_avg"],
                    "user_grade_count": aggregated["user_grade_count"],
                    "user_grade_stddev": aggregated["user_grade_stddev"],
                    "system_grade": analysis.grade,
                    "system_score": analysis.score,
                    "accuracy_rating_avg": aggregated.get("accuracy_rating_avg"),
                    "usefulness_rating_avg": aggregated.get("usefulness_rating_avg"),
                    "clarity_rating_avg": aggregated.get("clarity_rating_avg"),
                    "query_complexity": query.estimated_complexity,
                    "table_count": query.table_count,
                    "join_count": query.join_count,
                },
            )

            if not created:
                # Update existing record with new data
                training_data.user_grade_avg = aggregated["user_grade_avg"]
                training_data.user_grade_count = aggregated["user_grade_count"]
                training_data.user_grade_stddev = aggregated["user_grade_stddev"]
                training_data.accuracy_rating_avg = aggregated.get(
                    "accuracy_rating_avg"
                )
                training_data.usefulness_rating_avg = aggregated.get(
                    "usefulness_rating_avg"
                )
                training_data.clarity_rating_avg = aggregated.get("clarity_rating_avg")
                training_data.save()

            logger.info(
                f"{'Created' if created else 'Updated'} training data for query {query.id}"
            )
            return training_data

        except Exception as e:
            logger.error(f"Error creating training data for query {query.id}: {str(e)}")
            raise

    def _create_learning_records(
        self,
        user_histories: List[UserQueryHistory],
        analysis: QueryAnalysis,
        aggregated: Dict,
    ):
        """Create individual learning records for each feedback instance."""
        for history in user_histories:
            try:
                feedback_data = self._extract_feedback_from_history(history)
                if not feedback_data:
                    continue

                # Convert user grade (1-5) to system score scale (0-100)
                feedback_score = (feedback_data["user_grade"] - 1) * 25  # 1->0, 5->100
                grade_difference = feedback_score - analysis.score

                # Calculate feedback weight
                user_reliability = self._get_user_reliability(feedback_data["user"])
                feedback_weight = user_reliability * feedback_data["confidence"]

                FeedbackLearning.objects.update_or_create(
                    user_history=history,
                    defaults={
                        "original_grade": analysis.grade,
                        "original_score": analysis.score,
                        "original_confidence": 0.8,  # Default system confidence
                        "feedback_grade_equivalent": feedback_score,
                        "grade_difference": grade_difference,
                        "feedback_weight": feedback_weight,
                        "user_reliability_score": user_reliability,
                        "context_similarity_score": 0.0,  # To be calculated later
                    },
                )

            except Exception as e:
                logger.warning(
                    f"Error creating learning record for history {history.id}: {str(e)}"
                )

    def batch_collect_feedback(self, days_back: int = 7) -> List[TrainingData]:
        """Collect feedback for all queries with recent activity."""
        cutoff_date = timezone.now() - timedelta(days=days_back)

        # Find queries with recent feedback
        recent_queries = Query.objects.filter(
            userqueryhistory__submitted_at__gte=cutoff_date
        ).distinct()

        training_data_list = []

        for query in recent_queries:
            training_data = self.collect_feedback_for_query(query.id)
            if training_data:
                training_data_list.append(training_data)

        logger.info(
            f"Collected feedback for {len(training_data_list)} queries from last {days_back} days"
        )
        return training_data_list

    def get_training_dataset(
        self, min_feedback_count: int = 3, include_validated_only: bool = False
    ) -> List[TrainingData]:
        """Get all training data suitable for model training."""
        queryset = TrainingData.objects.filter(user_grade_count__gte=min_feedback_count)

        if include_validated_only:
            queryset = queryset.filter(is_validated=True)

        return list(queryset.order_by("-updated_at"))
