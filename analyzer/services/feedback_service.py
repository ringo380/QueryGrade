"""
Feedback Service

Handles all business logic related to user feedback collection including:
- Feedback creation and updates
- Quick thumbs up/down feedback
- Feedback learning record generation
- User reliability scoring
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from django.contrib.auth.models import User
from django.db import models

from ..models import FeedbackLearning, QueryFeedback, UserQueryHistory

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSubmission:
    """DTO for feedback submission"""

    user: User
    analysis_id: int
    accuracy_rating: Optional[int] = None
    usefulness_rating: Optional[int] = None
    clarity_rating: Optional[int] = None
    suggestions: str = ""
    would_recommend: Optional[bool] = None
    is_helpful: Optional[bool] = None  # For quick feedback


@dataclass
class FeedbackResult:
    """DTO for feedback operation results"""

    success: bool
    feedback: Optional[QueryFeedback] = None
    user_history: Optional[UserQueryHistory] = None
    created: bool = False
    error: Optional[str] = None


class FeedbackService:
    """
    Service for handling user feedback operations.

    This service encapsulates the business logic for:
    - Creating and updating feedback
    - Quick thumbs up/down feedback
    - Generating learning records for ML training
    - Calculating user reliability scores
    """

    def submit_detailed_feedback(
        self, submission: FeedbackSubmission
    ) -> FeedbackResult:
        """
        Submit or update detailed feedback for a query analysis.

        Args:
            submission: FeedbackSubmission containing feedback details

        Returns:
            FeedbackResult containing operation results
        """
        try:
            # Get user history for this analysis
            user_history = self._get_user_history(
                submission.user, submission.analysis_id
            )

            if not user_history:
                return FeedbackResult(
                    success=False,
                    error="Analysis not found or you don't have permission to provide feedback.",
                )

            # Check if feedback already exists
            existing_feedback = QueryFeedback.objects.filter(
                user_history=user_history
            ).first()

            if existing_feedback:
                # Update existing feedback
                feedback = self._update_feedback(existing_feedback, submission)
                created = False
            else:
                # Create new feedback
                feedback = self._create_feedback(user_history, submission)
                created = True

            # Update user history flags
            user_history.was_helpful = True
            user_history.feedback_comments = submission.suggestions
            user_history.save()

            # Create or update learning record
            self._create_learning_record(user_history, feedback)

            return FeedbackResult(
                success=True,
                feedback=feedback,
                user_history=user_history,
                created=created,
            )

        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return FeedbackResult(success=False, error=str(e))

    def submit_quick_feedback(self, submission: FeedbackSubmission) -> FeedbackResult:
        """
        Submit quick thumbs up/down feedback.

        Args:
            submission: FeedbackSubmission with is_helpful flag

        Returns:
            FeedbackResult containing operation results
        """
        try:
            # Get user history for this analysis
            user_history = self._get_user_history(
                submission.user, submission.analysis_id
            )

            if not user_history:
                return FeedbackResult(success=False, error="Analysis not found.")

            # Update user history with quick feedback
            user_history.was_helpful = submission.is_helpful
            user_history.save()

            # Get or create feedback record
            feedback, created = QueryFeedback.objects.get_or_create(
                user_history=user_history,
                defaults={
                    "accuracy_rating": 3 if submission.is_helpful else 1,
                    "usefulness_rating": 3 if submission.is_helpful else 1,
                    "clarity_rating": 3,
                    "would_recommend": submission.is_helpful,
                },
            )

            if not created:
                # Update existing feedback
                feedback.accuracy_rating = 5 if submission.is_helpful else 1
                feedback.usefulness_rating = 5 if submission.is_helpful else 1
                feedback.would_recommend = submission.is_helpful
                feedback.save()

            # Create learning record
            self._create_learning_record(user_history, feedback)

            return FeedbackResult(
                success=True,
                feedback=feedback,
                user_history=user_history,
                created=created,
            )

        except Exception as e:
            logger.error(f"Error submitting quick feedback: {e}")
            return FeedbackResult(success=False, error=str(e))

    def get_feedback_for_analysis(
        self, user: User, analysis_id: int
    ) -> Optional[QueryFeedback]:
        """
        Get existing feedback for an analysis.

        Args:
            user: User who submitted the feedback
            analysis_id: ID of the analysis

        Returns:
            QueryFeedback object or None
        """
        try:
            user_history = self._get_user_history(user, analysis_id)
            if not user_history:
                return None

            return QueryFeedback.objects.filter(user_history=user_history).first()

        except Exception as e:
            logger.error(f"Error retrieving feedback: {e}")
            return None

    def get_feedback_statistics(self, user: Optional[User] = None) -> Dict[str, Any]:
        """
        Get feedback statistics.

        Args:
            user: Optional user to filter by

        Returns:
            Dictionary containing feedback statistics
        """
        from django.db.models import Avg, Count

        queryset = QueryFeedback.objects.all()
        if user:
            queryset = queryset.filter(user_history__user=user)

        stats = queryset.aggregate(
            total_count=Count("id"),
            avg_accuracy=Avg("accuracy_rating"),
            avg_usefulness=Avg("usefulness_rating"),
            avg_clarity=Avg("clarity_rating"),
            recommend_count=Count("id", filter=models.Q(would_recommend=True)),
        )

        return {
            "total_feedback": stats["total_count"] or 0,
            "average_accuracy": round(stats["avg_accuracy"] or 0, 2),
            "average_usefulness": round(stats["avg_usefulness"] or 0, 2),
            "average_clarity": round(stats["avg_clarity"] or 0, 2),
            "recommendation_rate": (
                (stats["recommend_count"] / stats["total_count"] * 100)
                if stats["total_count"] > 0
                else 0
            ),
        }

    def _get_user_history(
        self, user: User, analysis_id: int
    ) -> Optional[UserQueryHistory]:
        """Get user history for the given analysis."""
        try:
            return UserQueryHistory.objects.get(
                query__analysis__id=analysis_id, user=user
            )
        except UserQueryHistory.DoesNotExist:
            logger.warning(
                f"User {user.username} attempted to access non-existent "
                f"analysis {analysis_id}"
            )
            return None

    def _create_feedback(
        self, user_history: UserQueryHistory, submission: FeedbackSubmission
    ) -> QueryFeedback:
        """Create new feedback record."""
        feedback = QueryFeedback.objects.create(
            user_history=user_history,
            accuracy_rating=submission.accuracy_rating or 3,
            usefulness_rating=submission.usefulness_rating or 3,
            clarity_rating=submission.clarity_rating or 3,
            suggestions=submission.suggestions,
            would_recommend=submission.would_recommend,
        )

        logger.info(
            f"User {submission.user.username} created feedback for "
            f"analysis {submission.analysis_id}"
        )

        return feedback

    def _update_feedback(
        self, feedback: QueryFeedback, submission: FeedbackSubmission
    ) -> QueryFeedback:
        """Update existing feedback record."""
        feedback.accuracy_rating = (
            submission.accuracy_rating or feedback.accuracy_rating
        )
        feedback.usefulness_rating = (
            submission.usefulness_rating or feedback.usefulness_rating
        )
        feedback.clarity_rating = submission.clarity_rating or feedback.clarity_rating
        feedback.suggestions = submission.suggestions
        feedback.would_recommend = submission.would_recommend
        feedback.save()

        logger.info(
            f"User {submission.user.username} updated feedback for "
            f"analysis {submission.analysis_id}"
        )

        return feedback

    def _create_learning_record(
        self, user_history: UserQueryHistory, feedback: QueryFeedback
    ) -> None:
        """Create or update learning record for ML training."""
        try:
            # Get the query analysis to extract original grade and score
            try:
                analysis = user_history.query.analysis
                original_grade = analysis.grade
                original_score = analysis.score
            except Exception:
                logger.warning(
                    f"Could not retrieve analysis for query {user_history.query.id}"
                )
                return

            # Calculate aggregated feedback score from ratings (convert 1-5 scale to 0-100 scale)
            avg_rating = (
                feedback.accuracy_rating
                + feedback.usefulness_rating
                + feedback.clarity_rating
            ) / 3

            # Convert 1-5 rating scale to 0-100 score scale
            # 1 -> 0, 2 -> 25, 3 -> 50, 4 -> 75, 5 -> 100
            feedback_score = (avg_rating - 1) * 25

            # Calculate the grade difference (user feedback vs system analysis)
            grade_difference = feedback_score - original_score

            # Get or create learning record
            learning_record, created = FeedbackLearning.objects.get_or_create(
                user_history=user_history,
                defaults={
                    "original_grade": original_grade,
                    "original_score": original_score,
                    "feedback_grade_equivalent": feedback_score,
                    "grade_difference": grade_difference,
                    "user_reliability_score": self._calculate_user_reliability(
                        user_history.user
                    ),
                },
            )

            if not created:
                learning_record.original_grade = original_grade
                learning_record.original_score = original_score
                learning_record.feedback_grade_equivalent = feedback_score
                learning_record.grade_difference = grade_difference
                learning_record.user_reliability_score = (
                    self._calculate_user_reliability(user_history.user)
                )
                learning_record.save()

            logger.info(
                f"Created/updated learning record for user {user_history.user.username}: "
                f"Original: {original_grade}/{original_score}, Feedback: {feedback_score}, Diff: {grade_difference}"
            )

        except Exception as e:
            logger.warning(f"Failed to create learning record: {e}")

    def _calculate_user_reliability(self, user: User) -> float:
        """
        Calculate user reliability score based on feedback history.

        Args:
            user: User to calculate reliability for

        Returns:
            Reliability score between 0 and 1
        """
        from django.db.models import Avg, Count

        feedback_stats = QueryFeedback.objects.filter(
            user_history__user=user
        ).aggregate(
            total_count=Count("id"),
            avg_variance=Avg("accuracy_rating") - Avg("usefulness_rating"),
        )

        total = feedback_stats["total_count"] or 0
        variance = abs(feedback_stats["avg_variance"] or 0)

        # More feedback = higher reliability, less variance = higher reliability
        if total == 0:
            return 0.5  # Neutral for new users

        # Reliability increases with feedback count (up to 20 submissions)
        count_factor = min(total / 20, 1.0)

        # Reliability decreases with high variance (inconsistent ratings)
        consistency_factor = max(1.0 - (variance / 5.0), 0.0)

        return (count_factor + consistency_factor) / 2
