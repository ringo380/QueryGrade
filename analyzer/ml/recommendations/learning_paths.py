"""
Learning Path Generator

This module generates personalized learning paths for SQL skill improvement
based on user performance, identified weaknesses, and learning goals.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class SkillLevel(Enum):
    """SQL skill levels"""

    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


class TopicCategory(Enum):
    """Learning topic categories"""

    FUNDAMENTALS = "fundamentals"
    JOINS = "joins"
    AGGREGATION = "aggregation"
    SUBQUERIES = "subqueries"
    OPTIMIZATION = "optimization"
    INDEXES = "indexes"
    TRANSACTIONS = "transactions"
    WINDOW_FUNCTIONS = "window_functions"
    STORED_PROCEDURES = "stored_procedures"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    BEST_PRACTICES = "best_practices"


class LearningFormat(Enum):
    """Learning content formats"""

    TUTORIAL = "tutorial"
    VIDEO = "video"
    EXERCISE = "exercise"
    QUIZ = "quiz"
    PROJECT = "project"
    DOCUMENTATION = "documentation"
    INTERACTIVE = "interactive"
    CASE_STUDY = "case_study"


@dataclass
class LearningResource:
    """Represents a learning resource"""

    resource_id: str
    title: str
    category: TopicCategory
    skill_level: SkillLevel
    format: LearningFormat
    estimated_time_minutes: int
    description: str
    objectives: List[str]
    prerequisites: List[str]
    url: Optional[str] = None
    exercises: List[Dict[str, Any]] = field(default_factory=list)
    assessment_criteria: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)


@dataclass
class LearningModule:
    """A collection of related learning resources"""

    module_id: str
    title: str
    description: str
    skill_level: SkillLevel
    resources: List[LearningResource]
    total_time_hours: float
    learning_objectives: List[str]
    completion_criteria: List[str]
    next_modules: List[str] = field(default_factory=list)


@dataclass
class PersonalizedLearningPath:
    """Personalized learning path for a user"""

    user_id: str
    current_skill_level: SkillLevel
    target_skill_level: SkillLevel
    identified_gaps: List[str]
    recommended_modules: List[LearningModule]
    estimated_completion_time: str
    milestones: List[Dict[str, Any]]
    practice_exercises: List[Dict[str, Any]]
    assessment_schedule: List[Dict[str, Any]]
    motivational_tips: List[str]
    progress_tracking: Dict[str, Any]


class LearningPathGenerator:
    """Generates personalized SQL learning paths"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.learning_resources = self._initialize_learning_resources()
        self.learning_modules = self._create_learning_modules()
        self.skill_dependencies = self._define_skill_dependencies()

    def _initialize_learning_resources(self) -> Dict[str, LearningResource]:
        """Initialize the learning resource library"""
        resources = {}

        # Fundamentals
        resources["sql_basics"] = LearningResource(
            resource_id="sql_basics",
            title="SQL Basics: SELECT, WHERE, ORDER BY",
            category=TopicCategory.FUNDAMENTALS,
            skill_level=SkillLevel.BEGINNER,
            format=LearningFormat.TUTORIAL,
            estimated_time_minutes=60,
            description="Learn fundamental SQL query structure and basic filtering",
            objectives=[
                "Understand SELECT statement syntax",
                "Filter data with WHERE clauses",
                "Sort results with ORDER BY",
            ],
            prerequisites=[],
            exercises=[
                {"type": "write_query", "difficulty": "easy", "topic": "basic_select"},
                {"type": "fix_error", "difficulty": "easy", "topic": "where_clause"},
            ],
            tags={"basics", "select", "where", "filtering"},
        )

        resources["joins_intro"] = LearningResource(
            resource_id="joins_intro",
            title="Introduction to SQL JOINs",
            category=TopicCategory.JOINS,
            skill_level=SkillLevel.BEGINNER,
            format=LearningFormat.VIDEO,
            estimated_time_minutes=45,
            description="Understanding different types of JOINs and when to use them",
            objectives=[
                "Understand INNER JOIN",
                "Learn LEFT/RIGHT JOIN",
                "Master JOIN conditions",
            ],
            prerequisites=["sql_basics"],
            tags={"joins", "relationships", "tables"},
        )

        resources["aggregation_basics"] = LearningResource(
            resource_id="aggregation_basics",
            title="Aggregation Functions and GROUP BY",
            category=TopicCategory.AGGREGATION,
            skill_level=SkillLevel.INTERMEDIATE,
            format=LearningFormat.INTERACTIVE,
            estimated_time_minutes=90,
            description="Learn to summarize data with aggregation functions",
            objectives=[
                "Use COUNT, SUM, AVG, MAX, MIN",
                "Understand GROUP BY",
                "Filter groups with HAVING",
            ],
            prerequisites=["sql_basics"],
            tags={"aggregation", "group_by", "having"},
        )

        resources["subqueries_mastery"] = LearningResource(
            resource_id="subqueries_mastery",
            title="Mastering Subqueries and CTEs",
            category=TopicCategory.SUBQUERIES,
            skill_level=SkillLevel.INTERMEDIATE,
            format=LearningFormat.TUTORIAL,
            estimated_time_minutes=120,
            description="Advanced query composition with subqueries and CTEs",
            objectives=[
                "Write scalar subqueries",
                "Use correlated subqueries",
                "Implement Common Table Expressions",
            ],
            prerequisites=["joins_intro", "aggregation_basics"],
            tags={"subqueries", "cte", "advanced"},
        )

        resources["performance_tuning"] = LearningResource(
            resource_id="performance_tuning",
            title="Query Performance Optimization",
            category=TopicCategory.OPTIMIZATION,
            skill_level=SkillLevel.ADVANCED,
            format=LearningFormat.CASE_STUDY,
            estimated_time_minutes=180,
            description="Optimize slow queries for better performance",
            objectives=[
                "Analyze execution plans",
                "Identify bottlenecks",
                "Apply optimization techniques",
            ],
            prerequisites=["subqueries_mastery"],
            tags={"performance", "optimization", "tuning"},
        )

        resources["index_strategy"] = LearningResource(
            resource_id="index_strategy",
            title="Database Indexing Strategies",
            category=TopicCategory.INDEXES,
            skill_level=SkillLevel.ADVANCED,
            format=LearningFormat.PROJECT,
            estimated_time_minutes=240,
            description="Design and implement effective indexing strategies",
            objectives=[
                "Understand B-tree and hash indexes",
                "Design covering indexes",
                "Balance read/write performance",
            ],
            prerequisites=["performance_tuning"],
            tags={"indexes", "performance", "database"},
        )

        resources["window_functions"] = LearningResource(
            resource_id="window_functions",
            title="Window Functions for Advanced Analytics",
            category=TopicCategory.WINDOW_FUNCTIONS,
            skill_level=SkillLevel.ADVANCED,
            format=LearningFormat.INTERACTIVE,
            estimated_time_minutes=150,
            description="Use window functions for complex analytical queries",
            objectives=[
                "Understand window function syntax",
                "Use ranking functions",
                "Calculate running totals and moving averages",
            ],
            prerequisites=["aggregation_basics"],
            tags={"window", "analytics", "advanced"},
        )

        resources["security_best_practices"] = LearningResource(
            resource_id="security_best_practices",
            title="SQL Security and Injection Prevention",
            category=TopicCategory.SECURITY,
            skill_level=SkillLevel.INTERMEDIATE,
            format=LearningFormat.TUTORIAL,
            estimated_time_minutes=90,
            description="Secure your SQL queries against common vulnerabilities",
            objectives=[
                "Understand SQL injection attacks",
                "Implement parameterized queries",
                "Apply principle of least privilege",
            ],
            prerequisites=["sql_basics"],
            tags={"security", "injection", "safety"},
        )

        return resources

    def _create_learning_modules(self) -> Dict[str, LearningModule]:
        """Create structured learning modules"""
        modules = {}

        # Beginner Module
        modules["beginner_foundations"] = LearningModule(
            module_id="beginner_foundations",
            title="SQL Foundations",
            description="Build a solid foundation in SQL querying",
            skill_level=SkillLevel.BEGINNER,
            resources=[self.learning_resources["sql_basics"]],
            total_time_hours=2.0,
            learning_objectives=[
                "Write basic SQL queries",
                "Filter and sort data",
                "Understand SQL syntax",
            ],
            completion_criteria=[
                "Complete all exercises",
                "Pass foundation quiz with 80%+",
            ],
            next_modules=["intermediate_skills"],
        )

        # Intermediate Module
        modules["intermediate_skills"] = LearningModule(
            module_id="intermediate_skills",
            title="Intermediate SQL Skills",
            description="Master joins, aggregations, and subqueries",
            skill_level=SkillLevel.INTERMEDIATE,
            resources=[
                self.learning_resources["joins_intro"],
                self.learning_resources["aggregation_basics"],
                self.learning_resources["subqueries_mastery"],
                self.learning_resources["security_best_practices"],
            ],
            total_time_hours=6.0,
            learning_objectives=[
                "Join multiple tables effectively",
                "Aggregate and summarize data",
                "Write complex nested queries",
                "Implement security best practices",
            ],
            completion_criteria=[
                "Complete 80% of exercises",
                "Build a multi-table query project",
                "Pass intermediate assessment",
            ],
            next_modules=["advanced_optimization"],
        )

        # Advanced Module
        modules["advanced_optimization"] = LearningModule(
            module_id="advanced_optimization",
            title="Advanced Query Optimization",
            description="Optimize performance and implement advanced patterns",
            skill_level=SkillLevel.ADVANCED,
            resources=[
                self.learning_resources["performance_tuning"],
                self.learning_resources["index_strategy"],
                self.learning_resources["window_functions"],
            ],
            total_time_hours=10.0,
            learning_objectives=[
                "Analyze and optimize query performance",
                "Design effective indexing strategies",
                "Implement advanced analytical queries",
            ],
            completion_criteria=[
                "Optimize 5 real-world slow queries",
                "Design index strategy for sample database",
                "Complete window function challenges",
            ],
            next_modules=["expert_architecture"],
        )

        return modules

    def _define_skill_dependencies(self) -> Dict[str, List[str]]:
        """Define dependencies between skills"""
        return {
            "basic_select": [],
            "filtering": ["basic_select"],
            "sorting": ["basic_select"],
            "joins": ["filtering"],
            "aggregation": ["filtering", "sorting"],
            "subqueries": ["joins", "aggregation"],
            "optimization": ["subqueries"],
            "indexes": ["optimization"],
            "window_functions": ["aggregation"],
            "stored_procedures": ["subqueries"],
            "transactions": ["basic_select"],
            "security": ["basic_select"],
        }

    def generate_learning_path(
        self, user_profile: Dict[str, Any]
    ) -> PersonalizedLearningPath:
        """Generate a personalized learning path based on user profile"""

        # Extract user information
        user_id = user_profile.get("user_id", "anonymous")
        current_skills = user_profile.get("current_skills", [])
        weak_areas = user_profile.get("weak_areas", [])
        learning_goals = user_profile.get("learning_goals", [])
        available_time_per_week = user_profile.get("hours_per_week", 5)
        preferred_formats = user_profile.get("preferred_formats", [])

        # Assess current skill level
        current_level = self._assess_skill_level(current_skills, weak_areas)

        # Determine target skill level
        target_level = self._determine_target_level(current_level, learning_goals)

        # Identify skill gaps
        skill_gaps = self._identify_skill_gaps(current_skills, weak_areas, target_level)

        # Select appropriate modules
        recommended_modules = self._select_modules(
            current_level, target_level, skill_gaps, preferred_formats
        )

        # Create milestones
        milestones = self._create_milestones(
            recommended_modules, available_time_per_week
        )

        # Generate practice exercises
        exercises = self._generate_practice_exercises(skill_gaps, current_level)

        # Create assessment schedule
        assessments = self._create_assessment_schedule(recommended_modules)

        # Generate motivational tips
        tips = self._generate_motivational_tips(current_level, target_level)

        # Calculate total time
        total_hours = sum(module.total_time_hours for module in recommended_modules)
        weeks_needed = int(total_hours / available_time_per_week) + 1

        return PersonalizedLearningPath(
            user_id=user_id,
            current_skill_level=current_level,
            target_skill_level=target_level,
            identified_gaps=skill_gaps,
            recommended_modules=recommended_modules,
            estimated_completion_time=f"{weeks_needed} weeks at {available_time_per_week} hours/week",
            milestones=milestones,
            practice_exercises=exercises,
            assessment_schedule=assessments,
            motivational_tips=tips,
            progress_tracking={
                "total_modules": len(recommended_modules),
                "completed_modules": 0,
                "total_hours": total_hours,
                "completed_hours": 0,
                "current_streak_days": 0,
                "badges_earned": [],
            },
        )

    def _assess_skill_level(
        self, current_skills: List[str], weak_areas: List[str]
    ) -> SkillLevel:
        """Assess user's current skill level"""

        skill_points = 0

        # Award points for current skills
        beginner_skills = {"basic_select", "filtering", "sorting"}
        intermediate_skills = {"joins", "aggregation", "subqueries"}
        advanced_skills = {"optimization", "indexes", "window_functions"}

        for skill in current_skills:
            if skill in beginner_skills:
                skill_points += 1
            elif skill in intermediate_skills:
                skill_points += 3
            elif skill in advanced_skills:
                skill_points += 5

        # Deduct points for weak areas
        for area in weak_areas:
            skill_points -= 1

        # Determine level based on points
        if skill_points <= 3:
            return SkillLevel.BEGINNER
        elif skill_points <= 8:
            return SkillLevel.INTERMEDIATE
        elif skill_points <= 15:
            return SkillLevel.ADVANCED
        else:
            return SkillLevel.EXPERT

    def _determine_target_level(
        self, current_level: SkillLevel, learning_goals: List[str]
    ) -> SkillLevel:
        """Determine appropriate target skill level"""

        # Check learning goals for level indicators
        if any("expert" in goal.lower() for goal in learning_goals):
            return SkillLevel.EXPERT
        elif any("advanced" in goal.lower() for goal in learning_goals):
            return SkillLevel.ADVANCED
        elif any("intermediate" in goal.lower() for goal in learning_goals):
            return SkillLevel.INTERMEDIATE

        # Default to one level above current
        level_progression = {
            SkillLevel.BEGINNER: SkillLevel.INTERMEDIATE,
            SkillLevel.INTERMEDIATE: SkillLevel.ADVANCED,
            SkillLevel.ADVANCED: SkillLevel.EXPERT,
            SkillLevel.EXPERT: SkillLevel.EXPERT,
        }

        return level_progression[current_level]

    def _identify_skill_gaps(
        self, current_skills: List[str], weak_areas: List[str], target_level: SkillLevel
    ) -> List[str]:
        """Identify gaps in user's SQL knowledge"""

        gaps = []

        # Add weak areas as gaps
        gaps.extend(weak_areas)

        # Add missing foundational skills
        required_skills = self._get_required_skills(target_level)
        for skill in required_skills:
            if skill not in current_skills:
                gaps.append(skill)

        # Remove duplicates
        return list(set(gaps))

    def _get_required_skills(self, level: SkillLevel) -> List[str]:
        """Get required skills for a given level"""

        skills = {
            SkillLevel.BEGINNER: ["basic_select", "filtering", "sorting"],
            SkillLevel.INTERMEDIATE: [
                "basic_select",
                "filtering",
                "sorting",
                "joins",
                "aggregation",
                "subqueries",
                "security",
            ],
            SkillLevel.ADVANCED: [
                "basic_select",
                "filtering",
                "sorting",
                "joins",
                "aggregation",
                "subqueries",
                "security",
                "optimization",
                "indexes",
                "window_functions",
            ],
            SkillLevel.EXPERT: [
                "basic_select",
                "filtering",
                "sorting",
                "joins",
                "aggregation",
                "subqueries",
                "security",
                "optimization",
                "indexes",
                "window_functions",
                "stored_procedures",
                "transactions",
                "architecture",
            ],
        }

        return skills.get(level, [])

    def _select_modules(
        self,
        current_level: SkillLevel,
        target_level: SkillLevel,
        skill_gaps: List[str],
        preferred_formats: List[str],
    ) -> List[LearningModule]:
        """Select appropriate learning modules"""

        selected_modules = []

        # Get all modules between current and target levels
        for module_id, module in self.learning_modules.items():
            if current_level.value <= module.skill_level.value <= target_level.value:
                # Check if module addresses skill gaps
                module_skills = self._extract_module_skills(module)
                if any(skill in skill_gaps for skill in module_skills):
                    selected_modules.append(module)

        # Sort by skill level
        selected_modules.sort(key=lambda m: m.skill_level.value)

        return selected_modules

    def _extract_module_skills(self, module: LearningModule) -> List[str]:
        """Extract skills covered by a module"""

        skills = []
        for resource in module.resources:
            # Extract from tags
            skills.extend(resource.tags)
            # Extract from category
            skills.append(resource.category.value)

        return list(set(skills))

    def _create_milestones(
        self, modules: List[LearningModule], hours_per_week: int
    ) -> List[Dict[str, Any]]:
        """Create learning milestones"""

        milestones = []
        cumulative_hours = 0

        for i, module in enumerate(modules):
            cumulative_hours += module.total_time_hours
            weeks_needed = int(cumulative_hours / hours_per_week) + 1

            milestone = {
                "milestone_id": f"milestone_{i+1}",
                "title": f"Complete {module.title}",
                "description": module.description,
                "target_week": weeks_needed,
                "requirements": module.completion_criteria,
                "rewards": self._generate_rewards(module.skill_level),
                "celebration_message": self._generate_celebration(module.title),
            }
            milestones.append(milestone)

        return milestones

    def _generate_rewards(self, level: SkillLevel) -> List[str]:
        """Generate rewards for completing milestones"""

        rewards = {
            SkillLevel.BEGINNER: ["SQL Novice Badge", "10 Skill Points"],
            SkillLevel.INTERMEDIATE: ["SQL Practitioner Badge", "25 Skill Points"],
            SkillLevel.ADVANCED: ["SQL Expert Badge", "50 Skill Points"],
            SkillLevel.EXPERT: ["SQL Master Badge", "100 Skill Points", "Certificate"],
        }

        return rewards.get(level, ["5 Skill Points"])

    def _generate_celebration(self, module_title: str) -> str:
        """Generate celebration message for milestone completion"""

        messages = [
            f"🎉 Congratulations! You've mastered {module_title}!",
            f"🌟 Amazing work completing {module_title}!",
            f"🚀 You're making great progress with {module_title}!",
            f"💪 Well done on finishing {module_title}!",
        ]

        return messages[len(module_title) % len(messages)]

    def _generate_practice_exercises(
        self, skill_gaps: List[str], level: SkillLevel
    ) -> List[Dict[str, Any]]:
        """Generate practice exercises for skill gaps"""

        exercises = []

        exercise_templates = {
            "basic_select": {
                "title": "Basic SELECT Practice",
                "type": "write_query",
                "difficulty": "easy",
                "scenario": "Retrieve specific columns from a table",
                "hints": ["Use SELECT", "Specify column names"],
            },
            "joins": {
                "title": "JOIN Practice",
                "type": "complete_query",
                "difficulty": "medium",
                "scenario": "Combine data from multiple tables",
                "hints": ["Consider JOIN type", "Check ON conditions"],
            },
            "optimization": {
                "title": "Query Optimization Challenge",
                "type": "optimize_query",
                "difficulty": "hard",
                "scenario": "Improve slow query performance",
                "hints": ["Check indexes", "Analyze execution plan"],
            },
        }

        for gap in skill_gaps[:10]:  # Limit to 10 exercises
            if gap in exercise_templates:
                exercise = exercise_templates[gap].copy()
                exercise["skill"] = gap
                exercise["estimated_time"] = 15 if level == SkillLevel.BEGINNER else 30
                exercises.append(exercise)

        return exercises

    def _create_assessment_schedule(
        self, modules: List[LearningModule]
    ) -> List[Dict[str, Any]]:
        """Create assessment schedule for learning path"""

        assessments = []

        for i, module in enumerate(modules):
            # Pre-assessment
            assessments.append(
                {
                    "assessment_id": f"pre_{module.module_id}",
                    "type": "pre_assessment",
                    "module": module.title,
                    "timing": f"Before starting {module.title}",
                    "purpose": "Baseline knowledge check",
                    "duration_minutes": 30,
                    "passing_score": 60,
                }
            )

            # Post-assessment
            assessments.append(
                {
                    "assessment_id": f"post_{module.module_id}",
                    "type": "post_assessment",
                    "module": module.title,
                    "timing": f"After completing {module.title}",
                    "purpose": "Mastery verification",
                    "duration_minutes": 45,
                    "passing_score": 80,
                }
            )

        # Final comprehensive assessment
        if modules:
            assessments.append(
                {
                    "assessment_id": "final_assessment",
                    "type": "comprehensive",
                    "module": "All modules",
                    "timing": "End of learning path",
                    "purpose": "Overall skill certification",
                    "duration_minutes": 90,
                    "passing_score": 85,
                }
            )

        return assessments

    def _generate_motivational_tips(
        self, current_level: SkillLevel, target_level: SkillLevel
    ) -> List[str]:
        """Generate motivational tips for the learner"""

        tips = [
            "🎯 Set aside dedicated time each day for SQL practice",
            "💡 Apply what you learn to real problems at work",
            "🤝 Join SQL communities and share your progress",
            "📊 Track your improvement with regular self-assessments",
            "🔄 Review previous concepts regularly to reinforce learning",
            "🎮 Treat exercises like puzzles - enjoy the challenge!",
            "📚 Keep a learning journal to document insights",
            "⏰ Even 15 minutes daily makes a big difference",
            "🏆 Celebrate small wins along your journey",
            "🔍 Don't hesitate to explore beyond the curriculum",
        ]

        # Add level-specific tips
        if current_level == SkillLevel.BEGINNER:
            tips.append("🌱 Everyone starts somewhere - be patient with yourself")
            tips.append("🔨 Focus on understanding concepts before memorizing syntax")
        elif current_level == SkillLevel.INTERMEDIATE:
            tips.append("🚀 You're ready to tackle more complex real-world problems")
            tips.append("🎯 Start contributing to open source SQL projects")
        elif current_level == SkillLevel.ADVANCED:
            tips.append("⭐ Consider mentoring others to reinforce your knowledge")
            tips.append("🏗️ Build portfolio projects to showcase your skills")

        return tips[:8]  # Return top 8 tips

    def adapt_learning_path(
        self, path: PersonalizedLearningPath, progress_data: Dict[str, Any]
    ) -> PersonalizedLearningPath:
        """Adapt learning path based on user progress"""

        # Analyze progress
        completion_rate = progress_data.get("completion_rate", 0)
        assessment_scores = progress_data.get("assessment_scores", [])
        time_spent = progress_data.get("time_spent_hours", 0)
        struggle_areas = progress_data.get("struggle_areas", [])

        # Adapt based on performance
        if (
            completion_rate < 0.5
            and assessment_scores
            and sum(assessment_scores) / len(assessment_scores) < 70
        ):
            # User is struggling - simplify path
            path = self._simplify_path(path, struggle_areas)
        elif (
            completion_rate > 0.8
            and assessment_scores
            and sum(assessment_scores) / len(assessment_scores) > 90
        ):
            # User is excelling - accelerate path
            path = self._accelerate_path(path)

        # Add remedial resources for struggle areas
        if struggle_areas:
            path = self._add_remedial_resources(path, struggle_areas)

        # Update time estimates based on actual pace
        if time_spent > 0:
            path = self._update_time_estimates(path, progress_data)

        return path

    def _simplify_path(
        self, path: PersonalizedLearningPath, struggle_areas: List[str]
    ) -> PersonalizedLearningPath:
        """Simplify learning path for struggling users"""

        # Add more beginner-friendly resources
        for area in struggle_areas:
            # Find easier alternatives
            easier_resources = [
                r
                for r in self.learning_resources.values()
                if area in r.tags and r.skill_level == SkillLevel.BEGINNER
            ]

            if easier_resources:
                # Insert at beginning of path
                new_module = LearningModule(
                    module_id=f"remedial_{area}",
                    title=f"Foundation Review: {area}",
                    description=f"Strengthen your understanding of {area}",
                    skill_level=SkillLevel.BEGINNER,
                    resources=easier_resources[:2],
                    total_time_hours=2.0,
                    learning_objectives=[f"Master {area} fundamentals"],
                    completion_criteria=["Complete all exercises with 80%+ accuracy"],
                )
                path.recommended_modules.insert(0, new_module)

        # Extend timeline
        current_weeks = int(path.estimated_completion_time.split()[0])
        path.estimated_completion_time = (
            f"{current_weeks + 2} weeks (adjusted for review)"
        )

        # Add encouraging message
        path.motivational_tips.insert(
            0, "📈 Taking time to build strong foundations pays off!"
        )

        return path

    def _accelerate_path(
        self, path: PersonalizedLearningPath
    ) -> PersonalizedLearningPath:
        """Accelerate learning path for excelling users"""

        # Skip to more advanced content
        if path.recommended_modules:
            # Remove beginner modules if user is ready
            path.recommended_modules = [
                m
                for m in path.recommended_modules
                if m.skill_level != SkillLevel.BEGINNER
            ]

        # Add bonus advanced content
        advanced_module = LearningModule(
            module_id="bonus_advanced",
            title="Bonus: Advanced SQL Techniques",
            description="Challenge yourself with cutting-edge SQL",
            skill_level=SkillLevel.EXPERT,
            resources=[
                r
                for r in self.learning_resources.values()
                if r.skill_level == SkillLevel.EXPERT
            ][:2],
            total_time_hours=4.0,
            learning_objectives=["Master expert-level SQL techniques"],
            completion_criteria=["Complete advanced challenges"],
        )
        path.recommended_modules.append(advanced_module)

        # Shorten timeline
        current_weeks = int(path.estimated_completion_time.split()[0])
        path.estimated_completion_time = (
            f"{max(1, current_weeks - 1)} weeks (accelerated)"
        )

        # Add achievement message
        path.motivational_tips.insert(
            0, "🚀 You're progressing faster than expected - great work!"
        )

        return path

    def _add_remedial_resources(
        self, path: PersonalizedLearningPath, struggle_areas: List[str]
    ) -> PersonalizedLearningPath:
        """Add remedial resources for struggle areas"""

        for area in struggle_areas:
            # Add specific exercises
            remedial_exercise = {
                "exercise_id": f"remedial_{area}",
                "title": f"Extra Practice: {area}",
                "type": "guided_practice",
                "difficulty": "easy_to_medium",
                "skill": area,
                "estimated_time": 20,
                "hints": [f"Review {area} concepts", "Start with simple examples"],
            }
            path.practice_exercises.insert(0, remedial_exercise)

        return path

    def _update_time_estimates(
        self, path: PersonalizedLearningPath, progress_data: Dict[str, Any]
    ) -> PersonalizedLearningPath:
        """Update time estimates based on actual progress"""

        actual_pace = progress_data.get("hours_per_week_actual", 5)
        planned_pace = progress_data.get("hours_per_week_planned", 5)

        if actual_pace > 0 and planned_pace > 0:
            pace_ratio = actual_pace / planned_pace

            # Adjust total time estimate
            current_weeks = int(path.estimated_completion_time.split()[0])
            adjusted_weeks = int(current_weeks / pace_ratio)

            path.estimated_completion_time = f"{adjusted_weeks} weeks at current pace"

        return path


def format_learning_path_markdown(path: PersonalizedLearningPath) -> str:
    """Format learning path as markdown for display"""

    lines = []

    lines.append("# 📚 Your Personalized SQL Learning Path")
    lines.append("")

    lines.append(f"**Current Level:** {path.current_skill_level.name}")
    lines.append(f"**Target Level:** {path.target_skill_level.name}")
    lines.append(f"**Estimated Time:** {path.estimated_completion_time}")
    lines.append("")

    if path.identified_gaps:
        lines.append("## 🎯 Skills to Develop")
        for gap in path.identified_gaps:
            lines.append(f"- {gap}")
        lines.append("")

    lines.append("## 📖 Learning Modules")
    for i, module in enumerate(path.recommended_modules, 1):
        lines.append(f"### Module {i}: {module.title}")
        lines.append(f"*{module.description}*")
        lines.append(f"- **Duration:** {module.total_time_hours} hours")
        lines.append(f"- **Level:** {module.skill_level.name}")
        lines.append("- **Objectives:**")
        for obj in module.learning_objectives[:3]:
            lines.append(f"  - {obj}")
        lines.append("")

    lines.append("## 🏆 Milestones")
    for milestone in path.milestones[:5]:
        lines.append(f"**Week {milestone['target_week']}:** {milestone['title']}")
        lines.append(f"- {milestone['celebration_message']}")
        lines.append("")

    lines.append("## 💡 Tips for Success")
    for tip in path.motivational_tips[:5]:
        lines.append(tip)
    lines.append("")

    lines.append("## 📊 Progress Tracking")
    tracking = path.progress_tracking
    lines.append(f"- Total Modules: {tracking['total_modules']}")
    lines.append(f"- Total Hours: {tracking['total_hours']}")
    lines.append(f"- Current Streak: {tracking['current_streak_days']} days")

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    generator = LearningPathGenerator()

    # Sample user profile
    user_profile = {
        "user_id": "user_123",
        "current_skills": ["basic_select", "filtering"],
        "weak_areas": ["joins", "optimization"],
        "learning_goals": ["Become proficient in query optimization"],
        "hours_per_week": 5,
        "preferred_formats": ["interactive", "tutorial"],
    }

    # Generate learning path
    learning_path = generator.generate_learning_path(user_profile)

    # Display formatted path
    print(format_learning_path_markdown(learning_path))

    print("\n" + "=" * 50)

    # Simulate progress and adapt
    progress_data = {
        "completion_rate": 0.6,
        "assessment_scores": [75, 82, 79],
        "time_spent_hours": 10,
        "struggle_areas": ["subqueries"],
        "hours_per_week_actual": 4,
        "hours_per_week_planned": 5,
    }

    adapted_path = generator.adapt_learning_path(learning_path, progress_data)
    print("\n## 🔄 Adapted Learning Path")
    print(f"Updated Timeline: {adapted_path.estimated_completion_time}")
    print(f"Added {len(adapted_path.practice_exercises)} practice exercises")
