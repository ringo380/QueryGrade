"""
Seed the ML training pipeline with synthetic training data derived from the rule-based grader.

The ML model initially learns to replicate rule-based scoring. As real user feedback
accumulates, process_ml_feedback overwrites these samples with higher-confidence data.
"""
import hashlib
import logging

from django.core.management.base import BaseCommand
from django.db import transaction

from analyzer.models import Query, TrainingData
from analyzer.analyzers.base import QueryGrader
from analyzer.ml.core.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

SEED_QUERIES = [
    # --- Grade A queries (clean, well-optimised) ---
    "SELECT u.id, u.name, u.email FROM users u WHERE u.id = 1",
    "SELECT o.id, o.total, u.name FROM orders o JOIN users u ON o.user_id = u.id WHERE o.status = 'active' LIMIT 50",
    "SELECT COUNT(*) FROM orders WHERE created_at >= '2024-01-01'",
    "SELECT p.id, p.title, c.name AS category FROM products p JOIN categories c ON p.category_id = c.id WHERE p.active = 1 ORDER BY p.title LIMIT 20",
    "SELECT u.id, COUNT(o.id) AS order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id HAVING COUNT(o.id) > 5",
    "SELECT id, name FROM products WHERE category_id = 3 AND active = 1 ORDER BY created_at DESC LIMIT 10",
    "SELECT AVG(score) FROM ratings WHERE product_id = 42",
    "SELECT e.id, e.name, d.name AS dept FROM employees e JOIN departments d ON e.dept_id = d.id WHERE e.active = 1",
    "SELECT id, title FROM articles WHERE published_at BETWEEN '2024-01-01' AND '2024-12-31' ORDER BY published_at DESC LIMIT 25",
    "SELECT user_id, SUM(amount) AS total FROM payments WHERE status = 'completed' GROUP BY user_id",

    # --- Grade B queries (mostly good, minor issues) ---
    "SELECT id, name, email, phone, address, city, country FROM users WHERE active = 1",
    "SELECT DISTINCT category FROM products WHERE price > 100",
    "SELECT u.id, u.name, o.id AS order_id FROM users u, orders o WHERE u.id = o.user_id AND o.status = 'active'",
    "SELECT p.*, c.name AS cat_name FROM products p JOIN categories c ON p.category_id = c.id",
    "SELECT id, name, score FROM leaderboard ORDER BY score DESC",
    "SELECT user_id, COUNT(*) AS cnt FROM sessions GROUP BY user_id HAVING cnt > 3",
    "SELECT a.id, a.title, COUNT(c.id) FROM articles a LEFT JOIN comments c ON a.id = c.article_id GROUP BY a.id, a.title ORDER BY a.published_at DESC",
    "SELECT id, name FROM customers WHERE LOWER(name) LIKE '%smith%'",
    "SELECT *, ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn FROM employees",
    "SELECT t.id, t.name, COUNT(p.id) AS project_count FROM teams t LEFT JOIN projects p ON t.id = p.team_id GROUP BY t.id, t.name ORDER BY project_count DESC LIMIT 10",

    # --- Grade C queries (noticeable performance issues) ---
    "SELECT * FROM users",
    "SELECT * FROM orders WHERE status = 'active'",
    "SELECT * FROM products WHERE YEAR(created_at) = 2024",
    "SELECT id, name FROM employees WHERE UPPER(department) = 'ENGINEERING'",
    "SELECT o.*, u.name FROM orders o JOIN users u ON o.user_id = u.id WHERE u.country = 'US' ORDER BY o.created_at",
    "SELECT * FROM logs WHERE DATE(created_at) = '2024-06-15'",
    "SELECT id, name FROM products WHERE category_id IN (SELECT id FROM categories WHERE active = 1)",
    "SELECT u.id, (SELECT COUNT(*) FROM orders WHERE user_id = u.id) AS cnt FROM users u",
    "SELECT * FROM events WHERE MONTH(event_date) = 3 AND YEAR(event_date) = 2024",
    "SELECT id, name FROM items WHERE LENGTH(description) > 100",

    # --- Grade D queries (significant issues) ---
    "SELECT * FROM users WHERE name LIKE '%john%'",
    "SELECT * FROM products WHERE price * 1.2 > 100",
    "SELECT * FROM orders o, order_items i WHERE o.id = i.order_id",
    "SELECT * FROM logs ORDER BY RAND() LIMIT 10",
    "SELECT * FROM users WHERE id NOT IN (SELECT user_id FROM orders)",
    "SELECT * FROM transactions WHERE CONVERT(VARCHAR, amount) = '100.00'",
    "SELECT * FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
    "SELECT DISTINCT * FROM products p JOIN categories c ON p.category_id = c.id",
    "SELECT * FROM messages WHERE LOWER(subject) LIKE '%urgent%' ORDER BY created_at",
    "SELECT * FROM audit_log WHERE CAST(user_id AS VARCHAR) = '42'",

    # --- Grade F queries (severe issues) ---
    "SELECT * FROM users",
    "SELECT *, (SELECT name FROM categories WHERE id = p.category_id) FROM products p",
    "SELECT * FROM orders o WHERE o.user_id IN (SELECT id FROM users WHERE country IN (SELECT id FROM countries WHERE region = 'EU'))",
    "SELECT * FROM big_table WHERE LIKE_PATTERN LIKE '%search_term%' ORDER BY RAND()",
    "SELECT * FROM logs WHERE FUNCTION_ON_COLUMN(created_at) = 'some_value'",
    "SELECT * FROM a, b, c WHERE a.id != b.id",
    "SELECT DISTINCT a.*, b.*, c.* FROM table_a a JOIN table_b b ON a.id = b.a_id JOIN table_c c ON b.id = c.b_id ORDER BY RAND()",
    "SELECT * FROM users WHERE NOT EXISTS (SELECT 1 FROM orders WHERE user_id = users.id AND status NOT IN (SELECT status FROM allowed_statuses WHERE active = 1))",
    "SELECT * FROM products WHERE SUBSTR(category_name, 1, 3) = 'ele' OR SUBSTR(category_name, 1, 3) = 'boo' ORDER BY price * quantity DESC",
    "SELECT id, name, (SELECT COUNT(*) FROM orders WHERE user_id = u.id) AS orders, (SELECT SUM(amount) FROM payments WHERE user_id = u.id) AS total_paid, (SELECT MAX(created_at) FROM sessions WHERE user_id = u.id) AS last_seen FROM users u",

    # --- Additional variety: JOINs and aggregations ---
    "SELECT d.name, COUNT(e.id) AS headcount, AVG(e.salary) AS avg_salary FROM departments d LEFT JOIN employees e ON d.id = e.dept_id GROUP BY d.id, d.name ORDER BY headcount DESC",
    "SELECT p.id, p.name, COALESCE(SUM(s.quantity), 0) AS units_sold FROM products p LEFT JOIN sales s ON p.id = s.product_id WHERE p.active = 1 GROUP BY p.id, p.name",
    "SELECT r.id, r.name, COUNT(rv.id) AS review_count, ROUND(AVG(rv.rating), 2) AS avg_rating FROM restaurants r JOIN reviews rv ON r.id = rv.restaurant_id GROUP BY r.id, r.name HAVING COUNT(rv.id) >= 10 ORDER BY avg_rating DESC LIMIT 20",
    "SELECT t.id, t.name, COUNT(tm.user_id) AS members FROM teams t JOIN team_members tm ON t.id = tm.team_id WHERE t.active = 1 GROUP BY t.id, t.name",
    "SELECT c.id, c.name, SUM(o.total) AS revenue FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.created_at >= '2024-01-01' GROUP BY c.id, c.name ORDER BY revenue DESC LIMIT 10",

    # --- Subqueries and CTEs ---
    "WITH active_users AS (SELECT id FROM users WHERE last_login >= '2024-01-01') SELECT COUNT(*) FROM active_users",
    "WITH ranked AS (SELECT id, score, RANK() OVER (ORDER BY score DESC) AS rnk FROM scores) SELECT * FROM ranked WHERE rnk <= 10",
    "SELECT u.id, u.name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.status = 'completed')",
    "SELECT p.id, p.name FROM products p WHERE NOT EXISTS (SELECT 1 FROM inventory i WHERE i.product_id = p.id AND i.quantity > 0)",
    "SELECT user_id, total FROM (SELECT user_id, SUM(amount) AS total FROM payments GROUP BY user_id) sub WHERE total > 1000",

    # --- Window functions ---
    "SELECT id, name, salary, RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS dept_rank FROM employees",
    "SELECT id, created_at, amount, SUM(amount) OVER (PARTITION BY user_id ORDER BY created_at) AS running_total FROM payments",
    "SELECT product_id, sale_date, revenue, LAG(revenue) OVER (PARTITION BY product_id ORDER BY sale_date) AS prev_revenue FROM daily_sales",

    # --- Date/time patterns (good) ---
    "SELECT id, name FROM events WHERE event_date >= CURRENT_DATE AND event_date < CURRENT_DATE + INTERVAL '7 days' ORDER BY event_date LIMIT 50",
    "SELECT DATE_TRUNC('month', created_at) AS month, COUNT(*) AS signups FROM users GROUP BY 1 ORDER BY 1",

    # --- Index-friendly vs index-unfriendly ---
    "SELECT id, email FROM users WHERE email = 'user@example.com'",
    "SELECT id FROM products WHERE category_id = 5 AND price BETWEEN 10 AND 50",
    "SELECT * FROM logs WHERE SUBSTRING(message, 1, 10) = 'ERROR:     '",
    "SELECT * FROM users WHERE CONCAT(first_name, ' ', last_name) = 'John Smith'",

    # --- INSERT / UPDATE / DELETE (non-SELECT) ---
    "INSERT INTO users (name, email, created_at) VALUES ('Alice', 'alice@example.com', NOW())",
    "UPDATE products SET price = price * 1.05 WHERE category_id = 3",
    "DELETE FROM sessions WHERE expires_at < NOW()",
    "UPDATE orders SET status = 'shipped', updated_at = NOW() WHERE id = 1234 AND status = 'pending'",

    # --- Complex real-world patterns ---
    "SELECT u.id, u.name, u.email, p.avatar_url, COUNT(DISTINCT f.follower_id) AS follower_count FROM users u LEFT JOIN profiles p ON u.id = p.user_id LEFT JOIN follows f ON u.id = f.followed_id WHERE u.active = 1 GROUP BY u.id, u.name, u.email, p.avatar_url ORDER BY follower_count DESC LIMIT 20",
    "SELECT o.id, o.created_at, o.total, u.name AS customer, GROUP_CONCAT(p.name ORDER BY p.name SEPARATOR ', ') AS products FROM orders o JOIN users u ON o.user_id = u.id JOIN order_items oi ON o.id = oi.order_id JOIN products p ON oi.product_id = p.id WHERE o.status = 'completed' GROUP BY o.id, o.created_at, o.total, u.name ORDER BY o.created_at DESC LIMIT 25",
    "SELECT s.store_id, s.name, SUM(t.amount) AS total_sales, COUNT(t.id) AS transaction_count, AVG(t.amount) AS avg_transaction FROM stores s JOIN transactions t ON s.id = t.store_id WHERE t.created_at BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY s.store_id, s.name HAVING SUM(t.amount) > 10000 ORDER BY total_sales DESC",

    # --- Anti-patterns ---
    "SELECT * FROM orders WHERE 1=1",
    "SELECT id, name FROM products WHERE id > 0 ORDER BY RAND() LIMIT 5",
    "SELECT * FROM users u JOIN orders o ON u.id = o.user_id JOIN order_items oi ON o.id = oi.order_id JOIN products p ON oi.product_id = p.id JOIN categories c ON p.category_id = c.id WHERE u.active = 1",
    "SELECT *, (SELECT MAX(created_at) FROM orders WHERE user_id = u.id) AS last_order FROM users u ORDER BY last_order DESC",
    "SELECT DISTINCT id, name, email FROM users WHERE active = 1 GROUP BY id, name, email",

    # --- More edge cases ---
    "SELECT id FROM users WHERE id BETWEEN 1000 AND 2000",
    "SELECT name FROM products WHERE name LIKE 'iPhone%' LIMIT 10",
    "SELECT id, amount FROM invoices WHERE amount > 0 AND status != 'void' ORDER BY created_at DESC LIMIT 100",
    "SELECT COUNT(DISTINCT user_id) FROM page_views WHERE page = '/home' AND created_at >= '2024-01-01'",
    "SELECT p.id, p.name, p.price FROM products p WHERE p.price < (SELECT AVG(price) FROM products WHERE category_id = p.category_id)",
]


class Command(BaseCommand):
    help = "Seed ML training data from the rule-based grader (synthetic bootstrap)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete existing synthetic samples before seeding",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be created without writing to DB",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        clear = options["clear"]

        grader = QueryGrader()
        extractor = FeatureExtractor()

        if clear and not dry_run:
            deleted, _ = TrainingData.objects.filter(validation_source="synthetic_seed").delete()
            self.stdout.write(f"Deleted {deleted} existing synthetic samples")

        created = 0
        skipped = 0
        errors = 0

        for sql in SEED_QUERIES:
            sql = sql.strip()
            if not sql:
                continue

            try:
                with transaction.atomic():
                    # Grade via rule-based system
                    query_obj, analysis = grader.analyze_query(sql)

                    # Extract features
                    features = extractor.extract_features(query_obj)
                    if features is None:
                        self.stderr.write(f"Feature extraction failed: {sql[:60]}")
                        errors += 1
                        continue

                    if dry_run:
                        self.stdout.write(
                            f"  [dry-run] {analysis.grade} ({analysis.score:.0f}) — {sql[:70]}"
                        )
                        created += 1
                        continue

                    # Skip if this exact query already has a synthetic sample
                    if TrainingData.objects.filter(
                        query=query_obj, validation_source="synthetic_seed"
                    ).exists():
                        skipped += 1
                        continue

                    # Convert letter grade to 1-5 user scale for user_grade_avg
                    grade_to_rating = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "F": 1.0}
                    user_grade_avg = grade_to_rating.get(analysis.grade, 3.0)

                    TrainingData.objects.create(
                        query=query_obj,
                        user_grade_avg=user_grade_avg,
                        user_grade_count=1,
                        user_grade_stddev=0.0,
                        system_grade=analysis.grade,
                        system_score=analysis.score,
                        features_json=features,
                        target_score=analysis.score,
                        feedback_weight=0.5,  # lower weight than real user feedback
                        query_complexity=query_obj.estimated_complexity,
                        table_count=query_obj.table_count,
                        join_count=query_obj.join_count,
                        is_validated=True,
                        validation_source="synthetic_seed",
                    )
                    created += 1

            except Exception as e:
                self.stderr.write(f"Error processing query: {sql[:60]}\n  {e}")
                errors += 1

        action = "Would create" if dry_run else "Created"
        self.stdout.write(
            self.style.SUCCESS(
                f"\n{action} {created} training samples  |  skipped {skipped}  |  errors {errors}"
            )
        )

        if not dry_run and created > 0:
            total = TrainingData.objects.count()
            self.stdout.write(f"Total TrainingData records: {total}")
            if total >= 50:
                self.stdout.write(
                    self.style.SUCCESS(
                        "Ready to train. Run: python manage.py train_ml_model --algorithm random_forest"
                    )
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f"Need {50 - total} more samples to reach training threshold (50)")
                )
