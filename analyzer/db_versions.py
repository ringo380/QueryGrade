"""
Curated database version lists for the grade form version datalist.

Each entry is a (value, label) tuple. Values are submitted as database_version.
Labels include EOL notes where relevant so users know what they're running.
Ordered newest-first within each database.
"""

DATABASE_VERSIONS = {
    "mysql": [
        ("9.1", "MySQL 9.1 (Innovation)"),
        ("9.0", "MySQL 9.0 (Innovation)"),
        ("8.4", "MySQL 8.4 LTS"),
        ("8.3", "MySQL 8.3"),
        ("8.2", "MySQL 8.2"),
        ("8.1", "MySQL 8.1"),
        ("8.0", "MySQL 8.0 LTS (EOL Apr 2026)"),
        ("5.7", "MySQL 5.7 (EOL Oct 2023)"),
        ("5.6", "MySQL 5.6 (EOL Feb 2021)"),
        ("5.5", "MySQL 5.5 (EOL Dec 2018)"),
    ],
    "postgresql": [
        ("17", "PostgreSQL 17"),
        ("16", "PostgreSQL 16"),
        ("15", "PostgreSQL 15"),
        ("14", "PostgreSQL 14"),
        ("13", "PostgreSQL 13"),
        ("12", "PostgreSQL 12"),
        ("11", "PostgreSQL 11 (EOL Nov 2023)"),
        ("10", "PostgreSQL 10 (EOL Nov 2022)"),
        ("9.6", "PostgreSQL 9.6 (EOL Nov 2021)"),
    ],
    "sqlite": [
        ("3.47", "SQLite 3.47 (2024)"),
        ("3.45", "SQLite 3.45 (2024)"),
        ("3.43", "SQLite 3.43 (2023)"),
        ("3.40", "SQLite 3.40 (2022)"),
        ("3.37", "SQLite 3.37 (2021)"),
        ("3.35", "SQLite 3.35 (2021)"),
        ("3.31", "SQLite 3.31 (2020)"),
    ],
    "oracle": [
        ("23c", "Oracle 23c"),
        ("21c", "Oracle 21c"),
        ("19c", "Oracle 19c LTS"),
        ("18c", "Oracle 18c"),
        ("12c", "Oracle 12c"),
        ("11g", "Oracle 11g (EOL Dec 2020)"),
    ],
    "sqlserver": [
        ("2022", "SQL Server 2022"),
        ("2019", "SQL Server 2019"),
        ("2017", "SQL Server 2017"),
        ("2016", "SQL Server 2016"),
        ("2014", "SQL Server 2014 (EOL Jul 2024)"),
        ("2012", "SQL Server 2012 (EOL Jul 2022)"),
        ("2008 R2", "SQL Server 2008 R2 (EOL Jul 2019)"),
    ],
}
