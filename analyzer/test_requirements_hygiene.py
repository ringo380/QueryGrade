"""Guards on requirements-worker.txt.

The worker image once carried torch (plus the whole NVIDIA CUDA stack),
transformers, sentence-transformers, lightgbm and matplotlib with zero import
sites anywhere in the tree - several GB of layers and ~2 min of build time per
deploy for code that did not exist (issue #130).

Nothing failed when they were there, which is why they survived so long: an
unused dependency is invisible to every other test in the suite. This file is
the check that notices.
"""

import re
from pathlib import Path

from django.test import SimpleTestCase

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_REQUIREMENTS = REPO_ROOT / "requirements-worker.txt"

# Distribution name -> the module name you would actually import. Only needed
# where the two differ; anything absent falls back to the normalized dist name.
IMPORT_NAME_OVERRIDES = {
    "scikit-learn": "sklearn",
    "sentence-transformers": "sentence_transformers",
    "psycopg2-binary": "psycopg2",
    "pillow": "PIL",
    "beautifulsoup4": "bs4",
    "python-decouple": "decouple",
}


def _parse_requirements(path):
    """Return the distribution names pinned directly in `path`.

    Skips comments, blank lines and `-r` includes - the point is what THIS
    file adds on top of requirements-prod.txt.
    """
    names = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        # Strip extras, version specifiers and environment markers.
        name = re.split(r"[\[<>=!~;]", line, 1)[0].strip()
        if name:
            names.append(name)
    return names


def _import_name(dist_name):
    key = dist_name.lower()
    return IMPORT_NAME_OVERRIDES.get(key, key.replace("-", "_"))


def _has_import_site(module):
    """True if any tracked .py file imports `module` at a real import site."""
    pattern = re.compile(
        rf"^[ \t]*(?:import[ \t]+{re.escape(module)}\b"
        rf"|from[ \t]+{re.escape(module)}[\. \t])",
        re.MULTILINE,
    )
    for py_file in REPO_ROOT.rglob("*.py"):
        parts = py_file.parts
        if any(p in {".venv", "venv", "node_modules", "migrations"} for p in parts):
            continue
        try:
            if pattern.search(py_file.read_text(errors="ignore")):
                return True
        except OSError:
            continue
    return False


class WorkerRequirementsTests(SimpleTestCase):
    def test_every_worker_dependency_is_actually_imported(self):
        """Every dep pinned in requirements-worker.txt has a real import site.

        This is the rule that torch, transformers, sentence-transformers,
        lightgbm and matplotlib all violated. Adding any of them back without
        a corresponding import fails here.
        """
        unused = []
        for dist in _parse_requirements(WORKER_REQUIREMENTS):
            module = _import_name(dist)
            if not _has_import_site(module):
                unused.append(f"{dist} (looked for `import {module}`)")

        self.assertEqual(
            unused,
            [],
            "requirements-worker.txt pins packages that nothing imports. Each "
            "one costs image size and build time on every worker deploy for "
            "code that does not exist. Either add the import or drop the "
            "dependency (issue #130). Unused: " + ", ".join(unused),
        )

    def test_guard_detects_an_unimported_package(self):
        """The guard above is only meaningful if it can actually fail.

        Asserting a known-absent package is reported keeps the detector honest:
        if _has_import_site ever started returning True for everything, the
        real test would pass vacuously and this one would fail.
        """
        self.assertFalse(
            _has_import_site("torch"),
            "Expected no torch import site. If torch is genuinely used now, "
            "add it back to requirements-worker.txt and update this test.",
        )
        self.assertTrue(
            _has_import_site("xgboost"),
            "Expected to find the xgboost import in analyzer/ml/ensemble/"
            "multi_model.py. If this fails the detector is broken, not the "
            "requirements file.",
        )
