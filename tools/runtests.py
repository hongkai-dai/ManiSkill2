import os

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(ROOT_DIR, "mani_skill2", "tests")
pytest.main([TEST_DIR])
