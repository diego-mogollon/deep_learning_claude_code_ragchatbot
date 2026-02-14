import sys
import os

# Add backend to path so bare imports (from vector_store import ...) resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Add tests dir so helpers module is importable
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from unittest.mock import MagicMock
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store():
    """VectorStore mock that returns one valid search result."""
    store = MagicMock()
    store.search.return_value = SearchResults(
        documents=["Neural networks use layers of interconnected nodes."],
        metadata=[{"course_title": "Deep Learning Basics", "lesson_number": 1}],
        distances=[0.25],
    )
    store.get_lesson_link.return_value = "https://example.com/dl/lesson1"
    store.get_course_outline.return_value = {
        "title": "Deep Learning Basics",
        "course_link": "https://example.com/dl",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/dl/l1"},
            {"lesson_number": 2, "lesson_title": "Backpropagation", "lesson_link": "https://example.com/dl/l2"},
        ],
    }
    return store


@pytest.fixture
def empty_search_results():
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    return SearchResults.empty("Search error: connection failed")
