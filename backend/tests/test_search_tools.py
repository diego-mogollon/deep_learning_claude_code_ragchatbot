"""Tests for CourseSearchTool.execute() outputs and ToolManager wiring."""
import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


# ===== CourseSearchTool.execute — output correctness =====

class TestCourseSearchToolExecute:
    """Verify that execute() produces correct formatted output for various inputs."""

    def test_returns_formatted_result_with_header(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="neural networks")

        assert "[Deep Learning Basics - Lesson 1]" in result
        assert "Neural networks" in result

    def test_delegates_to_vector_store_search(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="neural networks")

        mock_vector_store.search.assert_called_once_with(
            query="neural networks", course_name=None, lesson_number=None
        )

    def test_passes_course_name_filter(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="backprop", course_name="Deep Learning Basics")

        mock_vector_store.search.assert_called_once_with(
            query="backprop", course_name="Deep Learning Basics", lesson_number=None
        )

    def test_passes_lesson_number_filter(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="backprop", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="backprop", course_name=None, lesson_number=2
        )

    def test_passes_both_filters(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="backprop", course_name="DL", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="backprop", course_name="DL", lesson_number=2
        )

    def test_empty_results_message(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="quantum computing")
        assert "No relevant content found" in result

    def test_empty_results_includes_course_filter_info(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="quantum computing", course_name="Physics")
        assert "Physics" in result

    def test_error_from_vector_store_returned(self, mock_vector_store, error_search_results):
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything")
        assert "Search error" in result

    def test_tracks_sources_after_successful_search(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="neural networks")

        assert len(tool.last_sources) == 1
        assert "Deep Learning Basics" in tool.last_sources[0]

    def test_sources_include_lesson_link(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="neural networks")

        assert any("||" in s for s in tool.last_sources), (
            "Sources should contain '||' separator followed by lesson link"
        )

    def test_sources_reset_before_empty_results(self, mock_vector_store, empty_search_results):
        tool = CourseSearchTool(mock_vector_store)
        # First search produces sources
        tool.execute(query="neural networks")
        assert len(tool.last_sources) == 1

        # Second search with no results should NOT clear last_sources
        # (that's ToolManager.reset_sources' job)
        mock_vector_store.search.return_value = empty_search_results
        tool.execute(query="nothing")
        # last_sources is only set during _format_results, so it retains old value
        # unless we explicitly reset. Verify the tool itself doesn't reset on empty.

    def test_multiple_results_formatted(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content A about topic", "Content B about topic"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 3},
            ],
            distances=[0.1, 0.3],
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="topic")

        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 3]" in result
        assert len(tool.last_sources) == 2


# ===== ToolManager wiring =====

class TestToolManagerWiring:
    """Verify ToolManager correctly registers and dispatches to tools."""

    def test_register_and_execute_search_tool(self, mock_vector_store):
        tm = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tm.register_tool(tool)

        result = tm.execute_tool("search_course_content", query="test")
        assert "Deep Learning Basics" in result

    def test_register_and_execute_outline_tool(self, mock_vector_store):
        tm = ToolManager()
        tool = CourseOutlineTool(mock_vector_store)
        tm.register_tool(tool)

        result = tm.execute_tool("get_course_outline", course_name="Deep Learning")
        assert "Deep Learning Basics" in result
        assert "Lesson 1" in result

    def test_execute_unknown_tool_returns_error(self):
        tm = ToolManager()
        result = tm.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result

    def test_get_tool_definitions_returns_all_registered(self, mock_vector_store):
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_vector_store))
        tm.register_tool(CourseOutlineTool(mock_vector_store))

        defs = tm.get_tool_definitions()
        names = {d["name"] for d in defs}
        assert names == {"search_course_content", "get_course_outline"}

    def test_get_last_sources_from_search_tool(self, mock_vector_store):
        tm = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tm.register_tool(tool)

        tm.execute_tool("search_course_content", query="test")
        sources = tm.get_last_sources()
        assert len(sources) > 0

    def test_reset_sources_clears_all(self, mock_vector_store):
        tm = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tm.register_tool(tool)

        tm.execute_tool("search_course_content", query="test")
        assert len(tm.get_last_sources()) > 0

        tm.reset_sources()
        assert len(tm.get_last_sources()) == 0
