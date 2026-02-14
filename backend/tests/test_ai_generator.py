"""Tests for AIGenerator — focus on tool-calling flow with CourseSearchTool."""
import pytest
from unittest.mock import MagicMock, call
from ai_generator import AIGenerator
from helpers import MockTextBlock, MockToolUseBlock, MockResponse


def _make_generator(mock_client):
    """Helper: build an AIGenerator with an injected mock Anthropic client."""
    gen = AIGenerator.__new__(AIGenerator)
    gen.client = mock_client
    gen.model = "claude-sonnet-4-20250514"
    gen.base_params = {"model": gen.model, "temperature": 0, "max_tokens": 800}
    gen.SYSTEM_PROMPT = AIGenerator.SYSTEM_PROMPT
    return gen


# ===== Direct (no-tool) responses =====

class TestDirectResponse:
    def test_returns_text_when_no_tool_requested(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            content=[MockTextBlock("Hello!")], stop_reason="end_turn"
        )

        gen = _make_generator(mock_client)
        result = gen.generate_response(query="Hi there")

        assert result == "Hello!"
        mock_client.messages.create.assert_called_once()

    def test_passes_tools_in_api_params(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            content=[MockTextBlock("answer")], stop_reason="end_turn"
        )
        tools = [{"name": "search_course_content", "input_schema": {}}]

        gen = _make_generator(mock_client)
        gen.generate_response(query="question", tools=tools)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_kwarg_when_tools_is_none(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            content=[MockTextBlock("answer")], stop_reason="end_turn"
        )

        gen = _make_generator(mock_client)
        gen.generate_response(query="question", tools=None)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs


# ===== Tool-calling flow =====

class TestToolCallingFlow:
    """Verify the two-call flow: model requests tool → tool executed → result sent back."""

    def _setup_tool_flow(self, tool_name="search_course_content",
                         tool_input=None, tool_exec_result="search output",
                         final_text="Final answer"):
        """Wire up a mock client that does tool_use on first call, end_turn on second."""
        if tool_input is None:
            tool_input = {"query": "neural networks"}

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MockResponse(
                content=[MockToolUseBlock(tool_name, tool_input, id="call_001")],
                stop_reason="tool_use",
            ),
            MockResponse(
                content=[MockTextBlock(final_text)],
                stop_reason="end_turn",
            ),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = tool_exec_result

        gen = _make_generator(mock_client)
        return gen, mock_client, mock_tm

    def test_model_requests_search_tool_and_gets_result(self):
        gen, mock_client, mock_tm = self._setup_tool_flow()
        tools = [{"name": "search_course_content"}]

        result = gen.generate_response(query="What are neural networks?",
                                       tools=tools, tool_manager=mock_tm)

        # Tool was called with correct name and input
        mock_tm.execute_tool.assert_called_once_with(
            "search_course_content", query="neural networks"
        )
        # Final answer returned
        assert result == "Final answer"
        # Two API calls made
        assert mock_client.messages.create.call_count == 2

    def test_tool_result_message_structure(self):
        """The second API call must contain the correct tool_result message."""
        gen, mock_client, mock_tm = self._setup_tool_flow(
            tool_exec_result="[DL - Lesson 1]\nContent here"
        )

        gen.generate_response(query="test", tools=[{}], tool_manager=mock_tm)

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # 3 messages: user, assistant (tool_use), user (tool_result)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        tool_result_content = messages[2]["content"]
        assert len(tool_result_content) == 1
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "call_001"
        assert tool_result_content[0]["content"] == "[DL - Lesson 1]\nContent here"

    def test_second_call_excludes_tools(self):
        """After tool execution, the follow-up call should NOT include tools."""
        gen, mock_client, mock_tm = self._setup_tool_flow()
        gen.generate_response(query="test", tools=[{}], tool_manager=mock_tm)

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs

    def test_system_prompt_preserved_in_second_call(self):
        gen, mock_client, mock_tm = self._setup_tool_flow()
        gen.generate_response(query="test", tools=[{}], tool_manager=mock_tm)

        first_kwargs = mock_client.messages.create.call_args_list[0][1]
        second_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert second_kwargs["system"] == first_kwargs["system"]

    def test_no_tool_manager_skips_execution(self):
        """If tool_manager is None, tool_use response is NOT handled — direct text returned."""
        mock_client = MagicMock()
        # Model returns tool_use but we pass tool_manager=None
        mock_client.messages.create.return_value = MockResponse(
            content=[MockTextBlock("I tried to search"), MockToolUseBlock("search_course_content", {"query": "x"})],
            stop_reason="tool_use",
        )

        gen = _make_generator(mock_client)
        # With tool_manager=None, the code falls through to response.content[0].text
        result = gen.generate_response(query="test", tools=[{}], tool_manager=None)

        assert result == "I tried to search"
        assert mock_client.messages.create.call_count == 1

    def test_text_and_tool_use_in_same_response(self):
        """Model may return TextBlock + ToolUseBlock together."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MockResponse(
                content=[
                    MockTextBlock("Let me search for that."),
                    MockToolUseBlock("search_course_content", {"query": "test"}),
                ],
                stop_reason="tool_use",
            ),
            MockResponse(
                content=[MockTextBlock("Here are the results.")],
                stop_reason="end_turn",
            ),
        ]

        mock_tm = MagicMock()
        mock_tm.execute_tool.return_value = "search results"

        gen = _make_generator(mock_client)
        result = gen.generate_response(query="test", tools=[{}], tool_manager=mock_tm)

        mock_tm.execute_tool.assert_called_once()
        assert result == "Here are the results."

    def test_conversation_history_included_in_system(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            content=[MockTextBlock("answer")], stop_reason="end_turn"
        )

        gen = _make_generator(mock_client)
        gen.generate_response(query="follow-up", conversation_history="User: hi\nAssistant: hello")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_kwargs["system"]
        assert "User: hi" in call_kwargs["system"]
