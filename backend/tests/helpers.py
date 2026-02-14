"""Shared mock objects for tests."""


class MockTextBlock:
    """Mimics anthropic.types.TextBlock"""
    def __init__(self, text):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    """Mimics anthropic.types.ToolUseBlock"""
    def __init__(self, name, input_data, id="tool_123"):
        self.type = "tool_use"
        self.name = name
        self.input = input_data
        self.id = id


class MockResponse:
    """Mimics anthropic.types.Message"""
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason
