"""Tests for RAGSystem.query() — the full orchestration flow for content queries."""
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from helpers import MockTextBlock, MockToolUseBlock, MockResponse


@dataclass
class FakeConfig:
    ANTHROPIC_API_KEY: str = "fake-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "/tmp/test_chroma"


def _build_rag_system(mock_anthropic_client, mock_vector_store):
    """Build a RAGSystem with real AIGenerator logic but mocked externals.

    Patches:
      - VectorStore (avoids ChromaDB)
      - anthropic.Anthropic (injects mock client into real AIGenerator)
    """
    with patch("vector_store.chromadb"), \
         patch("rag_system.VectorStore") as MockVS, \
         patch("ai_generator.anthropic.Anthropic") as MockAnthropicCls:

        MockVS.return_value = mock_vector_store
        MockAnthropicCls.return_value = mock_anthropic_client

        from rag_system import RAGSystem
        rag = RAGSystem(FakeConfig())

    return rag


# ===== RAGSystem.query integration =====

class TestRAGSystemQuery:
    """Test the query method end-to-end with mocked AI and vector store."""

    def test_query_without_tool_use(self, mock_vector_store):
        """When model answers directly (no tool), query returns the text."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            content=[MockTextBlock("General knowledge answer")],
            stop_reason="end_turn",
        )

        rag = _build_rag_system(mock_client, mock_vector_store)
        answer, sources = rag.query("What is deep learning?")

        assert answer == "General knowledge answer"
        assert isinstance(sources, list)

    def test_query_with_tool_use_returns_final_answer(self, mock_vector_store):
        """Full flow: model calls search tool → tool executed → final answer."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            # First call: model requests search tool
            MockResponse(
                content=[MockToolUseBlock("search_course_content", {"query": "neural nets"})],
                stop_reason="tool_use",
            ),
            # Second call: model produces final answer
            MockResponse(
                content=[MockTextBlock("Neural networks are computational models...")],
                stop_reason="end_turn",
            ),
        ]

        rag = _build_rag_system(mock_client, mock_vector_store)
        answer, sources = rag.query("What are neural networks?")

        assert "Neural networks" in answer
        assert mock_client.messages.create.call_count == 2

    def test_query_returns_sources_from_search(self, mock_vector_store):
        """After a search, sources should be populated and then reset."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MockResponse(
                content=[MockToolUseBlock("search_course_content", {"query": "test"})],
                stop_reason="tool_use",
            ),
            MockResponse(
                content=[MockTextBlock("Answer")],
                stop_reason="end_turn",
            ),
        ]

        rag = _build_rag_system(mock_client, mock_vector_store)
        answer, sources = rag.query("test question")

        # Sources should have been collected from CourseSearchTool
        assert isinstance(sources, list)
        # After query(), sources should have been reset
        assert len(rag.tool_manager.get_last_sources()) == 0

    def test_query_passes_tools_to_ai(self, mock_vector_store):
        """Verify that tool definitions are passed to the Anthropic API call."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            content=[MockTextBlock("answer")], stop_reason="end_turn"
        )

        rag = _build_rag_system(mock_client, mock_vector_store)
        rag.query("any question")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        tool_names = {t["name"] for t in call_kwargs["tools"]}
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_query_with_session_tracks_history(self, mock_vector_store):
        """Conversation history should be added after each query."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockResponse(
            content=[MockTextBlock("first answer")], stop_reason="end_turn"
        )

        rag = _build_rag_system(mock_client, mock_vector_store)
        session_id = rag.session_manager.create_session()

        rag.query("first question", session_id=session_id)
        history = rag.session_manager.get_conversation_history(session_id)

        assert history is not None
        assert "first answer" in history

    def test_query_exception_propagates(self, mock_vector_store):
        """If the AI call raises, the exception should propagate (not be swallowed)."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        rag = _build_rag_system(mock_client, mock_vector_store)

        with pytest.raises(Exception, match="API error"):
            rag.query("anything")

    def test_tool_execution_uses_real_search_tool(self, mock_vector_store):
        """Verify that tool_use triggers the real CourseSearchTool, not a mock."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MockResponse(
                content=[MockToolUseBlock("search_course_content", {"query": "layers"})],
                stop_reason="tool_use",
            ),
            MockResponse(
                content=[MockTextBlock("Layers explanation")],
                stop_reason="end_turn",
            ),
        ]

        rag = _build_rag_system(mock_client, mock_vector_store)
        rag.query("tell me about layers")

        # The real CourseSearchTool should have called vector_store.search
        mock_vector_store.search.assert_called_once_with(
            query="layers", course_name=None, lesson_number=None
        )

    def test_tool_result_content_sent_to_model(self, mock_vector_store):
        """The tool result string should appear in the second API call messages."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MockResponse(
                content=[MockToolUseBlock("search_course_content", {"query": "nodes"}, id="call_x")],
                stop_reason="tool_use",
            ),
            MockResponse(
                content=[MockTextBlock("Final")],
                stop_reason="end_turn",
            ),
        ]

        rag = _build_rag_system(mock_client, mock_vector_store)
        rag.query("what are nodes?")

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Find the tool_result message
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        tool_result = tool_result_msg["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "call_x"
        # Content should be the formatted search output from CourseSearchTool
        assert "Deep Learning Basics" in tool_result["content"]


# ===== Config → VectorStore MAX_RESULTS pipeline =====

class TestMaxResultsConfig:
    """Verify that MAX_RESULTS flows from config through VectorStore to ChromaDB.

    The bug: MAX_RESULTS = 0 in config.py causes n_results=0 in every ChromaDB
    query, silently returning empty results for all searches.
    """

    def test_max_results_passed_to_vector_store_constructor(self, mock_vector_store):
        """Config.MAX_RESULTS must be forwarded to VectorStore(max_results=...)."""
        with patch("vector_store.chromadb"), \
             patch("rag_system.VectorStore") as MockVS, \
             patch("ai_generator.anthropic.Anthropic"):
            MockVS.return_value = mock_vector_store

            from rag_system import RAGSystem
            rag = RAGSystem(FakeConfig(MAX_RESULTS=5))

            # VectorStore constructor should have received max_results=5
            _, call_kwargs = MockVS.call_args
            assert call_kwargs.get("max_results", MockVS.call_args[0][2] if len(MockVS.call_args[0]) > 2 else None) == 5

    def test_max_results_zero_produces_empty_search(self):
        """With MAX_RESULTS=0, VectorStore.search passes n_results=0 to ChromaDB → no results."""
        from vector_store import VectorStore, SearchResults

        mock_chroma_client = MagicMock()
        mock_collection = MagicMock()
        # ChromaDB returns empty when n_results=0
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch("vector_store.chromadb.PersistentClient", return_value=mock_chroma_client), \
             patch("vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"):
            store = VectorStore("/tmp/test", "all-MiniLM-L6-v2", max_results=0)

        results = store.search(query="neural networks")

        # n_results=0 was passed to ChromaDB
        mock_collection.query.assert_called_once()
        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["n_results"] == 0, (
            f"Expected n_results=0 (the bug), got n_results={call_kwargs['n_results']}"
        )
        # Result is empty — this is the broken behavior
        assert results.is_empty()

    def test_max_results_five_requests_five_from_chromadb(self):
        """With MAX_RESULTS=5, VectorStore.search passes n_results=5 to ChromaDB."""
        mock_chroma_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"course_title": "C1"}, {"course_title": "C2"}]],
            "distances": [[0.1, 0.2]],
        }
        mock_chroma_client.get_or_create_collection.return_value = mock_collection

        with patch("vector_store.chromadb.PersistentClient", return_value=mock_chroma_client), \
             patch("vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"):
            from vector_store import VectorStore
            store = VectorStore("/tmp/test", "all-MiniLM-L6-v2", max_results=5)

        results = store.search(query="neural networks")

        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["n_results"] == 5
        assert not results.is_empty()

    def test_config_max_results_must_be_positive(self):
        """Guard: MAX_RESULTS should be > 0 for searches to work."""
        from config import Config
        cfg = Config()
        assert cfg.MAX_RESULTS > 0, (
            f"MAX_RESULTS is {cfg.MAX_RESULTS} — must be > 0 or all searches return empty"
        )


# ===== Tool registration =====

class TestRAGSystemToolRegistration:
    """Verify that both tools are properly registered."""

    def test_both_tools_registered(self, mock_vector_store):
        mock_client = MagicMock()
        rag = _build_rag_system(mock_client, mock_vector_store)

        tool_names = {t["name"] for t in rag.tool_manager.get_tool_definitions()}
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
