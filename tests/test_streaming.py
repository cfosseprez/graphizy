"""
Tests for the real-time data streaming managers.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch

from graphizy import Graphing
from graphizy.streaming import StreamManager, AsyncStreamManager

# Note: Running the async tests requires `pytest-asyncio`
# You can install it with: pip install pytest-asyncio


@pytest.fixture
def grapher():
    """Provides a basic, configured Graphing instance for testing."""
    g = Graphing(dimension=(100, 100))
    # Configure with a graph type so that update_graphs() can be called
    g.set_graph_type('proximity', proximity_thresh=50.0)
    return g


@pytest.fixture
def sample_data():
    """Provides sample data for streaming."""
    return np.array([[0, 10, 10], [1, 20, 20], [2, 30, 30]])


class TestStreamManager:
    """Tests for the thread-based StreamManager."""

    def test_initialization(self, grapher):
        """Test that StreamManager initializes correctly."""
        sm = StreamManager(grapher, buffer_size=50, update_interval=0.5)
        assert sm.grapher is grapher
        assert sm.buffer_size == 50
        assert sm.update_interval == 0.5
        assert sm.data_queue.maxsize == 50
        assert not sm.is_streaming

    def test_add_callback(self, grapher):
        """Test adding a callback."""
        sm = StreamManager(grapher)
        mock_callback = Mock()
        sm.add_callback(mock_callback)
        assert mock_callback in sm.callbacks

    def test_push_data_and_buffer_full(self, grapher, sample_data):
        """Test pushing data and handling a full buffer."""
        sm = StreamManager(grapher, buffer_size=2)
        assert sm.push_data(sample_data) is True
        assert sm.push_data(sample_data) is True
        # Buffer is now full
        assert sm.push_data(sample_data) is False
        assert sm.stats['dropped_frames'] == 1

    @patch('threading.Thread')
    def test_start_and_stop_streaming(self, mock_thread, grapher):
        """Test the lifecycle of the streaming thread."""
        sm = StreamManager(grapher)
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        sm.start_streaming()
        assert sm.is_streaming
        mock_thread.assert_called_once_with(target=sm._process_stream, daemon=True)
        mock_thread_instance.start.assert_called_once()

        sm.stop_streaming()
        assert not sm.is_streaming
        mock_thread_instance.join.assert_called_once_with(timeout=1.0)

    def test_processing_loop_calls_update(self, grapher, sample_data):
        """Test that the processing loop correctly calls grapher.update_graphs."""
        grapher.update_graphs = Mock(return_value={'proximity': Mock()})

        sm = StreamManager(grapher, update_interval=0.01)
        sm.start_streaming()
        time.sleep(0.02)  # Give thread time to start
        sm.push_data(sample_data)
        time.sleep(0.05)  # Wait for processing
        sm.stop_streaming()

        grapher.update_graphs.assert_called_once()
        # Check the positional argument, not kwargs
        np.testing.assert_array_equal(grapher.update_graphs.call_args.args[0], sample_data)

    def test_callback_is_called(self, grapher, sample_data):
        """Test that registered callbacks are executed after an update."""
        mock_callback = Mock()
        mock_graph_dict = {'proximity': Mock()}
        grapher.update_graphs = Mock(return_value=mock_graph_dict)

        sm = StreamManager(grapher, update_interval=0.01)
        sm.add_callback(mock_callback)
        sm.start_streaming()
        time.sleep(0.02)
        sm.push_data(sample_data)
        time.sleep(0.05)
        sm.stop_streaming()

        mock_callback.assert_called_once_with(mock_graph_dict)


@pytest.mark.asyncio
class TestAsyncStreamManager:
    """Tests for the asyncio-based AsyncStreamManager."""

    async def test_initialization_async(self, grapher):
        """Test async manager initialization."""
        asm = AsyncStreamManager(grapher, buffer_size=50)
        assert asm.grapher is grapher
        assert asm.data_queue.maxsize == 50
        assert not asm.is_streaming
        assert 'dropped_frames' in asm.stats

    async def test_push_data_async(self, grapher, sample_data):
        """Test async data pushing and buffer limits."""
        asm = AsyncStreamManager(grapher, buffer_size=1)
        assert await asm.push_data(sample_data) is True
        # Queue is now full
        assert await asm.push_data(sample_data) is False
        assert asm.stats['dropped_frames'] == 1

    async def test_processing_loop_calls_update_async(self, grapher, sample_data):
        """Test that the async loop calls update_graphs."""
        grapher.update_graphs = Mock(return_value={'proximity': Mock()})
        asm = AsyncStreamManager(grapher)

        await asm.start_streaming()
        await asm.push_data(sample_data)
        await asyncio.sleep(0.05)  # Allow time for processing
        await asm.stop_streaming()

        grapher.update_graphs.assert_called_once()
        # Check positional argument
        np.testing.assert_array_equal(grapher.update_graphs.call_args.args[0], sample_data)
        assert asm.stats['updates_processed'] == 1

    async def test_callback_is_called_async(self, grapher, sample_data):
        """Test that registered async callbacks are executed."""
        mock_callback = Mock()
        mock_graph_dict = {'proximity': Mock()}
        grapher.update_graphs = Mock(return_value=mock_graph_dict)

        asm = AsyncStreamManager(grapher)
        asm.add_callback(mock_callback)

        await asm.start_streaming()
        await asm.push_data(sample_data)
        await asyncio.sleep(0.05)
        await asm.stop_streaming()

        mock_callback.assert_called_once_with(mock_graph_dict)