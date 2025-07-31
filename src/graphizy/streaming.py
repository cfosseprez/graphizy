"""
Real-time data streaming managers for Graphizy.

This module provides both thread-based and asyncio-based managers for handling
real-time data streams, processing them into graphs, and invoking callbacks.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL-2.0-or-later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import logging
import asyncio
import threading
from queue import Queue, Empty, Full
from typing import Callable, Any
import numpy as np
import time


class StreamManager:
    """Manages streaming data processing for real-time graph updates using threads."""

    def __init__(self, grapher: 'Graphing', buffer_size: int = 1000,
                 update_interval: float = 0.1, auto_memory: bool = True):
        self.grapher = grapher
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.auto_memory = auto_memory

        self.data_queue = Queue(maxsize=buffer_size)
        self.is_streaming = False
        self.callbacks = []

        # Performance metrics
        self.stats = {
            'updates_processed': 0,
            'avg_processing_time': 0,
            'dropped_frames': 0
        }

    def add_callback(self, callback: Callable[[Any], None]) -> None:
        """Add a callback function to be called on each graph update."""
        self.callbacks.append(callback)

    def start_streaming(self) -> None:
        """Start the streaming processing thread."""
        if self.is_streaming:
            return

        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._process_stream, daemon=True)
        self.stream_thread.start()

    def stop_streaming(self) -> None:
        """Stop the streaming processing thread."""
        self.is_streaming = False
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join(timeout=1.0)

    def push_data(self, data_points: np.ndarray) -> bool:
        """
        Push new data to the streaming queue.

        Returns:
            bool: True if data was queued, False if the queue is full.
        """
        try:
            self.data_queue.put_nowait(data_points)
            return True
        except Full:
            self.stats['dropped_frames'] += 1
            return False

    def _process_stream(self) -> None:
        """Main streaming processing loop, run in a separate thread."""
        while self.is_streaming:
            try:
                # Get the latest data point, discarding older ones to reduce latency
                data = self.data_queue.get(timeout=self.update_interval)
                while not self.data_queue.empty():
                    try:
                        data = self.data_queue.get_nowait()
                        self.stats['dropped_frames'] += 1
                    except Empty:
                        break

                # Process the latest data
                start_time = time.time()
                self._process_data_update(data)

                # Update performance stats
                processing_time = time.time() - start_time
                self.stats['updates_processed'] += 1
                self.stats['avg_processing_time'] = (
                        (self.stats['avg_processing_time'] * (self.stats['updates_processed'] - 1) +
                         processing_time) / self.stats['updates_processed']
                )

            except Empty:
                # This is expected when no new data arrives within the timeout
                continue
            except Exception as e:
                logging.error(f"Stream processing error: {e}")

    def _process_data_update(self, data_points: np.ndarray) -> None:
        """Process a single data update and invoke callbacks."""
        try:
            # Auto-update graphs if configured
            if hasattr(self.grapher, 'graph_types'):
                graphs = self.grapher.update_graphs(
                    data_points,
                    use_memory=self.auto_memory,
                    update_memory=self.auto_memory
                )

                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(graphs)
                    except Exception as e:
                        logging.error(f"Callback error: {e}")

        except Exception as e:
            logging.error(f"Data processing error: {e}")


class AsyncStreamManager:
    """Async version of StreamManager for high-throughput applications using asyncio."""

    def __init__(self, grapher: 'Graphing', buffer_size: int = 1000):
        self.grapher = grapher
        self.data_queue = asyncio.Queue(maxsize=buffer_size)
        self.callbacks = []
        self.processing_task = None
        self.is_streaming = False
        self.stats = {
            'updates_processed': 0,
            'dropped_frames': 0
        }

    def add_callback(self, callback: Callable[[Any], None]) -> None:
        """Add a callback function to be called on each graph update."""
        self.callbacks.append(callback)

    async def start_streaming(self) -> None:
        """Start the async streaming task."""
        if self.is_streaming:
            return
        self.is_streaming = True
        self.processing_task = asyncio.create_task(self._process_stream())

    async def stop_streaming(self) -> None:
        """Stop the async streaming task gracefully."""
        self.is_streaming = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                logging.info("Async streaming task stopped.")
        self.processing_task = None

    async def push_data(self, data_points: np.ndarray) -> bool:
        """Push data asynchronously without blocking."""
        try:
            self.data_queue.put_nowait(data_points)
            return True
        except asyncio.QueueFull:
            self.stats['dropped_frames'] += 1
            return False

    async def _process_stream(self) -> None:
        """Async processing loop."""
        while self.is_streaming:
            try:
                # Wait for the next item, but with a timeout to remain responsive
                data = await asyncio.wait_for(self.data_queue.get(), timeout=0.1)

                # Discard older frames to process the most recent one
                while not self.data_queue.empty():
                    try:
                        data = self.data_queue.get_nowait()
                        self.stats['dropped_frames'] += 1
                    except asyncio.QueueEmpty:
                        break

                await self._process_data_async(data)
                self.stats['updates_processed'] += 1

            except asyncio.TimeoutError:
                continue  # No data, just loop
            except asyncio.CancelledError:
                break  # Exit loop on cancellation
            except Exception as e:
                logging.error(f"Async stream error: {e}")

    async def _process_data_async(self, data_points: np.ndarray) -> None:
        """Process data asynchronously and call callbacks."""
        # In a real high-performance scenario, you might run this in an executor
        # for CPU-bound work. For consistency, we call it directly here.
        if hasattr(self.grapher, 'graph_types'):
            graphs = self.grapher.update_graphs(data_points, use_memory=True, update_memory=True)

            # Call callbacks
            for callback in self.callbacks:
                try:
                    # If the callback is async, await it
                    if asyncio.iscoroutinefunction(callback):
                        await callback(graphs)
                    else:
                        callback(graphs)
                except Exception as e:
                    logging.error(f"Async callback error: {e}")