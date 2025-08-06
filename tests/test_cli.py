"""
Tests for the Command Line Interface (CLI) in graphizy.cli.
"""
import pytest
import sys
import numpy as np
from unittest.mock import patch, Mock
from argparse import Namespace

from graphizy.cli import create_parser, main, cmd_delaunay, cmd_proximity


@patch('graphizy.cli.Graphing')
@patch('graphizy.cli.generate_data')
@patch('graphizy.cli.setup_logging')
def test_cmd_delaunay(mock_setup_logging, mock_generate_data, mock_graphing_class):
    """Test the delaunay command execution logic."""
    # Mock the return values of dependencies
    mock_generate_data.return_value = np.array([[0, 1, 2]])
    mock_grapher = Mock()
    mock_graphing_class.return_value = mock_grapher
    mock_grapher.get_graph_info.return_value = {
        'vertex_count': 1, 'edge_count': 0, 'density': 0.0
    }

    # Simulate arguments passed from the command line
    args = Namespace(
        size=100, particles=10, verbose=True, config=None,
        output="test.jpg", show=True, line_color='255,0,0',
        point_color='0,0,255', line_thickness=1, point_radius=5
    )

    # Run the command function
    cmd_delaunay(args)

    # Assert that the correct functions were called
    mock_setup_logging.assert_called_once_with(True)
    mock_generate_data.assert_called_once()
    mock_graphing_class.assert_called_once()
    mock_grapher.make_graph.assert_called_once_with("delaunay", mock_generate_data.return_value)
    mock_grapher.draw_graph.assert_called_once()
    mock_grapher.save_graph.assert_called_with(mock_grapher.draw_graph.return_value, "test.jpg")
    mock_grapher.show_graph.assert_called_once()


@patch('graphizy.cli.Graphing')
@patch('graphizy.cli.generate_data')
@patch('graphizy.cli.setup_logging')
def test_cmd_proximity(mock_setup_logging, mock_generate_data, mock_graphing_class):
    """Test the proximity command execution logic."""
    # Mock dependencies
    mock_generate_data.return_value = np.array([[0, 1, 2]])
    mock_grapher = Mock()
    mock_graphing_class.return_value = mock_grapher
    mock_grapher.get_graph_info.return_value = {
        'vertex_count': 1, 'edge_count': 0, 'density': 0.0, 'is_connected': True
    }

    # Simulate arguments
    args = Namespace(
        size=100, particles=10, verbose=False, config=None,
        output="prox.jpg", show=False, line_color='0,255,0',
        point_color='0,0,255', line_thickness=1, point_radius=8,
        threshold=75.0, metric='euclidean'
    )

    cmd_proximity(args)

    mock_setup_logging.assert_called_with(False)
    mock_generate_data.assert_called_once()
    mock_graphing_class.assert_called_once()
    # Fix: The CLI now calls make_graph with "proximity"
    mock_grapher.make_graph.assert_called_once_with("proximity", mock_generate_data.return_value, proximity_thresh=75.0, metric='euclidean')
    mock_grapher.save_graph.assert_called_once()


def test_parser_creation():
    """Test that the argument parser is created with the correct commands."""
    parser = create_parser()
    
    # Check that the 'delaunay' command exists
    args = parser.parse_args(['delaunay', '--size', '100'])
    assert args.command == 'delaunay'
    assert args.size == 100

    # Check that the 'proximity' command exists
    args = parser.parse_args(['proximity', '--threshold', '75'])
    assert args.command == 'proximity'
    assert args.threshold == 75.0


def test_parser_requires_command():
    """Test that parser requires a command."""
    parser = create_parser()
    # Check that no command raises SystemExit with help message
    with pytest.raises(SystemExit):
        parser.parse_args([])


@patch('sys.argv', ['graphizy', 'delaunay', '--size', '100', '--particles', '10'])
@patch('graphizy.cli.cmd_delaunay')
def test_main_dispatch(mock_cmd_delaunay):
    """Test that the main function dispatches to the correct command function."""
    main()
    mock_cmd_delaunay.assert_called_once()
    # Check that the parsed args object is passed
    args = mock_cmd_delaunay.call_args[0][0]
    assert isinstance(args, Namespace)
    assert args.size == 100
    assert args.particles == 10

