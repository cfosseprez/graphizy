"""
CLI tests for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock
from argparse import Namespace

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphizy.cli import (
    create_parser, parse_color, load_config, create_config_from_args,
    cmd_delaunay, cmd_proximity, cmd_both, cmd_info, main
)
from graphizy import GraphizyConfig, GraphizyError


class TestCLIParser:
    """Test CLI argument parsing"""

    def test_create_parser_basic(self):
        """Test basic parser creation"""
        parser = create_parser()
        assert parser is not None

        # Test help doesn't crash
        with pytest.raises(SystemExit):
            parser.parse_args(['--help'])

    def test_delaunay_command(self):
        """Test delaunay subcommand parsing"""
        parser = create_parser()
        args = parser.parse_args(['delaunay', '--size', '800', '--particles', '100'])

        assert args.command == 'delaunay'
        assert args.size == 800
        assert args.particles == 100

    def test_proximity_command(self):
        """Test proximity subcommand parsing"""
        parser = create_parser()
        args = parser.parse_args([
            'proximity', '--size', '600', '--particles', '50',
            '--threshold', '25.5', '--metric', 'manhattan'
        ])

        assert args.command == 'proximity'
        assert args.size == 600
        assert args.particles == 50
        assert args.threshold == 25.5
        assert args.metric == 'manhattan'

    def test_both_command(self):
        """Test both subcommand parsing"""
        parser = create_parser()
        args = parser.parse_args([
            'both', '--size', '1000', '--threshold', '30',
            '--delaunay-output', 'del.jpg', '--proximity-output', 'prox.jpg'
        ])

        assert args.command == 'both'
        assert args.size == 1000
        assert args.threshold == 30
        assert args.delaunay_output == 'del.jpg'
        assert args.proximity_output == 'prox.jpg'

    def test_info_command(self):
        """Test info subcommand parsing"""
        parser = create_parser()
        args = parser.parse_args(['info', '--size', '500', '--output', 'info.json'])

        assert args.command == 'info'
        assert args.size == 500
        assert args.output == 'info.json'

    def test_common_arguments(self):
        """Test common arguments across subcommands"""
        parser = create_parser()
        args = parser.parse_args([
            'delaunay', '--size', '400', '--particles', '75',
            '--show', '--verbose', '--output', 'test.jpg',
            '--line-color', '255,0,0', '--point-color', '0,255,0',
            '--line-thickness', '2', '--point-radius', '10'
        ])

        assert args.size == 400
        assert args.particles == 75
        assert args.show is True
        assert args.verbose is True
        assert args.output == 'test.jpg'
        assert args.line_color == '255,0,0'
        assert args.point_color == '0,255,0'
        assert args.line_thickness == 2
        assert args.point_radius == 10


class TestCLIUtilities:
    """Test CLI utility functions"""

    def test_parse_color_valid(self):
        """Test valid color parsing with correct BGR conversion"""
        assert parse_color('255,0,0') == (0, 0, 255)  # Red
        assert parse_color('0,255,0') == (0, 255, 0)  # Green
        assert parse_color('0,0,255') == (255, 0, 0)  # Blue

    def test_parse_color_corrected(self):
        """Test color parsing with correct BGR conversion"""
        color = parse_color('255,0,0')  # Red RGB -> Blue BGR
        assert color == (0, 0, 255)

        color = parse_color('0,255,0')  # Green RGB -> Green BGR
        assert color == (0, 255, 0)

        color = parse_color('0,0,255')  # Blue RGB -> Red BGR
        assert color == (255, 0, 0)

    def test_parse_color_invalid(self):
        """Test invalid color parsing"""
        with pytest.raises(ValueError):
            parse_color('255,0')
        with pytest.raises(ValueError):
            parse_color('invalid')

    def test_parse_color_edge_cases(self):
        """Test color parsing edge cases"""
        # Test boundary values
        color = parse_color('0,0,0')
        assert color == (0, 0, 0)

        color = parse_color('255,255,255')
        assert color == (255, 255, 255)

    def test_load_config_valid(self):
        """Test loading valid configuration file"""
        config_data = {"drawing": {"point_radius": 12}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        try:
            loaded_config = load_config(config_file)
            assert loaded_config == config_data
        finally:
            os.unlink(config_file)

    def test_load_config_invalid(self):
        """Test loading invalid or non-existent configuration file"""
        with pytest.raises(GraphizyError):
            load_config('non_existent_file.json')
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('invalid json')
            config_file = f.name
        try:
            with pytest.raises(GraphizyError):
                load_config(config_file)
        finally:
            os.unlink(config_file)

class TestConfigCreation:
    """Test configuration creation from CLI arguments"""

    def test_create_config_basic(self):
        """Test basic configuration creation"""
        args = Namespace(
            size=800,
            particles=150,
            verbose=False,
            config=None,
            line_color='255,0,0',
            point_color='0,255,0',
            line_thickness=2,
            point_radius=10
        )

        config = create_config_from_args(args)

        assert config.graph.dimension == (800, 800)
        assert config.generation.num_particles == 150
        assert config.drawing.line_color == (0, 0, 255)  # BGR conversion
        assert config.drawing.point_color == (0, 255, 0)
        assert config.drawing.line_thickness == 2
        assert config.drawing.point_radius == 10

    def test_create_config_with_file(self):
        """Test configuration creation with config file"""
        # Create temporary config file
        config_data = {
            "drawing": {
                "line_thickness": 3
            },
            "graph": {
                "proximity_threshold": 75.0
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            args = Namespace(
                size=600,
                particles=100,
                verbose=True,
                config=config_file,
                line_color='0,0,255',
                point_color='255,255,0',
                line_thickness=1,  # This gets overridden by file setting
                point_radius=8
            )

            config = create_config_from_args(args)

            # File settings should be loaded
            assert config.graph.proximity_threshold == 75.0

            # CLI args should override file settings
            assert config.drawing.line_thickness == 3
            assert config.graph.dimension == (600, 600)

        finally:
            os.unlink(config_file)

    def test_create_config_proximity_args(self):
        """Test configuration with proximity-specific arguments"""
        args = Namespace(
            size=1000,
            particles=200,
            verbose=False,
            config=None,
            threshold=25.5,
            metric='manhattan',
            line_color='128,128,128',
            point_color='64,64,64',
            line_thickness=1,
            point_radius=6
        )

        config = create_config_from_args(args)

        assert config.graph.proximity_threshold == 25.5
        assert config.graph.distance_metric == 'manhattan'


class TestCommandFunctions:
    """Test CLI command functions"""

    @patch('graphizy.cli.generate_data')
    @patch('graphizy.cli.Graphing')
    @patch('graphizy.cli.setup_logging')
    def test_cmd_delaunay(self, mock_setup_logging, mock_graphing_class, mock_generate_data):
        """Test delaunay command execution"""
        # Mock data and objects
        mock_data = Mock()
        mock_generate_data.return_value = mock_data

        mock_grapher = Mock()
        mock_graph = Mock()
        mock_grapher.make_delaunay.return_value = mock_graph
        mock_grapher.get_graph_info.return_value = {
            'vertex_count': 100,
            'edge_count': 250,
            'density': 0.05,
            'average_path_length': 3.2
        }
        mock_grapher.draw_graph.return_value = Mock()
        mock_graphing_class.return_value = mock_grapher

        # Create args
        args = Namespace(
            size=800,
            particles=100,
            verbose=False,
            config=None,
            output='test.jpg',
            show=False,
            line_color='0,255,0',
            point_color='0,0,255',
            line_thickness=1,
            point_radius=8
        )

        # Execute command
        cmd_delaunay(args)

        # Verify calls
        mock_setup_logging.assert_called_once_with(False)
        mock_generate_data.assert_called_once()
        mock_grapher.make_delaunay.assert_called_once_with(mock_data)
        mock_grapher.save_graph.assert_called_once()

    @patch('graphizy.cli.generate_data')
    @patch('graphizy.cli.Graphing')
    @patch('graphizy.cli.setup_logging')
    def test_cmd_proximity(self, mock_setup_logging, mock_graphing_class, mock_generate_data):
        """Test proximity command execution"""
        # Mock data and objects
        mock_data = Mock()
        mock_generate_data.return_value = mock_data

        mock_grapher = Mock()
        mock_graph = Mock()
        mock_grapher.make_proximity.return_value = mock_graph
        mock_grapher.get_graph_info.return_value = {
            'vertex_count': 100,
            'edge_count': 150,
            'density': 0.03,
            'is_connected': True,
            'average_path_length': 2.8
        }
        mock_grapher.draw_graph.return_value = Mock()
        mock_graphing_class.return_value = mock_grapher

        # Create args
        args = Namespace(
            size=600,
            particles=150,
            verbose=True,
            config=None,
            output='proximity.jpg',
            show=True,
            threshold=30.0,
            metric='euclidean',
            line_color='255,0,0',
            point_color='0,255,0',
            line_thickness=2,
            point_radius=10
        )

        # Execute command
        cmd_proximity(args)

        # Verify calls
        mock_setup_logging.assert_called_once_with(True)
        mock_generate_data.assert_called_once()
        mock_grapher.make_proximity.assert_called_once_with(mock_data, 30.0, 'euclidean')
        mock_grapher.save_graph.assert_called_once()
        mock_grapher.show_graph.assert_called_once()

    @patch('graphizy.cli.generate_data')
    @patch('graphizy.cli.Graphing')
    @patch('graphizy.cli.setup_logging')
    def test_cmd_both(self, mock_setup_logging, mock_graphing_class, mock_generate_data):
        """Test both command execution"""
        # Mock data and objects
        mock_data = Mock()
        mock_generate_data.return_value = mock_data

        mock_grapher = Mock()
        mock_del_graph = Mock()
        mock_prox_graph = Mock()
        mock_grapher.make_delaunay.return_value = mock_del_graph
        mock_grapher.make_proximity.return_value = mock_prox_graph
        mock_grapher.get_graph_info.side_effect = [
            {'vertex_count': 100, 'edge_count': 250, 'density': 0.05},
            {'vertex_count': 100, 'edge_count': 180, 'density': 0.036}
        ]
        mock_grapher.draw_graph.side_effect = [Mock(), Mock()]
        mock_graphing_class.return_value = mock_grapher

        # Create args
        args = Namespace(
            size=800,
            particles=100,
            verbose=False,
            config=None,
            output=None,
            show=False,
            threshold=40.0,
            metric='manhattan',
            delaunay_output='del_output.jpg',
            proximity_output='prox_output.jpg',
            line_color='0,0,255',
            point_color='255,255,0',
            line_thickness=1,
            point_radius=8
        )

        # Execute command
        cmd_both(args)

        # Verify calls
        mock_grapher.make_delaunay.assert_called_once_with(mock_data)
        mock_grapher.make_proximity.assert_called_once_with(mock_data, 40.0, 'manhattan')
        assert mock_grapher.save_graph.call_count == 2

    @patch('graphizy.cli.generate_data')
    @patch('graphizy.cli.Graphing')
    @patch('graphizy.cli.setup_logging')
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_cmd_info(self, mock_json_dump, mock_open, mock_setup_logging,
                      mock_graphing_class, mock_generate_data):
        """Test info command execution"""
        # Mock data and objects
        mock_data = Mock()
        mock_generate_data.return_value = mock_data

        mock_grapher = Mock()
        mock_del_graph = Mock()
        mock_prox_graph = Mock()
        mock_grapher.make_delaunay.return_value = mock_del_graph
        mock_grapher.make_proximity.return_value = mock_prox_graph
        mock_grapher.get_graph_info.side_effect = [
            {
                'vertex_count': 100,
                'edge_count': 250,
                'density': 0.05,
                'average_path_length': 3.2,
                'is_connected': True
            },
            {
                'vertex_count': 100,
                'edge_count': 180,
                'density': 0.036,
                'average_path_length': 2.8,
                'is_connected': False
            }
        ]
        mock_graphing_class.return_value = mock_grapher

        # Create args
        args = Namespace(
            size=1000,
            particles=200,
            verbose=True,
            config=None,
            output='summary.json',
            threshold=50.0,
            line_color='128,128,128',
            point_color='64,64,64',
            line_thickness=1,
            point_radius=6
        )

        # Execute command
        cmd_info(args)

        # Verify calls
        mock_grapher.make_delaunay.assert_called_once_with(mock_data)
        mock_grapher.make_proximity.assert_called_once_with(mock_data, 50.0)
        mock_json_dump.assert_called_once()


class TestMainFunction:
    """Test main CLI function"""

    @patch('graphizy.cli.cmd_delaunay')
    @patch('sys.argv', ['graphizy', 'delaunay', '--size', '800'])
    def test_main_delaunay(self, mock_cmd_delaunay):
        """Test main function with delaunay command"""
        main()
        mock_cmd_delaunay.assert_called_once()

    @patch('graphizy.cli.cmd_proximity')
    @patch('sys.argv', ['graphizy', 'proximity', '--threshold', '25'])
    def test_main_proximity(self, mock_cmd_proximity):
        """Test main function with proximity command"""
        main()
        mock_cmd_proximity.assert_called_once()

    @patch('sys.argv', ['graphizy'])
    def test_main_no_command(self):
        """Test main function with no command"""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch('sys.argv', ['graphizy', 'invalid_command'])
    def test_main_invalid_command(self):
        """Test main function with invalid command"""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2

    @patch('graphizy.cli.cmd_delaunay')
    @patch('sys.argv', ['graphizy', 'delaunay'])
    def test_main_keyboard_interrupt(self, mock_cmd_delaunay):
        """Test main function handling keyboard interrupt"""
        mock_cmd_delaunay.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch('graphizy.cli.cmd_delaunay')
    @patch('sys.argv', ['graphizy', 'delaunay'])
    def test_main_unexpected_error(self, mock_cmd_delaunay):
        """Test main function handling unexpected errors"""
        mock_cmd_delaunay.side_effect = Exception("Unexpected error")

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestCLIIntegration:
    """Integration tests for CLI"""

    def test_argument_propagation(self):
        """Test that CLI arguments properly propagate to configuration"""
        parser = create_parser()
        args = parser.parse_args([
            'delaunay',
            '--size', '500',
            '--particles', '75',
            '--line-color', '200,100,50',
            '--point-radius', '12',
            '--verbose'
        ])

        config = create_config_from_args(args)

        # Verify all arguments made it to config
        assert config.graph.dimension == (500, 500)
        assert config.generation.num_particles == 75
        assert config.drawing.line_color == (50, 100, 200)  # BGR conversion
        assert config.drawing.point_radius == 12
        assert config.logging.level == 'DEBUG'

    def test_config_file_override(self):
        """Test configuration file and CLI argument interaction"""
        # Create config file with some settings
        config_data = {
            "drawing": {
                "line_thickness": 5,
                "point_radius": 15
            },
            "graph": {
                "proximity_threshold": 80.0
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            parser = create_parser()
            args = parser.parse_args([
                'proximity',
                '--config', config_file,
                '--size', '600',  # Not in config file
                '--threshold', '25.0',  # Overrides config file
                '--point-radius', '10'  # Overrides config file
            ])

            config = create_config_from_args(args)

            # From config file (not overridden)
            assert config.drawing.line_thickness == 5

            # From CLI (new setting)
            assert config.graph.dimension == (600, 600)

            # From CLI (overrides config file)
            assert config.graph.proximity_threshold == 25.0
            assert config.drawing.point_radius == 10

        finally:
            os.unlink(config_file)

    @patch('graphizy.cli.Graphing')
    @patch('graphizy.cli.generate_data')
    def test_end_to_end_simulation(self, mock_generate_data, mock_graphing_class):
        """Test end-to-end CLI simulation"""
        # Mock the heavy dependencies
        mock_data = Mock()
        mock_generate_data.return_value = mock_data

        mock_grapher = Mock()
        mock_graph = Mock()
        mock_grapher.make_delaunay.return_value = mock_graph
        mock_grapher.get_graph_info.return_value = {
            'vertex_count': 50,
            'edge_count': 120,
            'density': 0.1,
            'average_path_length': None
        }
        mock_image = Mock()
        mock_grapher.draw_graph.return_value = mock_image
        mock_graphing_class.return_value = mock_grapher

        # Test complete workflow
        with patch('sys.argv', ['graphizy', 'delaunay', '--size', '400', '--particles', '50']):
            main()

        # Verify the workflow executed
        mock_generate_data.assert_called_once()
        mock_grapher.make_delaunay.assert_called_once()
        mock_grapher.get_graph_info.assert_called_once()
        mock_grapher.draw_graph.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
