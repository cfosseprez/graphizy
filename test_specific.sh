#!/bin/bash
# Quick test runner to validate our fix

echo "Testing ID Normalization Fix for Graphizy"
echo "=========================================="

# Set directory
cd "C:\\Users\\nakag\\Desktop\\code\\graphizy"

echo "Running comprehensive test..."
python test_fix_comprehensive.py

echo ""
echo "Running specific failing tests..."
echo "1. Testing create_memory_graph_array test:"
python -m pytest tests/test_memory_graph.py::TestMemoryGraphCreation::test_create_memory_graph_array -v

echo ""
echo "2. Testing update_memory_from_proximity test:"
python -m pytest tests/test_memory_graph.py::TestMemoryUpdates::test_update_memory_from_proximity -v

echo ""
echo "3. Testing make_memory_graph_with_explicit_connections test:"
python -m pytest tests/test_memory_graph.py::TestGraphingIntegration::test_make_memory_graph_with_explicit_connections -v

echo ""
echo "If all tests pass, the fix is successful!"
