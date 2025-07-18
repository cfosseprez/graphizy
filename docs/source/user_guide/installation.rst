Installation
============

Graphizy can be installed via pip or from source for development purposes.

Requirements
------------

**System Requirements:**
- Python >= 3.8
- Operating System: Windows, macOS, or Linux

**Required Dependencies:**
- NumPy >= 1.20.0
- OpenCV >= 4.5.0  
- python-igraph >= 0.9.0
- SciPy >= 1.7.0

**Optional Dependencies:**
- matplotlib (for plotting time series)
- pandas (for data analysis examples)
- pytest (for running tests)

Installation Methods
--------------------

**Standard Installation:**

.. code-block:: bash

   pip install graphizy

**Development Installation:**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/cfosseprez/graphizy.git
   cd graphizy
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -e ".[dev]"

**Verify Installation:**

.. code-block:: python

   import graphizy
   from graphizy import Graphing, generate_positions
   import numpy as np
   
   print(f"Graphizy version: {graphizy.__version__}")
   
   # Quick test
   positions = generate_positions(100, 100, 10)
   data = np.column_stack((np.arange(len(positions)), positions))
   grapher = Graphing(dimension=(100, 100))
   graph = grapher.make_delaunay(data)
   print(f"Test successful: Created graph with {graph.vcount()} vertices")

Troubleshooting
---------------

**Common Issues:**

1. **OpenCV Installation Problems:**
   
   .. code-block:: bash
   
      # Try different OpenCV package
      pip uninstall opencv-python
      pip install opencv-python-headless

2. **igraph Installation on Windows:**
   
   .. code-block:: bash
   
      # Use conda for easier igraph installation
      conda install python-igraph

3. **SciPy/NumPy Conflicts:**
   
   .. code-block:: bash
   
      # Reinstall scientific stack
      pip install --upgrade numpy scipy

**Platform-Specific Notes:**

**macOS:**
   - May need to install Xcode command line tools
   - Use Homebrew for system dependencies if needed

**Linux:**
   - Install system packages: ``sudo apt-get install python3-dev``
   - For igraph: ``sudo apt-get install libigraph0-dev``

**Windows:**
   - Use Anaconda/Miniconda for easier dependency management
   - Visual Studio Build Tools may be required for some packages

**Docker Installation:**

.. code-block:: dockerfile

   FROM python:3.9-slim
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       libgl1-mesa-glx \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender-dev \
       libgomp1
   
   # Install graphizy
   RUN pip install graphizy
   
   # Verify installation
   RUN python -c "import graphizy; print('Graphizy installed successfully')"

Getting Started
---------------

After installation, check out the :doc:`graph_types` guide to understand the different types of graphs you can create, or jump straight into the :doc:`examples` for hands-on tutorials.
