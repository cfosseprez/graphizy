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

**Input Data Validation:**

Graphizy provides a built-in validation function to help debug input data issues:

.. code-block:: python

   from graphizy import validate_graphizy_input
   import numpy as np
   
   # Your data
   data = np.array([
       [0, 100, 200],
       [1, 300, 400],
       [2, 500, 600]
   ])
   
   # Validate your input
   result = validate_graphizy_input(
       data, 
       aspect="array",           # or "dict"
       dimension=(800, 800),     # your image dimensions
       proximity_thresh=50.0,    # if using proximity graphs
       verbose=True              # print detailed results
   )
   
   if not result["valid"]:
       print("Input validation errors:")
       for error in result["errors"]:
           print(f"  - {error}")
   else:
       print("Input data is valid!")
       print(f"Found {result['info']['num_points']} points")

**Common Data Issues:**

1. **String IDs Error**: ``Object IDs must be numeric, not string type``
   
   .. code-block:: python
   
      # ❌ Wrong - string IDs cause issues
      bad_data = np.array([
          ["obj1", 100, 200],
          ["obj2", 300, 400]
      ])
      
      # ✅ Correct - numeric IDs
      good_data = np.array([
          [0, 100, 200],
          [1, 300, 400]
      ], dtype=int)

2. **Wrong Data Dimensions**: ``Data array must be 2D``
   
   .. code-block:: python
   
      # ❌ Wrong - 1D array
      bad_data = np.array([1, 2, 3])
      
      # ✅ Correct - 2D array with [id, x, y]
      good_data = np.array([[0, 100, 200]])

3. **Insufficient Columns**: ``Data array needs at least 3 columns (id, x, y)``
   
   .. code-block:: python
   
      # ❌ Wrong - only 2 columns
      bad_data = np.array([[0, 100], [1, 200]])
      
      # ✅ Correct - at least 3 columns
      good_data = np.array([[0, 100, 200], [1, 300, 400]])

4. **Coordinates Outside Bounds**: ``X coordinates outside dimension bounds``
   
   .. code-block:: python
   
      # Coordinates should be within [0, dimension)
      data = np.array([[0, 1300, 200]])  # x=1300 > dimension[0]=1200
      
      # Fix by clipping or scaling
      data[:, 1] = np.clip(data[:, 1], 0, 1199)

5. **Dictionary Format Issues**: For ``aspect="dict"``
   
   .. code-block:: python
   
      # ✅ Correct dictionary format
      dict_data = {
          "id": [0, 1, 2],
          "x": [100, 300, 500],
          "y": [200, 400, 600]
      }
      
      # All arrays must have the same length
      validate_graphizy_input(dict_data, aspect="dict")

**Quick Validation Workflow:**

.. code-block:: python

   def debug_my_data(data, aspect="array"):
       """Quick debugging helper for your data"""
       result = validate_graphizy_input(data, aspect=aspect, verbose=True)
       
       if result["valid"]:
           print("✅ Data is ready for graphizy!")
           return True
       else:
           print("❌ Please fix these issues:")
           for error in result["errors"]:
               print(f"   • {error}")
           return False
   
   # Use it before creating graphs
   if debug_my_data(my_data):
       grapher = Graphing()
       graph = grapher.make_delaunay(my_data)

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
