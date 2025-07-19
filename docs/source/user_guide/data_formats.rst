Data Formats
============

This guide explains the data formats that Graphizy accepts and how to structure your data for optimal performance.

Overview
--------

Graphizy accepts data in two primary formats:

1. **Array Format** (``aspect="array"``) - NumPy arrays with structured columns
2. **Dictionary Format** (``aspect="dict"``) - Python dictionaries with named keys

Both formats represent the same information: a collection of objects with IDs and 2D coordinates.

Array Format (aspect="array")
-----------------------------

**Structure:**
The array format uses a 2D NumPy array where each row represents one object and columns contain the object's attributes.

**Required Columns:**
- Column 0: Object ID (numeric)
- Column 1: X coordinate 
- Column 2: Y coordinate
- Columns 3+: Additional attributes (optional)

**Basic Example:**

.. code-block:: python

   import numpy as np
   from graphizy import Graphing

   # Basic format: [id, x, y]
   data = np.array([
       [0, 100, 200],    # Object 0 at position (100, 200)
       [1, 300, 400],    # Object 1 at position (300, 400)
       [2, 500, 600],    # Object 2 at position (500, 600)
   ])

   # Create grapher and use the data
   grapher = Graphing(dimension=(800, 800), aspect="array")
   graph = grapher.make_delaunay(data)

**Extended Example with Additional Attributes:**

.. code-block:: python

   # Extended format: [id, x, y, speed, active, type]
   data = np.array([
       [0, 100, 200, 1.5, 1, 0],    # Object 0: speed=1.5, active=True, type=0
       [1, 300, 400, 2.3, 1, 1],    # Object 1: speed=2.3, active=True, type=1
       [2, 500, 600, 0.8, 0, 0],    # Object 2: speed=0.8, active=False, type=0
   ])

   # Graphizy will use columns 0, 1, 2 for id, x, y
   # Additional columns are preserved but not used for graph creation

Dictionary Format (aspect="dict")
---------------------------------

**Structure:**
The dictionary format uses a Python dictionary with three required keys, each containing a list of values.

**Required Keys:**
- ``"id"``: List of object IDs (numeric)
- ``"x"``: List of X coordinates
- ``"y"``: List of Y coordinates

**Basic Example:**

.. code-block:: python

   # Dictionary format
   data = {
       "id": [0, 1, 2],
       "x": [100, 300, 500],
       "y": [200, 400, 600]
   }

   # Create grapher and use the data
   grapher = Graphing(dimension=(800, 800), aspect="dict")
   graph = grapher.make_delaunay(data)

**Extended Example with Additional Attributes:**

.. code-block:: python

   # Dictionary with additional attributes
   data = {
       "id": [0, 1, 2, 3],
       "x": [100, 300, 500, 700],
       "y": [200, 400, 600, 800],
       "speed": [1.5, 2.3, 0.8, 1.9],
       "color": ["red", "blue", "green", "yellow"],
       "active": [True, True, False, True],
       "category": ["A", "B", "A", "C"]
   }

   # Graphizy will use id, x, y for graph creation
   # Additional keys are preserved for your use

Converting Between Formats
--------------------------

**Array to Dictionary:**

.. code-block:: python

   def array_to_dict(data_array):
       """Convert array format to dictionary format"""
       return {
           "id": data_array[:, 0].tolist(),
           "x": data_array[:, 1].tolist(),
           "y": data_array[:, 2].tolist()
       }

   # Example usage
   array_data = np.array([[0, 100, 200], [1, 300, 400]])
   dict_data = array_to_dict(array_data)

**Dictionary to Array:**

.. code-block:: python

   def dict_to_array(data_dict):
       """Convert dictionary format to array format"""
       return np.column_stack([
           data_dict["id"],
           data_dict["x"], 
           data_dict["y"]
       ])

   # Example usage
   dict_data = {"id": [0, 1], "x": [100, 300], "y": [200, 400]}
   array_data = dict_to_array(dict_data)

Common Data Sources
-------------------

**From CSV Files:**

.. code-block:: python

   import pandas as pd

   # Read CSV file
   df = pd.read_csv("objects.csv")  # columns: object_id, pos_x, pos_y
   
   # Convert to array format
   data_array = df[["object_id", "pos_x", "pos_y"]].values
   
   # Or convert to dictionary format
   data_dict = {
       "id": df["object_id"].tolist(),
       "x": df["pos_x"].tolist(),
       "y": df["pos_y"].tolist()
   }

**From Object Detection:**

.. code-block:: python

   # From YOLO or similar detection systems
   def detections_to_graphizy(detections):
       """Convert detection results to graphizy format"""
       data = []
       for i, detection in enumerate(detections):
           x_center, y_center = detection[0], detection[1]
           data.append([i, x_center, y_center])
       return np.array(data)

**From Simulation Systems:**

.. code-block:: python

   # From particle simulation
   def particles_to_graphizy(particles, include_velocity=False):
       """Convert particle objects to graphizy format"""
       if include_velocity:
           return np.array([
               [p.id, p.x, p.y, p.vx, p.vy] for p in particles
           ])
       else:
           return np.array([
               [p.id, p.x, p.y] for p in particles
           ])

Data Validation
---------------

Always validate your data before creating graphs:

.. code-block:: python

   from graphizy import validate_graphizy_input

   # Validate your data
   result = validate_graphizy_input(
       data, 
       aspect="array",           # or "dict"
       dimension=(800, 800),
       verbose=True
   )

   if not result["valid"]:
       print("Data issues found:")
       for error in result["errors"]:
           print(f"  - {error}")

For complete validation details, see the :doc:`data_validation` guide.

Best Practices
--------------

1. **Use numeric IDs only** - String IDs will cause errors
2. **Ensure coordinates fit within dimensions** - Points outside bounds will generate warnings  
3. **Choose array format for large datasets** - Better memory efficiency
4. **Choose dictionary format for mixed data types** - More readable and flexible
5. **Always validate data before graph creation** - Catch issues early

Performance Tips
----------------

.. code-block:: python

   # For large datasets, use appropriate data types
   large_data = np.random.randint(0, 1000, (10000, 3), dtype=np.int32)
   
   # Array format is generally faster for large datasets
   grapher = Graphing(aspect="array", dimension=(1000, 1000))
