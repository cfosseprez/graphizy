Data Validation
===============

Graphizy provides comprehensive input validation to help you debug data format issues and ensure optimal performance. This guide covers the validation function and common data problems you might encounter.

Quick Validation
----------------

Use the built-in validation function to check your data before creating graphs:

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
   
   if result["valid"]:
       print("✅ Data is ready!")
   else:
       print("❌ Issues found:")
       for error in result["errors"]:
           print(f"  - {error}")

Validation Function Reference
-----------------------------

**Function Signature:**

.. code-block:: python

   validate_graphizy_input(
       data_points,                    # Your data (array or dict)
       aspect="array",                 # "array" or "dict"
       data_shape=None,               # Expected data structure
       dimension=(1200, 1200),        # Image dimensions (width, height)
       proximity_thresh=None,         # Proximity threshold if applicable
       verbose=True                   # Print detailed results
   )

**Return Value:**

The function returns a dictionary with:

.. code-block:: python

   {
       "valid": True/False,           # Overall validity
       "errors": [],                  # List of error messages
       "warnings": [],                # List of warning messages
       "info": {},                    # Data information (shape, ranges, etc.)
       "suggestions": []              # Performance and usage suggestions
   }

Data Format Requirements
------------------------

**Array Format (aspect="array"):**

Your data should be a 2D NumPy array with at least 3 columns:

.. code-block:: python

   # ✅ Correct format: [id, x, y, additional_columns...]
   data = np.array([
       [0, 100, 200],      # object 0 at (100, 200)
       [1, 300, 400],      # object 1 at (300, 400)
       [2, 500, 600]       # object 2 at (500, 600)
   ])
   
   # ✅ With additional columns is fine
   data = np.array([
       [0, 100, 200, 1.5, True],   # [id, x, y, speed, active]
       [1, 300, 400, 2.0, False],
       [2, 500, 600, 1.8, True]
   ])

**Dictionary Format (aspect="dict"):**

Your data should be a dictionary with required keys:

.. code-block:: python

   # ✅ Correct format
   data = {
       "id": [0, 1, 2],
       "x": [100, 300, 500],
       "y": [200, 400, 600]
   }
   
   # ✅ Additional keys are fine
   data = {
       "id": [0, 1, 2],
       "x": [100, 300, 500],
       "y": [200, 400, 600],
       "speed": [1.5, 2.0, 1.8],
       "color": ["red", "blue", "green"]
   }

Common Data Issues and Solutions
--------------------------------

**1. String IDs (Most Common Issue)**

**Problem:** Using string identifiers instead of numeric ones.

.. code-block:: python

   # ❌ WRONG - This will cause errors
   bad_data = np.array([
       ["particle_1", 100, 200],
       ["particle_2", 300, 400]
   ])
   
   # Error: "Object IDs must be numeric, not string type"

**Solution:** Use numeric IDs:

.. code-block:: python

   # ✅ CORRECT
   good_data = np.array([
       [0, 100, 200],      # Use 0, 1, 2... or any numeric IDs
       [1, 300, 400]
   ], dtype=int)
   
   # Or convert strings to numbers
   string_ids = ["particle_1", "particle_2", "particle_3"]
   numeric_ids = list(range(len(string_ids)))  # [0, 1, 2]

**2. Wrong Array Dimensions**

**Problem:** 1D arrays or wrong shapes.

.. code-block:: python

   # ❌ WRONG - 1D array
   bad_data = np.array([1, 2, 3, 4, 5, 6])
   
   # ❌ WRONG - 3D array
   bad_data = np.array([[[1, 2, 3]]])

**Solution:** Use 2D arrays:

.. code-block:: python

   # ✅ CORRECT - Reshape if needed
   data_1d = np.array([0, 100, 200, 1, 300, 400])
   good_data = data_1d.reshape(-1, 3)  # Reshape to 2D
   
   # Result: [[0, 100, 200], [1, 300, 400]]

**3. Insufficient Columns**

**Problem:** Less than 3 columns (need at least id, x, y).

.. code-block:: python

   # ❌ WRONG - Only 2 columns
   bad_data = np.array([[0, 100], [1, 200]])

**Solution:** Add the missing coordinate:

.. code-block:: python

   # ✅ CORRECT - Add missing y coordinates
   x_coords = np.array([[0, 100], [1, 200]])
   y_coords = np.random.randint(0, 400, (len(x_coords), 1))
   good_data = np.column_stack([x_coords, y_coords])
   
   # Or create from scratch
   good_data = np.array([[0, 100, 150], [1, 200, 250]])

**4. Coordinates Outside Bounds**

**Problem:** Points outside the defined image dimensions.

.. code-block:: python

   # ❌ PROBLEMATIC - x=1300 exceeds dimension width of 1200
   data = np.array([[0, 1300, 200]])
   
   # Warning: "X coordinates outside dimension bounds [0, 1200)"

**Solutions:**

.. code-block:: python

   # Option 1: Clip coordinates to bounds
   data[:, 1] = np.clip(data[:, 1], 0, 1199)  # x coordinates
   data[:, 2] = np.clip(data[:, 2], 0, 1199)  # y coordinates
   
   # Option 2: Scale coordinates to fit
   def scale_to_fit(data, dimension):
       x_min, x_max = data[:, 1].min(), data[:, 1].max()
       y_min, y_max = data[:, 2].min(), data[:, 2].max()
       
       # Scale x coordinates
       if x_max > x_min:
           data[:, 1] = (data[:, 1] - x_min) / (x_max - x_min) * (dimension[0] - 1)
       
       # Scale y coordinates  
       if y_max > y_min:
           data[:, 2] = (data[:, 2] - y_min) / (y_max - y_min) * (dimension[1] - 1)
       
       return data
   
   scaled_data = scale_to_fit(data, (1200, 1200))
   
   # Option 3: Increase image dimensions
   larger_dimension = (1500, 1500)  # Make room for all points

**5. Dictionary Format Issues**

**Problem:** Missing keys or mismatched array lengths.

.. code-block:: python

   # ❌ WRONG - Missing 'y' key
   bad_data = {"id": [0, 1], "x": [100, 300]}
   
   # ❌ WRONG - Mismatched lengths
   bad_data = {
       "id": [0, 1, 2],        # 3 elements
       "x": [100, 300],        # 2 elements
       "y": [200, 400, 600]    # 3 elements
   }

**Solution:**

.. code-block:: python

   # ✅ CORRECT - All required keys with matching lengths
   good_data = {
       "id": [0, 1, 2],
       "x": [100, 300, 500],
       "y": [200, 400, 600]
   }
   
   # Fix mismatched lengths by trimming or padding
   def fix_dict_lengths(data_dict):
       min_length = min(len(v) for v in data_dict.values())
       return {k: v[:min_length] for k, v in data_dict.items()}
   
   fixed_data = fix_dict_lengths(bad_data)

Best Practices
--------------

**Performance Tips:**

1. **Use appropriate data types:**
   
   .. code-block:: python
   
      # Use int32 for coordinates if possible (saves memory)
      data = np.array(coordinates, dtype=np.int32)
      
      # Use float32 instead of float64 for large datasets
      data = data.astype(np.float32)

2. **Validate early and often:**
   
   .. code-block:: python
   
      # Validate immediately after loading data
      def load_and_validate(filename):
          data = np.loadtxt(filename)  # or your loading method
          result = validate_graphizy_input(data, verbose=False)
          
          if not result["valid"]:
              raise ValueError(f"Invalid data: {result['errors']}")
          
          return data

3. **Handle large datasets efficiently:**
   
   .. code-block:: python
   
      # For very large datasets, validate a sample first
      def validate_large_dataset(data, sample_size=1000):
          if len(data) > sample_size:
              sample_indices = np.random.choice(len(data), sample_size, replace=False)
              sample_data = data[sample_indices]
              result = validate_graphizy_input(sample_data, verbose=False)
              
              if not result["valid"]:
                  print("❌ Sample validation failed - likely issues in full dataset")
                  return result
              
              print(f"✅ Sample of {sample_size} points validated successfully")
          
          return validate_graphizy_input(data, verbose=True)

**Integration with Graphizy Workflow:**

.. code-block:: python

   def safe_graphizy_workflow(data, graph_type="delaunay"):
       """Complete safe workflow with validation"""
       
       # Step 1: Validate
       result = validate_graphizy_input(data, verbose=False)
       if not result["valid"]:
           print("❌ Validation failed:")
           for error in result["errors"]:
               print(f"  - {error}")
           return None
       
       # Step 2: Create grapher
       dimension = result["info"].get("dimension", (1200, 1200))
       grapher = Graphing(dimension=dimension)
       
       # Step 3: Create graph
       try:
           if graph_type == "delaunay":
               graph = grapher.make_delaunay(data)
           elif graph_type == "proximity":
               graph = grapher.make_proximity(data, proximity_thresh=50.0)
           elif graph_type == "mst":
               graph = grapher.make_mst(data)
           else:
               raise ValueError(f"Unknown graph type: {graph_type}")
           
           print(f"✅ Successfully created {graph_type} graph with {graph.vcount()} vertices")
           return graph, grapher
           
       except Exception as e:
           print(f"❌ Graph creation failed: {e}")
           return None

   # Use the safe workflow
   result = safe_graphizy_workflow(my_data, "delaunay")
   if result:
       graph, grapher = result
       image = grapher.draw_graph(graph)
       grapher.show_graph(image)

This validation system helps ensure your data works perfectly with Graphizy and provides clear guidance when issues arise. Always validate your data first - it will save you time debugging later!
