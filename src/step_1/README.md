# viewer_runner.py

```viewer_runner.py``` uses the ```ModelViewer``` class from ```model_viewer.py``` and actually runs the application. See this example to see how you can use the ```ModelViewer``` in your code.

You can run it from the root directory with:  
```python -m src.step_1.viewer_runner ./3d_models/AquaticAnimal/m59.obj ./resampled_models/AquaticAnimal/m59.obj -d 0.0 0.5 0```  
mesh_paths = One or more 3D `.obj` shape paths you want to load and render.  
-d / --delta = Is the delta (X Y Z offset) you want between the shapes. For example 0.0 0.5 0.0 will result in a delta of 0.5 in the Y axis.

# model_viewer.py

```model_viewer.py``` has the ```ModelViewer``` class that handles model loading and rendering.
