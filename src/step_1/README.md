# viewer_example.py

```view_example.py``` uses the ```ModelViewer``` class from ```model_viewer.py``` and actually runs the application. See this example to see how you can use the ```ModeViewer``` in your code.

You can run it from the root directory with:  
```python -m src.step_1.viewer_runner ./3d_models/AquaticAnimal/m59.obj ./resampled_models/AquaticAnimal/m59.obj -d 0.0 0.5 0```  
3d_shape_path = the 3Ds shape you want to load and render  
-d = Is the delta you want between the shapes. For example 0.0 0.5 0.0 will result in a delta of 0.5 in the Y axis.  
-b = The folder shown in the content browser (default: ```./3d_models```).

The shapes are optional, so ```python -m src.step_1.viewer_runner``` starts an empty viewer and you pick a shape in the content browser.

## What is in the viewer
- **Content browser** (right): a tree of all .obj files. Double-click or Enter opens a shape and replaces the current one. A file that cannot be loaded gives a message in the status bar and the current shape stays.
- **Camera** (left): the camera mode and the **FOV** slider (10 to 120 degrees).
- **Edges, Vertices, Shading**: width of the edges, size of the vertices and smooth or flat shading. They stay as set when a new shape is opened.
- **FPS** (status bar, bottom right): the frames per second while you move the camera, ```idle``` when nothing moves, and the time of the last frame.

# model_viewer.py

```model_viewer.py``` has the ```ModelViewer``` class that handles model loading and rendering. ```load_shapes([paths])``` replaces the shapes on screen, which is what the content browser uses.