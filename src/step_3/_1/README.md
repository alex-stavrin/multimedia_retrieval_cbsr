# 3.1
File: ```full_normalization.py```

Normalizes the position, orientation and size of every shape in four steps, in
this order:

1. **center**: the barycenter goes to the origin.
2. **align**: the shape is rotated (PCA) so that its longest direction is the
   x axis and the second longest the y axis.
3. **flip**: an axis is mirrored when its moment test is negative, so the bulk
   of the shape is on the positive side.
4. **scale**: the longest side of the bounding box becomes 1. This is last on
   purpose, because the bounding box depends on the rotation.

## How to run
From the project root:

```
python -m src.step_3._1.full_normalization -i ./3d_models_reduced -o ./3d_models_full_normalized
```

- `-i` is a folder of .obj files (or a single .obj file), `-o` is where the
  result is written.
- The last lines re-check every saved shape (barycenter, size, axes, flip).
  Each line ends with a count like `276/276`, which should equal the number of
  shapes.
- The reduced database takes about 30 seconds. Use `./resampled_models` (the
  result of 2.3) for the final results.

One shape:

```
python -m src.step_3._1.full_normalization -i ./3d_models/Hand/D01193.obj -o ./3d_models_full_normalized/Hand/D01193.obj
```
