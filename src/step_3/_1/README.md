# 3.1

The python file for this is `full_normalization.py`

It runs the four normalization steps on every shape, in this order:

1. translation: the barycenter is moved to the origin,
2. alignment: the shape is rotated (PCA) so that its direction of largest spread
   is the x axis and the second largest the y axis,
3. flipping: an axis is mirrored when its moment test is negative, so that the
   bulk of the shape is on the positive side of every axis,
4. scaling: the shape is scaled so that the longest side of its bounding box is 1.

Scaling is last on purpose. The bounding box depends on the orientation of the
shape, so scaling before aligning would give copies of the same shape that are
rotated differently different sizes.

## How to run

You run it from project root directory. Examples:

One shape:
`python -m src.step_3._1.full_normalization -i ./3d_models/Hand/D01193.obj -o ./3d_models_full_normalized/Hand/D01193.obj`

A whole folder (the class sub-folders are kept in the output):
`python -m src.step_3._1.full_normalization -i ./3d_models_reduced -o ./3d_models_full_normalized`

The -i flag is a single _.obj file_ or a _folder_, the -o flag is where the
result is written. For a folder it prints a summary and then re-reads the saved
files and recomputes, for every shape: the barycenter, the longest side, that
the covariance matrix is diagonal (the axes are the principal axes), that the
variance is ordered x >= y >= z and that the moment test is non-negative on
every axis. Each line ends with a count such as `276/276`, which should equal
the number of shapes. `./3d_models_reduced` takes about 30 seconds. Use
`./resampled_models` (the result of 2.3) as input for the final results.