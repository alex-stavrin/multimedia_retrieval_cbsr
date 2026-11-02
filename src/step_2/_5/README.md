# 2.5
File: ```normalizer.py```

Moves the barycenter of every shape to the origin and scales it so that the
longest side of its bounding box is 1.

## How to run
From the project root:

```
python -m src.step_2._5.normalizer -i ./3d_models_reduced -o ./3d_models_normalized
```

- `-i` is a folder of .obj files (or a single .obj file), `-o` is where the
  result is written.
- Add `--verify` (`-v`) to re-check every saved shape afterwards. `276/276` means
  all 276 shapes passed.
- The shapes are processed in parallel, one process per CPU core, so the whole
  database (2,483 shapes) takes about 30 seconds and the reduced one under 10.
  `-j 1` turns this off, `-j 4` uses 4 processes.
- Use `./resampled_models` (the result of 2.3) for the final results.

One shape:

```
python -m src.step_2._5.normalizer -i ./3d_models/Cup/D00035.obj -o ./3d_models_normalized/Cup/D00035.obj
```
