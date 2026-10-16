# Reduced database
File: ```reduced_database.py```

Copies a small, class-balanced part of the database (4 shapes per class, 276
shapes in total) so the other steps can be tested quickly. It uses a fixed
random seed, so everybody gets exactly the same shapes from their own
`3d_models` folder.

## How to run
From the project root:

```
python -m src.step_2.reduced_database -i ./3d_models -o ./3d_models_reduced
```

Optional: `-n` sets the number of shapes per class and `--seed` the random seed.

The instructions for each step of 2 are in the readme of its folder, for
example `src/step_2/_5/README.md` for 2.5.
