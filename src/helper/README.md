# Helper
## Prepare database
File is `prepare_database.py`. 

### How to run
Run it from the project root using `python -m src.helper.prepare_database -i path/to/raw_database -o path/to/output_directory`. The required `-i` flag specifies the raw 3D-model database directory, while the optional `-o` flag specifies the output directory, defaulting to `3d_models_prepared`. The output directory must be outside the raw database directory.

The pipeline saves a checkpoint after each successful step, allowing interrupted runs to resume by skipping completed steps with valid outputs. Use `--reset_checkpoint` (or `--reset_pickle`) to restart the entire pipeline, or `--reset_last_step` to rerun the last unfinished or completed step and all subsequent steps. These options are mutually exclusive and require confirmation.
