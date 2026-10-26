#!/bin/bash

uv run -m src.step_4.distance_calculations.plot_differences --raw 3d_models_prepared/analysis/simple_descriptors.csv --processed 3d_models_prepared/analysis/simple_descriptors_processed.csv --bins 50

uv run -m src.step_4.distance_calculations.plot_differences --raw 3d_models_prepared/analysis/simple_descriptors.csv --processed 3d_models_prepared/analysis/simple_descriptors_processed.csv --bins 50 --descriptor eccentricity
uv run -m src.step_4.distance_calculations.plot_differences --raw 3d_models_prepared/analysis/simple_descriptors.csv --processed 3d_models_prepared/analysis/simple_descriptors_processed.csv --bins 50 --descriptor rectangularity
uv run -m src.step_4.distance_calculations.plot_differences --raw 3d_models_prepared/analysis/simple_descriptors.csv --processed 3d_models_prepared/analysis/simple_descriptors_processed.csv --bins 50 --descriptor surface_area
uv run -m src.step_4.distance_calculations.plot_differences --raw 3d_models_prepared/analysis/simple_descriptors.csv --processed 3d_models_prepared/analysis/simple_descriptors_processed.csv --bins 50 --descriptor convexity
uv run -m src.step_4.distance_calculations.plot_differences --raw 3d_models_prepared/analysis/simple_descriptors.csv --processed 3d_models_prepared/analysis/simple_descriptors_processed.csv --bins 50 --descriptor compactness