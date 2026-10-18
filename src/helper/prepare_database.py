import src.step_2._3.resampling as resampling
import src.step_2._1.shapes_analyzer as shape_analyzer
import src.step_2._5.normalizer as normalizer
import step_2.standardization.standardize_wireframe as standardisation
import step_2.standardization.standardization_analyzer as standardisation_analyser
import src.step_3._1.full_normalization as full_normalisation
import src.step_3._2.descriptor_analyzer as descriptor_analyzer
import pickle
from pathlib import Path
import os


def prepare_database(database_path: str, output_path: str = ""):
    if output_path == "":
        output_path = "3d_models_prepared"

    output_databases_path = os.path.join(output_path, "3d_models_steps")

    # ---
    # Analyse shapes data

    print("---\nAnalysing raw database...\n---\n")
    analysis_path: str = os.path.join(output_path, "analysis")
    shape_analysis_path_raw: str = os.path.join(
        analysis_path, "raw_shape_analysis.csv"
    )
    shape_analyzer.analyze_database(database_path, shape_analysis_path_raw)

    # ---
    # Resample shapes

    print("---\nResampling database...\n---\n")
    database_path_as_path = Path(database_path)
    shape_analysis_path_as_path = Path(analysis_path)

    resample_output_path = os.path.join(
        output_databases_path, "1_models_resampled"
    )
    resample_output_path_as_path = Path(resample_output_path)

    resampling.resample_meshes(
        models_path=database_path_as_path,
        input_csv_path=shape_analysis_path_as_path,
        output_path=resample_output_path_as_path,
    )

    # Analysis the resampled database
    print("---\nAnalysing resampled database...\n---\n")
    shape_analysis_resampled_path = "resampled_shape_analysis.csv"
    shape_analyzer.analyze_database(
        resample_output_path, shape_analysis_resampled_path
    )

    # ---
    # Standardizing datbase

    print("---\nStandardising database...\n---\n")
    standardised_output_path = os.path.join(
        output_databases_path, "2_models_standardized"
    )
    standardisation.standardize_directory(
        resample_output_path, standardised_output_path
    )

    # Analise standardization
    print("---\nAnalysing standardisation...\n---\n")
    standardisation_analysis_path = os.path.join(
        analysis_path, "standardization_analysis.csv"
    )
    standardisation_analyser.compare_hole_fill_stats(
        resample_output_path,
        standardised_output_path,
        standardisation_analysis_path,
    )

    # ---
    # Normalise shapes

    print("---\nSimple normalisation of database...\n---\n")
    normalize_output_path = os.path.join(
        output_databases_path, "3_models_normalized"
    )
    normalizer.center_and_scale_database(
        standardised_output_path, normalize_output_path
    )

    # ---
    # Fully nurmalise shapes

    print("---\nFull normalisation of database...\n---\n")
    full_normalize_output_path = os.path.join(
        output_databases_path, "4_models_fully_normalized"
    )
    full_normalisation.full_normalize_database(
        input_root=normalize_output_path,
        output_root=full_normalize_output_path,
    )

    # ---
    # Calculate descriptors

    print("---\nFull normalisation of database...\n---\n")
    descriptor_analysis_path = os.path.join(analysis_path, "descriptors.csv")
    descriptor_analyzer.calculate_database(
        input_dir_path=full_normalize_output_path,
        output_path=descriptor_analysis_path,
    )


def main():
    resampling.resample_meshes


if __name__ == "__main__":
    main()
