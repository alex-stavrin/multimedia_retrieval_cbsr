"""Resumable 3D-model preparation pipeline.

A step is checkpointed only after success. Before continuing past a database
step, every raw input file must exist in that step's output at the same
relative path. Extra output files are allowed. This assumes steps preserve
filenames and directory structure.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import Callable, Any

import src.helper.silence_vtk  # noqa: F401
import src.step_2._3.resampling as resampling
import src.step_2._1.shapes_analyzer as shape_analyzer
import src.step_2._3.repairment.repair_wireframe as reparation
import src.step_2._3.repairment.repair_analyzer as rep_analyser
import src.step_3._1.full_normalization as full_normalisation
import src.step_3._2.elementary_descriptors.descriptor_analyzer as descriptor_analyzer
import src.step_3._2.histogram_descriptors.analyze_and_save as histogram_descriptor_analyzer
import src.step_4.distance_calculations.normalize_descriptors as normalize_descriptors
import src.step_5.DR.tsne_engine as dr_engine
from src.step_5.DR.scatterplot import Scatterplot
import src.step_6.run_and_evaluate_knn as evaluate_knn
import src.step_6.run_and_evaluate_rnn as evaluate_rnn
import src.step_6.run_and_evaluate_weighted as evaluate_weighted
import src.step_6.run_and_evaluate_all as evaluate_all
from src.helper.database_functions import verify_database_complete

CHECKPOINT_VERSION = 1

# Keep this order synchronized with the steps in prepare_database().
STEP_ORDER = [
    "analyze_raw_database",
    "repair_database",
    "analyze_reparation",
    "reparation_shape_analysis",
    "resample_database",
    "analyze_resampled_database",
    "fully_normalize_database",
    "fully_normalized_shape_analysis",
    "calculate_descriptors",
    "normalize_descriptors",
    "calculate_histogram_descriptor",
    "dimensionality_reduction_tsne",
    "run_and_evaluate_knn",
    "run_and_evaluate_rnn",
    "run_and_evaluate_weighted",
    "run_and_evaluate_all",
]


def _save_checkpoint(state: dict, path: Path) -> None:
    """Atomically save the pickle checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _load_checkpoint(path: Path, raw_path: Path) -> dict:
    if not path.exists():
        return {
            "version": CHECKPOINT_VERSION,
            "raw_database_path": str(raw_path.resolve()),
            "completed_steps": {},
            "current_step": None,
        }
    try:
        with path.open("rb") as f:
            state = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read checkpoint {path}. Move/delete it to start fresh."
        ) from exc

    if state.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(
            "Incompatible checkpoint version; move/delete the checkpoint."
        )
    if state.get("raw_database_path") != str(raw_path.resolve()):
        raise RuntimeError(
            "Checkpoint belongs to a different raw database: "
            f"{state.get('raw_database_path')}"
        )
    state.setdefault("completed_steps", {})
    state.setdefault("current_step", None)
    return state


def _reset_last_step(state: dict, checkpoint_path: Path) -> None:
    """Invalidate the last unfinished/completed step and every later step."""
    completed = state["completed_steps"]
    current_step = state.get("current_step")

    if current_step in STEP_ORDER and completed.get(current_step) is not True:
        target_index = STEP_ORDER.index(current_step)
    else:
        completed_indices = [
            STEP_ORDER.index(name)
            for name in STEP_ORDER
            if completed.get(name) is True
        ]
        if not completed_indices:
            print("[RESET] No completed step found; nothing to reset.")
            return
        target_index = max(completed_indices)

    target_name = STEP_ORDER[target_index]
    for name in STEP_ORDER[target_index:]:
        completed[name] = False

    state["current_step"] = None
    _save_checkpoint(state, checkpoint_path)
    print(
        f"[RESET] Reset '{target_name}' and invalidated any later steps. "
        "They will run again."
    )


def _run_step(
    name: str,
    action: Callable[[], Any],
    state: dict,
    checkpoint_path: Path,
    raw_path: Path,
    output_database_path: Path | None = None,
) -> None:
    if state["completed_steps"].get(name) is True:
        if output_database_path is None:
            print(f"[SKIP] {name}: already complete.")
            return
        try:
            verify_database_complete(raw_path, output_database_path, name)
        except RuntimeError:
            print(
                f"[RESUME] {name}: checkpoint says complete, but output is "
                "incomplete."
            )
        else:
            print(f"[SKIP] {name}: already complete.")
            return

    # Persist an in-progress state before doing work.
    # An interrupted step is rerun.
    state["completed_steps"][name] = False
    state["current_step"] = name
    _save_checkpoint(state, checkpoint_path)

    # Remove partial output before retrying a database-producing step.
    if output_database_path is not None:
        if output_database_path.exists():
            print(
                f"[CLEAN] Removing incomplete output: {output_database_path}"
            )
            shutil.rmtree(output_database_path)
        output_database_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n---\nStarting step: {name}\n---\n")
    action()

    # Do not mark the step complete unless all raw files made it through.
    if output_database_path is not None:
        verify_database_complete(raw_path, output_database_path, name)

    state["completed_steps"][name] = True
    state["current_step"] = None
    _save_checkpoint(state, checkpoint_path)
    print(f"[DONE] {name}: checkpoint saved.")


def _run_tsne(
    elementary_csv: Path,
    histogram_csv: Path,
    output_csv: Path,
    output_image: Path | None = None,
    perplexity: float = 30.0,
    seed: int = 0,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    engine = dr_engine.DimensionalityReductionEngine(
        elementary_csv=elementary_csv,
        histogram_csv=histogram_csv,
    )
    points = engine.reduce(
        perplexity=perplexity,
        random_state=seed,
        balance=True,
    )
    engine.save(points, output_csv)
    features = engine.balanced_features()
    print(
        f"[t-SNE] Saved {len(points)} 2D embedding points to: {output_csv}\n"
        f"[t-SNE] 10 nearest neighbours class agreement: "
        f"{engine.class_agreement(features):.3f} (feature space), "
        f"{engine.class_agreement(points):.3f} (2D plot)"
    )
    if output_image is not None:
        plot = Scatterplot(str(output_csv))
        plot.save_image(output_image)


def prepare_database(
    database_path: str,
    output_path: str = "3d_models_prepared",
    *,
    analyze: bool = False,
    evaluate: bool = True,
    run_dr: bool = True,
    reset_checkpoint: bool = False,
    reset_last_step: bool = False,
    seed: int = 42,
    normalize_by_sum: bool = True,
    weights_csv: str | Path | None = None,
    tsne_perplexity: float = 30.0,
    knn_k: int = 10,
    rnn_radius: float = 0.5,
    weighted_k: int = 10,
    weighted_config: str = "emd",
) -> None:
    if reset_checkpoint and reset_last_step:
        raise ValueError(
            "Choose either reset_checkpoint or reset_last_step, not both."
        )

    raw_path = Path(database_path).expanduser().resolve()
    if not raw_path.is_dir():
        raise NotADirectoryError(
            f"Raw database directory not found: {raw_path}"
        )

    output_root = Path(output_path).expanduser().resolve()
    if output_root == raw_path or raw_path in output_root.parents:
        raise ValueError(
            "output_path must be outside the raw database directory."
        )

    analysis_path = output_root / "analysis"
    databases_path = output_root / "3d_models_steps"
    analysis_path.mkdir(parents=True, exist_ok=True)
    databases_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_root / "pipeline_checkpoint.pkl"

    if reset_checkpoint:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"[RESET] Deleted checkpoint: {checkpoint_path}")
        else:
            print("[RESET] No checkpoint exists; starting a fresh run.")

    state = _load_checkpoint(checkpoint_path, raw_path)

    if reset_last_step:
        _reset_last_step(state, checkpoint_path)

    _save_checkpoint(state, checkpoint_path)

    # ---
    # Analysing raw database (optional)

    if analyze:
        raw_analysis_dir = analysis_path / "0_raw"
        raw_analysis_dir.mkdir(parents=True, exist_ok=True)
        raw_analysis_csv = raw_analysis_dir / "raw_shape_analysis.csv"
        _run_step(
            "analyze_raw_database",
            lambda: shape_analyzer.analyze_database(
                str(raw_path), str(raw_analysis_csv)
            ),
            state,
            checkpoint_path,
            raw_path,
        )

    # ---
    # Repairing database

    repaired = databases_path / "1_models_reparation"
    _run_step(
        "repair_database",
        lambda: reparation.repair_directory(str(raw_path), str(repaired)),
        state,
        checkpoint_path,
        raw_path,
        repaired,
    )

    # ---
    # Analysing repairment (optional)

    if analyze:
        reparation_analysis_dir = analysis_path / "1_models_reparation"
        reparation_analysis_dir.mkdir(parents=True, exist_ok=True)
        repair_csv = reparation_analysis_dir / "reparation_analysis.csv"
        _run_step(
            "analyze_reparation",
            lambda: rep_analyser.compare_hole_fill_stats(
                str(raw_path),
                str(repaired),
                str(repair_csv),
                show_progress=True,
            ),
            state,
            checkpoint_path,
            raw_path,
        )

        reperation_shape_analysis_csv = (
            reparation_analysis_dir / "reparation_shape_analysis.csv"
        )
        _run_step(
            "reparation_shape_analysis",
            lambda: shape_analyzer.analyze_database(
                str(repaired), str(reperation_shape_analysis_csv)
            ),
            state,
            checkpoint_path,
            raw_path,
        )

    # ---
    # Resampling database

    resampled = databases_path / "2_models_resampled"
    _run_step(
        "resample_database",
        lambda: resampling.resample_meshes(
            input_path=repaired,
            output_path=resampled,
        ),
        state,
        checkpoint_path,
        raw_path,
        resampled,
    )

    if analyze:
        resampled_analysis_dir = analysis_path / "2_models_resampled"
        resampled_analysis_dir.mkdir(parents=True, exist_ok=True)
        resampled_analysis_csv = (
            resampled_analysis_dir / "resampled_shape_analysis.csv"
        )
        _run_step(
            "analyze_resampled_database",
            lambda: shape_analyzer.analyze_database(
                str(resampled), str(resampled_analysis_csv)
            ),
            state,
            checkpoint_path,
            raw_path,
        )

    # ---
    # Fully normalising database

    fully_normalized = databases_path / "3_models_fully_normalized"
    _run_step(
        "fully_normalize_database",
        lambda: full_normalisation.full_normalize_database(
            input_root=str(resampled), output_root=str(fully_normalized)
        ),
        state,
        checkpoint_path,
        raw_path,
        fully_normalized,
    )

    if analyze:
        normalized_analysis_dir = analysis_path / "3_models_fully_normalized"
        normalized_analysis_dir.mkdir(parents=True, exist_ok=True)
        fully_normalized_shape_analysis_csv = (
            normalized_analysis_dir / "fully_normalized_shape_analysis.csv"
        )
        _run_step(
            "fully_normalized_shape_analysis",
            lambda: shape_analyzer.analyze_database(
                root_dir=str(fully_normalized),
                output_path=str(fully_normalized_shape_analysis_csv),
            ),
            state,
            checkpoint_path,
            raw_path,
        )

    # ---
    # Calculating simple descriptors

    descriptors_analysis_dir = analysis_path / "4_descriptors"
    descriptors_analysis_dir.mkdir(parents=True, exist_ok=True)
    descriptors_csv = descriptors_analysis_dir / "simple_descriptors.csv"
    _run_step(
        "calculate_descriptors",
        lambda: descriptor_analyzer.calculate_database(
            input_dir_path=str(fully_normalized),
            output_path=str(descriptors_csv),
        ),
        state,
        checkpoint_path,
        raw_path,
    )

    # ---
    # Normalizing simple descriptors

    descriptors_normalized_csv = (
        descriptors_analysis_dir / "simple_descriptors_normalized.csv"
    )
    _run_step(
        "normalize_descriptors",
        lambda: normalize_descriptors.process_descriptors(
            csv_path=str(descriptors_csv),
            output_path=str(descriptors_normalized_csv),
        ),
        state,
        checkpoint_path,
        raw_path,
    )

    # ---
    # Calculate histogram descriptors

    hist_descriptors_csv = (
        descriptors_analysis_dir / "histogram_descriptors.csv"
    )
    _run_step(
        "calculate_histogram_descriptor",
        lambda: histogram_descriptor_analyzer.analyze_and_save(
            input=fully_normalized,
            output=hist_descriptors_csv,
            seed=seed,
            normalize_by_sum=normalize_by_sum,
        ),
        state,
        checkpoint_path,
        raw_path,
    )

    # ---
    # Dimensionality Reduction (t-SNE 2D embedding for scatterplot)

    if run_dr:
        embeddings_dir = output_root / "embeddings"
        tsne_csv = embeddings_dir / "tsne.csv"
        tsne_png = embeddings_dir / "tsne_scatterplot.png"
        _run_step(
            "dimensionality_reduction_tsne",
            lambda: _run_tsne(
                elementary_csv=descriptors_normalized_csv,
                histogram_csv=hist_descriptors_csv,
                output_csv=tsne_csv,
                output_image=tsne_png,
                perplexity=tsne_perplexity,
                seed=seed,
            ),
            state,
            checkpoint_path,
            raw_path,
        )

    # ---
    # Retrieval evaluations (kNN, rNN, weighted, and combined comparison)

    if evaluate:
        evaluation_path = output_root / "evaluation"
        evaluation_path.mkdir(parents=True, exist_ok=True)
        knn_output_dir = evaluation_path / "knn"
        rnn_output_dir = evaluation_path / "rnn"
        weighted_output_dir = evaluation_path / "weighted"

        _run_step(
            "run_and_evaluate_knn",
            lambda: evaluate_knn.run_and_evaluate_knn(
                elementary_csv=str(descriptors_normalized_csv),
                histogram_csv=str(hist_descriptors_csv),
                output_dir=str(knn_output_dir),
                k=knn_k,
            ),
            state,
            checkpoint_path,
            raw_path,
        )

        _run_step(
            "run_and_evaluate_rnn",
            lambda: evaluate_rnn.run_and_evaluate_rnn(
                elementary_csv=str(descriptors_normalized_csv),
                histogram_csv=str(hist_descriptors_csv),
                output_dir=str(rnn_output_dir),
                radius=rnn_radius,
            ),
            state,
            checkpoint_path,
            raw_path,
        )

        resolved_weights_csv: Path | None = None
        if weights_csv is not None:
            resolved_weights_csv = Path(weights_csv).expanduser().resolve()
            if (
                not resolved_weights_csv.is_file()
                and resolved_weights_csv.with_suffix(".csv").is_file()
            ):
                resolved_weights_csv = resolved_weights_csv.with_suffix(".csv")
        else:
            candidates: list[Path] = []
            if normalize_by_sum:
                candidates.append(
                    Path(
                        "analysis_output/distances/weights_emd_hist_sampling_descriptors.csv"
                    )
                )
            candidates.extend(
                [
                    Path("analysis_output/distances/weights_emd_descriptors.csv"),
                    Path("analysis_output/distances/weights_equal.csv"),
                ]
            )
            for candidate in candidates:
                if candidate.is_file():
                    resolved_weights_csv = candidate.resolve()
                    break

        if resolved_weights_csv is not None and resolved_weights_csv.is_file():
            _run_step(
                "run_and_evaluate_weighted",
                lambda: evaluate_weighted.run_and_evaluate_weighted(
                    elementary_csv=str(descriptors_normalized_csv),
                    histogram_csv=str(hist_descriptors_csv),
                    weights_csv=str(resolved_weights_csv),
                    output_dir=str(weighted_output_dir),
                    config=weighted_config,
                    k=weighted_k,
                ),
                state,
                checkpoint_path,
                raw_path,
            )
        else:
            print(
                "[WARN] Skipping 'run_and_evaluate_weighted': "
                "no weights CSV found. Specify with --weights_csv."
            )

        _run_step(
            "run_and_evaluate_all",
            lambda: evaluate_all.run_and_evaluate_all(
                elementary_csv=str(descriptors_normalized_csv),
                histogram_csv=str(hist_descriptors_csv),
                weights_csv=str(resolved_weights_csv) if resolved_weights_csv else None,
                output_dir=str(evaluation_path),
                weighted_configs=[weighted_config],
                knn_k=knn_k,
                rnn_radius=rnn_radius,
                weighted_k=weighted_k,
                reuse_existing=True,
            ),
            state,
            checkpoint_path,
            raw_path,
        )

    print(f"Pipeline complete. Checkpoint: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a database with resume support."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input path to the raw input database",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="3d_models_prepared",
        help="Output path to the directory containing all analysis and "
        "processing steps",
    )
    parser.add_argument(
        "-a",
        "--analyze",
        action="store_true",
        help="Run optional shape and reparation analysis steps.",
    )
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--reset_checkpoint",
        "--reset_pickle",
        dest="reset_checkpoint",
        action="store_true",
        help="Delete the checkpoint and rerun the entire pipeline from scratch.",
    )
    reset_group.add_argument(
        "--reset_last_step",
        action="store_true",
        help="Rerun the last unfinished/completed step and all steps after it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for histogram descriptors sampling (default: 42).",
    )
    parser.add_argument(
        "--normalize_by_sum",
        "--normalize-by-sum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize histogram descriptors by sum (sum = 1) instead of area (integral = 1) (default: True).",
    )
    parser.add_argument(
        "--normalize_by_area",
        "--normalize-by-area",
        dest="normalize_by_sum",
        action="store_false",
        help="Normalize histogram descriptors by area (integral = 1) instead of sum.",
    )
    parser.add_argument(
        "--dr",
        "--run_dr",
        dest="run_dr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run dimensionality reduction (t-SNE) and save 2D embeddings (default: True).",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="Perplexity parameter for t-SNE (default: 30.0).",
    )
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run retrieval evaluations (kNN, rNN, weighted, and combined comparison) (default: True).",
    )
    parser.add_argument(
        "--weights_csv",
        type=str,
        default=None,
        help="Path to descriptor weights CSV for weighted retrieval evaluation.",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=10,
        help="Number of nearest neighbors for kNN evaluation (default: 10).",
    )
    parser.add_argument(
        "--rnn_radius",
        type=float,
        default=0.5,
        help="Search radius for rNN evaluation (default: 0.5).",
    )
    parser.add_argument(
        "--weighted_k",
        type=int,
        default=10,
        help="Number of nearest neighbors for weighted evaluation (default: 10).",
    )
    parser.add_argument(
        "--weighted_config",
        type=str,
        choices=["emd", "l2", "cosine"],
        default="emd",
        help="Distance metric configuration for weighted evaluation (default: emd).",
    )
    args = parser.parse_args()

    if args.reset_checkpoint or args.reset_last_step:
        check: str = input("Are you sure to delete previous checkpoint? [y/n]")
        if check.lower() not in ("y", "yes"):
            print("Exiting...")
            return

    prepare_database(
        args.input,
        args.output,
        analyze=args.analyze,
        evaluate=args.evaluate,
        run_dr=args.run_dr,
        reset_checkpoint=args.reset_checkpoint,
        reset_last_step=args.reset_last_step,
        seed=args.seed,
        normalize_by_sum=args.normalize_by_sum,
        weights_csv=args.weights_csv,
        tsne_perplexity=args.tsne_perplexity,
        knn_k=args.knn_k,
        rnn_radius=args.rnn_radius,
        weighted_k=args.weighted_k,
        weighted_config=args.weighted_config,
    )


if __name__ == "__main__":
    main()
