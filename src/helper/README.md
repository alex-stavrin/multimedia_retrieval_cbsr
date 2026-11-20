# prepare_database.py

```prepare_database.py``` runs the end-to-end 3D model database pipeline (reparation, resampling, full normalization, elementary and histogram descriptor extraction, descriptor standardization, t-SNE dimensionality reduction, and retrieval evaluation) with atomic checkpointing and resume support.

You can run it from the root directory with:  
```python -m src.helper.prepare_database -i ./3d_models -o ./3d_models_final -a --evaluate --weights_csv ./analysis_output/distances/weights_emd_hist_sampling_descriptors.csv --knn_k 6 --rnn_radius 0.85 --weighted_k 6 --normalize_by_sum```  
-i / --input = Path to the raw input 3D shape database folder.  
-o / --output = Output directory for all processed database steps, descriptors, embeddings, and evaluations (default is `3d_models_prepared`).  
-a / --analyze = If present, runs optional shape and reparation analysis steps between pipeline stages.  
--reset_checkpoint = Delete the saved checkpoint and rerun the entire pipeline from scratch.  
--reset_last_step = Rerun the last unfinished/completed step and all subsequent steps.  
--seed = Random seed for histogram descriptor sampling (default is 42).  
--normalize_by_sum / --normalize_by_area = Normalize histograms by sum (default) or by area integral.  
--dr / --no-dr = Enable or disable the t-SNE dimensionality reduction step (enabled by default).  
--evaluate / --no-evaluate = Enable or disable the Step 6 retrieval evaluation steps (enabled by default).  
--weights_csv = Path to the descriptor weights CSV file for weighted retrieval evaluation.  
--knn_k = Number of nearest neighbors for kNN evaluation (default is 6).  
--rnn_radius = Search radius for rNN evaluation (default is 0.85).  
--weighted_k = Number of nearest neighbors for weighted evaluation (default is 6).  
--weighted_config = Distance configuration for weighted evaluation (`emd`, `l2`, or `cosine`; default is `emd`).
