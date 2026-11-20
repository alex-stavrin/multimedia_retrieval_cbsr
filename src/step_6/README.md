# run_and_evaluate_all.py

```run_and_evaluate_all.py``` runs and evaluates all retrieval methods (`knn`, `rnn`, and `weighted`), compares their quality metrics and retrieval times (`init_time_s`, `retrieval_time_s`, `mean_query_time_ms`, `total_time_s`) side by side in a single comparison table (`all_methods_comparison_table.csv`) and ranking table (`all_methods_ranking_table.csv`), and plots the combined ROC curves and metric/time comparisons.

You can run it from the root directory with:  
```python -m src.step_6.run_and_evaluate_all --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --weights_csv ./analysis_output/distances/weights_emd_hist_sampling_descriptors.csv --output_dir ./evaluation --knn_k 6 --rnn_radius 0.85 --weighted_k 6 --weighted_configs emd l2 cosine```  
--elementary_csv = Path to the normalized elementary descriptors CSV file.  
--histogram_csv = Path to the histogram descriptors CSV file.  
--weights_csv = Path to the descriptor weights CSV file used for weighted retrieval.  
-o / --output_dir = Root directory where individual and combined evaluation results will be saved.  
--methods = List of retrieval methods to evaluate (`knn`, `rnn`, `weighted`; default is all three).  
--weighted_configs = List of Step 4 weighted distance configurations to evaluate (`emd`, `l2`, `cosine`; default is `emd`).  
-k / --knn_k = Number of nearest neighbors `k` for kNN evaluation (default is 6).  
-r / --rnn_radius = Search radius `r` for rNN evaluation (default is 0.85).  
--weighted_k = Number of nearest neighbors `k` for weighted distance evaluation (default is 6).  
--reuse_existing = If present, reuses existing evaluation JSONs in `output_dir` instead of recomputing.

# run_and_evaluate_weighted.py

```run_and_evaluate_weighted.py``` uses ```ShapeDistanceCalculator```, ```RetrievalEvaluator```, and ```ROCEvaluator``` to evaluate the Step 4 custom weighted distance retrieval method across every query shape in the database and saves per-query, per-class, confusion matrix, ROC, and summary metrics.

You can run it from the root directory with:  
```python -m src.step_6.run_and_evaluate_weighted --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --weights_csv ./analysis_output/distances/weights_emd_hist_sampling_descriptors.csv --output_dir ./evaluation/weighted -c emd -k 6```  
--elementary_csv = Path to the normalized elementary descriptors CSV file.  
--histogram_csv = Path to the histogram descriptors CSV file.  
--weights_csv = Path to the descriptor weights CSV file.  
-o / --output_dir = Directory where evaluation CSVs, JSON stats, and ROC curve plots will be saved.  
-c / --config = Distance metric configuration (`emd`, `l2`, or `cosine`; default is `emd`).  
-k = Number of nearest neighbors to retrieve per query (default is 6).

# run_and_evaluate_knn.py

```run_and_evaluate_knn.py``` uses ```SpatialSearchEngine```, ```RetrievalEvaluator```, and ```ROCEvaluator``` to evaluate the Step 5 KD-Tree `kNN` retrieval method across every query shape in the database and saves per-query, per-class, confusion matrix, ROC, and summary metrics.

You can run it from the root directory with:  
```python -m src.step_6.run_and_evaluate_knn --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --output_dir ./evaluation/knn -k 6```  
--elementary_csv = Path to the normalized elementary descriptors CSV file.  
--histogram_csv = Path to the histogram descriptors CSV file.  
-o / --output_dir = Directory where evaluation CSVs, JSON stats, and ROC curve plots will be saved.  
-k = Number of nearest neighbors to retrieve per query (default is 6).  
-p = Minkowski distance order (default is `2.0` for Euclidean distance).  
--eps = KD-Tree approximation factor (default is `0.0` for exact search).

# run_and_evaluate_rnn.py

```run_and_evaluate_rnn.py``` uses ```SpatialSearchEngine```, ```RetrievalEvaluator```, and ```ROCEvaluator``` to evaluate the Step 5 KD-Tree `rNN` (range search) retrieval method across every query shape in the database and saves per-query, per-class, confusion matrix, ROC, and summary metrics.

You can run it from the root directory with:  
```python -m src.step_6.run_and_evaluate_rnn --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --output_dir ./evaluation/rnn -r 0.85```  
--elementary_csv = Path to the normalized elementary descriptors CSV file.  
--histogram_csv = Path to the histogram descriptors CSV file.  
-o / --output_dir = Directory where evaluation CSVs, JSON stats, and ROC curve plots will be saved.  
-r / --radius = Search radius distance threshold (default is 0.85).  
-p = Minkowski distance order (default is `2.0` for Euclidean distance).  
--eps = KD-Tree approximation factor (default is `0.0` for exact search).

# retrieval_evalulator.py

```retrieval_evalulator.py``` has the ```RetrievalEvaluator``` class that computes per-query, per-class, micro-averaged, and macro-averaged retrieval evaluation metrics (precision, recall/sensitivity, specificity, accuracy, F1 score, average precision / MAP, last rank, first tier, second tier, top-1 accuracy, query times, and confusion matrix).

# roc_evaluator.py

```roc_evaluator.py``` has the ```ROCEvaluator``` class that computes sensitivity vs. specificity ROC curves and AUROC scores (per-query, per-class, micro, and macro) for both ranked retrieval lists and global radius sweeps (`rNN`).
