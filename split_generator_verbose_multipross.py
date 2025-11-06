import pandas as pd
from ast import literal_eval
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import utils  # Assumes utils.py is in the same directory
import logging
import os
from datetime import datetime

# --- NEW IMPORTS FOR PARALLELISM ---
import concurrent.futures
from functools import partial
import os

# --- 1. Setup Proper Logging (This is for the MAIN process only) ---
# Setup logging to file and console
log_filename = f'run_log_MAIN_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MAIN] [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logging.info("Starting new experiment run...")


# --- 2. (REMOVED) analyze_path_counts function ---
# The incorrect function has been deleted.
# Its logic is now correctly implemented inside run_test.

# --- 3. Refactored Accuracy Calculation (NOW INCLUDES MCC) ---
# (This function is UNCHANGED)
def calculate_accuracy(test_df):
    """
    Calculates accuracy metrics based on the test_df.
    This is a modified version of get_accuracy_avg that returns
    metrics instead of printing them.
    Now also includes Matthews Correlation Coefficient (MCC).
    """
    diseases = test_df['disease'].unique()
    cat = ['DM', 'SYM', 'NOT']
    cat_map = {name: i for i, name in enumerate(cat)}  # {'DM': 0, 'SYM': 1, 'NOT': 2}
    num_classes = len(cat)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    correct = 0
    dm_cor = 0
    sym_cor = 0
    not_cor = 0
    dm_t = 0
    sym_t = 0
    not_t = 0

    # Add prediction columns
    test_df['prediction'] = 'NA'
    test_df['result'] = False

    for di in diseases:
        di_df = test_df[test_df['disease'] == di]
        drugs = di_df['drug'].unique()

        for d in drugs:
            # Get all rows for this specific drug-disease pair
            dr_df_indices = di_df[di_df['drug'] == d].index
            if dr_df_indices.empty:
                continue

            # Use .loc to ensure we are viewing/modifying the original test_df
            dr_df = test_df.loc[dr_df_indices]

            # Calculate the average scores for this drug-disease pair
            total_comp = [
                dr_df['dm'].replace('NA', 0).sum() / len(dr_df),
                dr_df['sym'].replace('NA', 0).sum() / len(dr_df),
                dr_df['not'].replace('NA', 0).sum() / len(dr_df)
            ]

            # Get the actual category (should be the same for all rows of this pair)
            r = dr_df.iloc[0]
            actual_cat = r['actual_category']

            if actual_cat == 'DM':
                dm_t += 1
            elif actual_cat == 'SYM':
                sym_t += 1
            elif actual_cat == 'NOT':
                not_t += 1

            # Make prediction
            pred_idx = np.argmax(total_comp)
            prediction = cat[pred_idx]
            is_correct = (prediction == actual_cat)

            # Update confusion matrix
            actual_idx = cat_map.get(actual_cat)
            if actual_idx is not None:
                confusion_matrix[actual_idx, pred_idx] += 1
            else:
                logging.warning(f"Unknown actual_category '{actual_cat}' encountered.")

            if is_correct:
                correct += 1
                if actual_cat == 'DM':
                    dm_cor += 1
                elif actual_cat == 'SYM':
                    sym_cor += 1
                elif actual_cat == 'NOT':
                    not_cor += 1

            # Update the original dataframe
            test_df.loc[dr_df_indices, 'prediction'] = prediction
            test_df.loc[dr_df_indices, 'result'] = is_correct

    # --- Calculate Accuracies ---
    total_items = dm_t + sym_t + not_t
    total_acc = correct / total_items if total_items > 0 else 0
    dm_acc = dm_cor / dm_t if dm_t > 0 else 0
    sym_acc = sym_cor / sym_t if sym_t > 0 else 0
    not_acc = not_cor / not_t if not_t > 0 else 0

    # --- Calculate Matthews Correlation Coefficient (MCC) ---
    c = np.trace(confusion_matrix)  # Total correct predictions
    C = np.sum(confusion_matrix)  # Total number of samples

    if C == 0:
        mcc = 0.0
        logging.warning("MCC calculation: Total samples is zero.")
    else:
        t = np.sum(confusion_matrix, axis=1)  # Row sums (true occurrences of each class)
        p = np.sum(confusion_matrix, axis=0)  # Col sums (predicted occurrences of each class)

        sum_tk_pk = np.sum(t * p)
        sum_tk_sq = np.sum(t ** 2)
        sum_pk_sq = np.sum(p ** 2)

        numerator = c * C - sum_tk_pk
        denominator_sq = (C ** 2 - sum_tk_sq) * (C ** 2 - sum_pk_sq)

        if denominator_sq <= 0:
            mcc = 0.0
            if C > 0:
                logging.warning(f"MCC denominator is non-positive ({denominator_sq}). Setting MCC to 0.")
        else:
            denominator = np.sqrt(denominator_sq)
            mcc = numerator / denominator

    # Log the confusion matrix
    logging.info(f"--- Confusion Matrix (Rows: Actual, Cols: Predicted) ---")
    cm_df = pd.DataFrame(confusion_matrix, index=cat, columns=cat)
    logging.info(f"\n{cm_df.to_string()}")

    # Return all metrics
    return total_acc, dm_acc, sym_acc, not_acc, mcc, test_df


# --- 4. Refactored run_test Function (NOW INCLUDES PATH LENGTH ANALYSIS) ---
# (This function is UNCHANGED)
def run_test(base_df, train_df, test_df, ext, seed, khop):
    """
    Main testing pipeline.
    - Uses the *improved* utils.get_int_map
    - Logs results instead of printing
    - Returns metrics for aggregation
    - *** NEW: Calculates path length stats for each test pair ***
    """
    logging.info(f"--- Running Test: split={ext}, seed={seed}, khop={khop} ---")

    # Merge splits with base path data
    merged_df_train = pd.merge(train_df, base_df, how='left', left_on=['drug', 'disease'], right_on=['drug', 'disease'])
    merged_df_test = pd.merge(test_df, base_df, how='left', left_on=['drug', 'disease'], right_on=['drug', 'disease'])

    # --- Train ---
    logging.info("Processing training data...")
    df = merged_df_train[merged_df_train['path_it'].notna()].copy()
    df['path_it'] = df['path_it'].apply(literal_eval)

    data = []
    train_diseases = df['disease'].unique()
    for di in tqdm(train_diseases, desc="Building disease profiles"):
        data = data + utils.get_int_map(df, di, khop)

    di_gene_df = pd.DataFrame(data, columns=['disease', 'gene', 'dm', 'sym', 'not'])
    logging.info(f"Built gene profile for {len(train_diseases)} diseases.")

    # --- Test ---
    logging.info("Processing test data...")
    df = merged_df_test[merged_df_test['path_it'].notna()].copy()
    df['path_it'] = df['path_it'].apply(literal_eval)

    path_vec = []

    # *** NEW: List to store pair-level path statistics ***
    pair_level_path_stats = []

    try:
        similarity = pd.read_csv('similarity.csv')
    except FileNotFoundError:
        logging.error("similarity.csv not found. Continuing without gene replacement.")
        similarity = pd.DataFrame(columns=['node1', 'node2', 'similarity'])  # Empty df

    unknown = 0
    known = 0
    fixed = 0

    # Helper function for softmax
    def avg_prob(x):
        e_x = x
        sum_e_x = e_x.sum(axis=1, keepdims=True)
        sum_e_x[sum_e_x == 0] = 1
        return e_x / sum_e_x

    def score_path(path, known_genes, map):
        return sum(map.get(gene[0], -1) for gene in path[1:-1] if gene[1] == 'Gene')

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Evaluating test pairs"):
        paths = r['path_it']

        # --- *** NEW: Path Length Analysis *** ---
        paths_to_process = paths[:khop]
        num_paths_processed = len(paths_to_process)

        avg_len, med_len, min_len, max_len = np.nan, np.nan, np.nan, np.nan

        if num_paths_processed > 0:
            # Calculate actual length (number of nodes) for each path
            path_lengths = [len(p) for p in paths_to_process if p and len(p) > 0]

            if path_lengths:  # Ensure list is not empty
                avg_len = np.mean(path_lengths)
                med_len = np.median(path_lengths)
                min_len = np.min(path_lengths)
                max_len = np.max(path_lengths)

        # Store the calculated stats for this pair
        pair_level_path_stats.append({
            'disease': r['disease'],
            'drug': r['drug'],
            'gold_label': r['category'],
            'avg_path_len': avg_len,
            'med_path_len': med_len,
            'min_path_len': min_len,
            'max_path_len': max_len,
            'num_paths': num_paths_processed
        })
        # --- *** END: Path Length Analysis *** ---

        # --- Gene processing (as before) ---
        sub_df = di_gene_df[di_gene_df['disease'] == r['disease']]
        known_genes = sub_df['gene'].unique()

        if sub_df.empty:
            logging.warning(f"No training gene profile for disease: {r['disease']}. Skipping drug: {r['drug']}")
            continue

        sub_df_sim = similarity[similarity['node1'].isin(known_genes)]

        sub_df = sub_df.reset_index(drop=True)
        probs = sub_df[['dm', 'sym', 'not']].values
        softmax_probs = avg_prob(probs)
        sub_df['max'] = np.max(softmax_probs, axis=1)

        allowed_known_map = {}
        for idx, sr in sub_df.iterrows():
            if sr['max'] >= 0.4:
                allowed_known_map[sr['gene']] = sr[['dm', 'sym', 'not']].values[np.argmax(softmax_probs[idx])]
            else:
                allowed_known_map[sr['gene']] = 0

        scored_paths = [(score_path(path, known_genes, allowed_known_map), path) for path in paths]
        sorted_paths = sorted(scored_paths, key=lambda x: x[0], reverse=True)
        sorted_paths_only = [path for score, path in sorted_paths]

        gene_counts_in_paths = {}
        processed_genes = set()
        for p in sorted_paths_only[:200]:
            for n in p[1:-1]:
                if n[1] == 'Gene':
                    gene_id = n[0]
                    processed_genes.add(gene_id)
                    gene_counts_in_paths[gene_id] = gene_counts_in_paths.get(gene_id, 0) + 1

        for k in gene_counts_in_paths.keys():
            gene_counts_in_paths[k] = 1

        for i in processed_genes:
            row = sub_df[sub_df['gene'] == i]
            gene_occurrence = gene_counts_in_paths.get(i, 0)

            if not row.empty:
                known += 1
                path_vec.append([r['disease'], r['drug'], i, 'NA', 'NA', gene_occurrence,
                                 row['dm'].values[0], row['sym'].values[0], row['not'].values[0], r['category']])
            else:
                unknown += 1
                matching_sim = sub_df_sim[sub_df_sim['node2'] == i]
                if not matching_sim.empty:
                    fixed += 1
                    sim_gene = matching_sim['node1'].values[0]
                    sim_score = matching_sim['similarity'].values[0]
                    row = sub_df[sub_df['gene'] == sim_gene]

                    if not row.empty:
                        path_vec.append([r['disease'], r['drug'], i, sim_gene, sim_score, gene_occurrence,
                                         row['dm'].values[0], row['sym'].values[0], row['not'].values[0],
                                         r['category']])
                    else:
                        path_vec.append([r['disease'], r['drug'], i, sim_gene, sim_score, gene_occurrence,
                                         'NA', 'NA', 'NA', r['category']])
                else:
                    path_vec.append([r['disease'], r['drug'], i, 'NA', 'NA', gene_occurrence,
                                     'NA', 'NA', 'NA', r['category']])

    if not path_vec:
        logging.error("No test data (path_vec) was generated. Check test set and disease profiles.")
        return 0, 0, 0, 0, 0, 0, 0, 0

    test_df_out = pd.DataFrame(path_vec, columns=['disease', 'drug', 'gene', 'gene_replaced_by', 'similarity',
                                                  'occurance_across_path', 'dm', 'sym', 'not', 'actual_category'])

    # --- Accuracy Calculation ---
    total_acc, dm_acc, sym_acc, not_acc, mcc, evaluated_test_df = calculate_accuracy(test_df_out)

    logging.info(f"Accuracy (Total): {total_acc:.4f}")
    logging.info(f"Accuracy (DM):    {dm_acc:.4f}")
    logging.info(f"Accuracy (SYM):   {sym_acc:.4f}")
    logging.info(f"Accuracy (NOT):   {not_acc:.4f}")
    logging.info(f"MCC:              {mcc:.4f}")

    # Save evaluated file
    out_file = f'test_df{khop}_evaluated_{ext}_{seed}.csv'
    evaluated_test_df.to_csv(out_file, index=False)
    logging.info(f"Saved evaluated test df to {out_file}")

    # --- *** NEW: Merge Path Stats with Predictions and Save *** ---
    pair_stats_df = pd.DataFrame(pair_level_path_stats)

    # Get unique predictions from the (gene-level) evaluated_test_df
    pair_predictions = evaluated_test_df[['disease', 'drug', 'prediction']].drop_duplicates()

    # Merge stats with predictions
    final_analysis_df = pd.merge(pair_stats_df, pair_predictions, on=['disease', 'drug'], how='left')

    # Save the final analysis file
    analysis_file = f'path_analysis_test_{khop}_{ext}_{seed}.csv'
    final_analysis_df.to_csv(analysis_file, index=False)
    logging.info(f"Saved path length analysis to {analysis_file}")
    # --- *** END: Save Analysis *** ---

    # --- Final Stats ---
    total_pairs = len(df)
    avg_unknown = unknown / total_pairs if total_pairs > 0 else 0
    avg_known = known / total_pairs if total_pairs > 0 else 0
    avg_fixed = fixed / total_pairs if total_pairs > 0 else 0

    logging.info(f"Average Unknown Genes per Pair: {avg_unknown:.4f}")
    logging.info(f"Average Known Genes per Pair:   {avg_known:.4f}")
    logging.info(f"Average Fixed Genes per Pair:   {avg_fixed:.4f}")
    logging.info(f"--- Finished Test: split={ext}, seed={seed}, khop={khop} ---")

    return total_acc, dm_acc, sym_acc, not_acc, mcc, avg_unknown, avg_known, avg_fixed


# --- 5. NEW Worker Function for Parallel Processing ---
def process_fold(khop, ik, train_index, test_index, base_df, df):
    """
    Worker function to process a single fold/khop combination.
    This function will be run in a separate process and will
    configure its own logging.
    """
    try:
        # --- Configure Logging for THIS worker ---
        # This is CRITICAL for parallel processing to avoid corrupting logs.
        # Each worker gets its own log file.
        worker_log_file = f'run_log_khop{khop}_fold{ik}.log'

        # Remove all handlers from the root logger (inherited from main process)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Add new handlers for this worker
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s [K{khop}-F{ik}] [%(levelname)s] - %(message)s',
            handlers=[
                logging.FileHandler(worker_log_file),
                logging.StreamHandler()  # Will interleave on console
            ]
        )
        # --- End Worker Logging Setup ---

        logging.info(f"Worker starting for khop={khop}, fold={ik}")

        # Create the data splits for this specific job
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        # Save the split files (as in the original script)
        train_split_file = f'train_split_fold{str(ik)}.csv'
        test_split_file = f'test_split_fold{str(ik)}.csv'
        train_df.to_csv(train_split_file, index=False)
        test_df.to_csv(test_split_file, index=False)
        logging.info(f"Saved {train_split_file} and {test_split_file}")

        # --- Run Main Test ---
        (total_acc, dm_acc, sym_acc, not_acc, mcc,
         avg_unknown, avg_known, avg_fixed) = run_test(
            base_df, train_df, test_df, "fold" + str(ik), "_", khop
        )

        logging.info(f"Worker finished successfully for khop={khop}, fold={ik}")

        # Return the results as a dictionary
        return {
            'split': ik,
            'khop': khop,
            'total_accuracy': total_acc,
            'dm_accuracy': dm_acc,
            'sym_accuracy': sym_acc,
            'not_accuracy': not_acc,
            'mcc': mcc,
            'avg_unknown_genes': avg_unknown,
            'avg_known_genes': avg_known,
            'avg_fixed_genes': avg_fixed
        }
    except Exception as e:
        # Log the error to this worker's log file
        logging.error(f"Error in worker (khop={khop}, split={ik}): {e}", exc_info=True)
        # Return an error dictionary
        return {
            'split': ik,
            'khop': khop,
            'total_accuracy': 'ERROR',
            'dm_accuracy': 'ERROR',
            'sym_accuracy': 'ERROR',
            'not_accuracy': 'ERROR',
            'mcc': 'ERROR',
            'avg_unknown_genes': 'ERROR',
            'avg_known_genes': 'ERROR',
            'avg_fixed_genes': 'ERROR'
        }


# --- 6. Main Execution Block (MODIFIED FOR PARALLELISM) ---
if __name__ == "__main__":
    try:
        split_train_base = pd.read_csv('split_train.csv', delimiter=",")
        split_test_base = pd.read_csv('split_test.csv', delimiter=",")
        base_df = pd.read_csv('indications_500.csv', delimiter=",")[['drug', 'disease', 'path_it']]
    except FileNotFoundError as e:
        logging.error(f"Error loading base file: {e}. Exiting.")
        exit()

    df = pd.concat([split_train_base, split_test_base])

    khops = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    fk = sklearn.model_selection.KFold(n_splits=10, shuffle=True)

    # --- 1. Generate list of all tasks to run ---
    tasks = []
    for khop in khops:
        ik = 0  # Reset fold index for each new KFold split
        for i, (train_index, test_index) in enumerate(fk.split(df)):
            ik += 1
            tasks.append((khop, ik, train_index, test_index))

    logging.info(f"Generated {len(tasks)} tasks (10 khops x 10 folds).")

    # --- 2. Run tasks in parallel ---
    results_log = []

    # Use a sensible number of workers (e.g., all cores minus one)
    num_workers = max(1, os.cpu_count() - 1)
    logging.info(f"Starting ProcessPoolExecutor with {num_workers} workers.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use functools.partial to "bake in" the large dataframes (base_df, df)
        # This is efficient on Linux/macOS (copy-on-write) and works on Windows (by pickling)
        worker_func = partial(process_fold, base_df=base_df, df=df)

        # Submit all tasks and create a dictionary of {future: task_metadata}
        futures = {executor.submit(worker_func, khop, ik, train_idx, test_idx): (khop, ik)
                   for (khop, ik, train_idx, test_idx) in tasks}

        # Process results as they complete, with a progress bar
        logging.info("Submitting tasks to pool...")
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing Tasks"):
            khop, ik = futures[future]  # Get metadata for the completed task
            try:
                result = future.result()  # Get the dictionary returned by process_fold
                results_log.append(result)
            except Exception as e:
                logging.error(f"Task (khop={khop}, split={ik}) failed unexpectedly in main thread: {e}")
                results_log.append({
                    'split': ik, 'khop': khop, 'total_accuracy': 'FAIL',
                    'dm_accuracy': 'FAIL', 'sym_accuracy': 'FAIL',
                    'not_accuracy': 'FAIL', 'mcc': 'FAIL',
                    'avg_unknown_genes': 'FAIL', 'avg_known_genes': 'FAIL',
                    'avg_fixed_genes': 'FAIL'
                })

    # --- 3. Save Final Aggregated Results ---
    logging.info("All runs complete. Aggregating results...")
    results_df = pd.DataFrame(results_log)

    # Sort the results for a cleaner output file
    results_df = results_df.sort_values(by=['khop', 'split'])

    results_file = 'experiment_results_log_parallel.csv'
    results_df.to_csv(results_file, index=False)
    logging.info(f"--- All runs complete. Aggregated results saved to {results_file} ---")