"""
Python-only MOTOR linear probe training.
Bypasses the buggy C++ create_batches/BatchLoader by tokenizing
patients directly in Python and running through the MOTOR transformer.

Usage:
    python motor_linear_probe_python.py \
        --task PE \
        --data_path output/inspect_femr_extract/extract \
        --model_dir motor_model/model \
        --dictionary_path motor_model/dictionary \
        --labels_dir output/labels_and_features \
        --cohort_path data/cohort_motor.csv \
        --output_dir output/motor_results/PE
"""

import os
os.environ["JAX_NUMPY_RANK_PROMOTION"] = "raise"

import argparse
import collections
import csv
import datetime
import functools
import logging
import pickle
import sys

import jax
import jax.numpy as jnp
import haiku as hk
import msgpack
import numpy as np
import sklearn.metrics

import femr.datasets
import femr.labelers
import femr.models.transformer

csv.field_size_limit(sys.maxsize)


def load_dictionary(path):
    with open(path, "rb") as f:
        return msgpack.load(f, use_list=False)


def load_model_config(model_dir):
    with open(os.path.join(model_dir, "config.msgpack"), "rb") as f:
        return msgpack.load(f, use_list=False)


def load_model_params(model_dir):
    with open(os.path.join(model_dir, "best"), "rb") as f:
        return pickle.load(f)


def build_code_to_token_ids(dictionary, vocab_size):
    """Build mapping from code_string -> list of token_ids (including parents).

    The ontology_rollup is a tuple where index = token_id, each entry has 'code_string'.
    all_parents maps code_string -> tuple of (self, parent1, parent2, ...) code_strings.
    """
    rollup = dictionary["ontology_rollup"]

    # Step 1: code_string -> token_id (first occurrence only)
    code_to_id = {}
    for token_id, entry in enumerate(rollup):
        cs = entry["code_string"]
        if cs not in code_to_id:
            code_to_id[cs] = token_id

    # Step 2: For each code, expand via all_parents to get hierarchical token_ids
    all_parents = dictionary.get("all_parents", {})
    code_to_token_ids = {}

    for code_string, parent_codes in all_parents.items():
        token_ids = []
        for pc in parent_codes:
            if pc in code_to_id:
                tid = code_to_id[pc]
                if tid < vocab_size:
                    token_ids.append(tid)
        if token_ids:
            code_to_token_ids[code_string] = tuple(token_ids)

    # Also add codes that are in rollup but not in all_parents (direct mapping)
    for cs, tid in code_to_id.items():
        if cs not in code_to_token_ids and tid < vocab_size:
            code_to_token_ids[cs] = (tid,)

    return code_to_token_ids


def tokenize_patient(patient, code_to_token_ids):
    """Tokenize a patient's events using hierarchical ontology.

    Returns list of (age_days, (token_id1, token_id2, ...)) for each event.
    Ages are in days (matching the MOTOR dictionary age_stats units).
    """
    birth_date = None
    events_data = []

    for event in patient.events:
        code = event.code
        time = event.start

        if "Birth" in code:
            birth_date = time
            continue

        if birth_date is None:
            continue

        age_days = (time - birth_date).total_seconds() / 86400.0
        if age_days < 0:
            continue

        if code in code_to_token_ids:
            events_data.append((age_days, code_to_token_ids[code]))

    return events_data, birth_date


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def main():
    parser = argparse.ArgumentParser(description="Python-only MOTOR linear probe")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dictionary_path", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    parser.add_argument("--cohort_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_events", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "log")),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Starting MOTOR linear probe for task: {args.task}")

    # Load components
    logging.info("Loading database...")
    database = femr.datasets.PatientDatabase(args.data_path)

    logging.info("Loading dictionary...")
    dictionary = load_dictionary(args.dictionary_path)

    logging.info("Loading model config...")
    config = load_model_config(args.model_dir)
    config = hk.data_structures.to_immutable_dict(config)
    vocab_size = config["transformer"]["vocab_size"]
    hidden_size = config["transformer"]["hidden_size"]

    logging.info("Loading model params...")
    params = load_model_params(args.model_dir)
    params = femr.models.transformer.convert_params(params, dtype=jnp.float16)

    logging.info("Building code-to-token mapping...")
    code_to_token_ids = build_code_to_token_ids(dictionary, vocab_size)
    logging.info(f"  {len(code_to_token_ids)} codes mapped to token IDs")

    age_stats = dictionary.get("age_stats", {"mean": 0.0, "std": 1.0})
    age_mean = age_stats["mean"]
    age_std = max(age_stats["std"], 1.0)

    # Load labels
    logging.info("Loading labels...")
    labels_path = os.path.join(args.labels_dir, args.task, "labeled_patients.csv")
    labeled_patients = femr.labelers.load_labeled_patients(labels_path)

    # Load cohort splits
    logging.info("Loading cohort splits...")
    pid_split = {}
    with open(args.cohort_path) as f:
        for row in csv.DictReader(f):
            pid_split[int(row["patient_id"])] = row["split"]

    # Collect all (pid, label_time, label_value, split)
    all_samples = []
    for pid, labels in labeled_patients.items():
        pid = int(pid)
        if pid not in pid_split:
            continue
        split = pid_split[pid]
        for label in labels:
            all_samples.append((pid, label.time, label.value, split))

    logging.info(f"Total samples: {len(all_samples)}")
    split_counts = collections.Counter(s[3] for s in all_samples)
    logging.info(f"Split distribution: {dict(split_counts)}")

    # Tokenize all patients (one-time cost)
    logging.info("Tokenizing patients...")
    patient_tokens = {}  # pid -> (events_data, birth_date)
    unique_pids = set(s[0] for s in all_samples)

    for i, pid in enumerate(unique_pids):
        if i % 2000 == 0:
            logging.info(f"  Tokenizing patient {i}/{len(unique_pids)}...")
        try:
            patient = database[pid]
            events, birth = tokenize_patient(patient, code_to_token_ids)
            if events and birth is not None:
                patient_tokens[pid] = (events, birth)
        except Exception as e:
            logging.warning(f"  Could not tokenize pid={pid}: {e}")

    logging.info(f"Successfully tokenized {len(patient_tokens)}/{len(unique_pids)} patients")

    # Build model function
    def model_fn(config, batch):
        return femr.models.transformer.EHRTransformer(config)(batch, no_task=True)

    model = hk.transform(model_fn)
    rng = jax.random.PRNGKey(42)

    # Check for cached representations
    cache_path = os.path.join(args.output_dir, "reprs_cache.pkl")
    use_cache = os.path.exists(cache_path)

    if use_cache:
        logging.info(f"Loading cached representations from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        reprs = cache["reprs"]
        label_pids = cache["pids"]
        label_values = cache["values"]
        splits = cache["splits"]
        all_label_times = cache["times"]
        logging.info(f"Loaded {reprs.shape[0]} cached representations, shape {reprs.shape}")

    if not use_cache:
        # Process samples through the transformer
        logging.info("Computing representations...")

        # Sort samples for efficient processing
        all_samples.sort(key=lambda s: (s[0], s[1]))

        FIXED_SEQ_LEN = 2048
        MAX_SPARSE = 36000

        @functools.partial(jax.jit, static_argnames=("config",))
        def compute_repr(params, rng, config, batch):
            features, mask = model.apply(params, rng, config, batch)
            bias = jnp.ones((features.shape[0], 1), dtype=features.dtype)
            return jnp.concatenate((features, bias), axis=-1)

        all_reprs = []
        all_label_pids = []
        all_label_values = []
        all_label_times = []
        all_splits = []
        processed = 0
        skipped = 0

        for sample_idx, (pid, label_time, label_value, split) in enumerate(all_samples):
            if pid not in patient_tokens:
                skipped += 1
                continue

            events, birth_date = patient_tokens[pid]
            birth_dt = (
                datetime.datetime.combine(birth_date, datetime.time.min)
                if isinstance(birth_date, datetime.date) and not isinstance(birth_date, datetime.datetime)
                else birth_date
            )
            label_age = (label_time - birth_dt).total_seconds() / 86400.0

            filtered = [(age, toks) for age, toks in events if age <= label_age]
            if not filtered:
                skipped += 1
                continue

            if len(filtered) > args.max_events:
                filtered = filtered[-args.max_events:]

            seq_len = len(filtered)

            ages = np.zeros(FIXED_SEQ_LEN, dtype=np.float32)
            normed_ages = np.zeros(FIXED_SEQ_LEN, dtype=np.float32)
            integer_ages = np.zeros(FIXED_SEQ_LEN, dtype=np.uint32)
            valid = np.zeros(FIXED_SEQ_LEN, dtype=np.uint8)
            sparse_list = []

            for i, (age, token_ids) in enumerate(filtered):
                ages[i] = float(age)
                integer_ages[i] = int(age)
                normed_ages[i] = (float(age) - age_mean) / age_std
                valid[i] = 1
                for tid in token_ids:
                    sparse_list.append((tid, i))

            sparse_list.sort(key=lambda x: x[1])
            if len(sparse_list) > MAX_SPARSE:
                sparse_list = sparse_list[:MAX_SPARSE]

            n_sparse = len(sparse_list)
            sparse_arr = np.zeros((MAX_SPARSE, 2), dtype=np.uint32)
            if n_sparse > 0:
                sparse_arr[:n_sparse] = np.array(sparse_list, dtype=np.uint32)
            order = np.argsort(sparse_arr[:, 1], kind="stable")
            sparse_arr = sparse_arr[order]

            label_idx = np.array([seq_len - 1], dtype=np.uint32)

            batch = {
                "transformer": {
                    "ages": jnp.array(ages),
                    "normalized_ages": jnp.array(normed_ages),
                    "length": jnp.uint32(FIXED_SEQ_LEN),
                    "valid_tokens": jnp.array(valid),
                    "label_indices": jnp.array(label_idx),
                    "integer_ages": jnp.array(integer_ages),
                    "sparse_token_indices": jnp.array(sparse_arr),
                },
                "task": {
                    "labels": jnp.array([float(label_value)], dtype=jnp.float32),
                },
                "num_indices": 1,
                "patient_ids": np.array([pid], dtype=np.uint64),
            }

            try:
                repr_out = compute_repr(params, rng, config, batch)
                rep = np.array(repr_out[0])
                all_reprs.append(rep)
                all_label_pids.append(pid)
                all_label_values.append(float(label_value))
                all_label_times.append(label_time)
                all_splits.append(split)
                processed += 1

                if processed % 500 == 0:
                    logging.info(f"  Processed {processed}/{len(all_samples)} (skipped {skipped})")
            except Exception as e:
                if processed < 5:
                    logging.error(f"  Error processing pid={pid}: {e}")
                    import traceback
                    traceback.print_exc()
                skipped += 1

        logging.info(f"Total processed: {processed}, skipped: {skipped}")

        if processed == 0:
            logging.error("No samples processed! Cannot train.")
            return

        reprs = np.stack(all_reprs)
        label_pids = np.array(all_label_pids, dtype=np.uint64)
        label_values = np.array(all_label_values)
        splits = np.array(all_splits)

        # Cache representations
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"reprs": reprs, "pids": label_pids, "values": label_values,
                 "splits": splits, "times": all_label_times}, f
            )
        logging.info(f"Cached representations to {cache_path}")

    logging.info(f"Representation shape: {reprs.shape}")

    train_mask = splits == "train"
    valid_mask = splits == "valid"
    test_mask = splits == "test"

    logging.info(f"Train: {train_mask.sum()}, Valid: {valid_mask.sum()}, Test: {test_mask.sum()}")
    logging.info(f"Prevalence: {label_values.mean():.4f}")

    # Train logistic regression with L2 regularization (use float32 for stability)
    reprs_f32 = jnp.array(reprs, dtype=jnp.float32)
    labels_jnp = jnp.array(label_values, dtype=jnp.float32)

    train_reprs = reprs_f32[train_mask]
    train_labels = labels_jnp[train_mask]

    best_valid_auroc = -1
    best_hazards = None
    best_l = None

    for l_exp in np.linspace(1, -5, num=20):
        l = 0 if l_exp == -5 else 10**l_exp

        # Gradient descent for logistic regression
        beta_curr = jnp.zeros(reprs.shape[-1], dtype=jnp.float32)
        for step in range(200):
            hazards = jnp.dot(train_reprs, beta_curr)
            preds = jax.nn.sigmoid(hazards)
            residuals = preds - train_labels
            grad = (
                jnp.dot(train_reprs.T, jnp.expand_dims(residuals, -1)).squeeze(-1)
                / len(train_labels)
            )
            grad = grad + l * jnp.concatenate(
                [beta_curr[:-1], jnp.zeros(1, dtype=beta_curr.dtype)]
            )
            beta_curr = beta_curr - 0.1 * grad

            grad_norm = jnp.linalg.norm(grad, ord=2)
            if grad_norm < 0.0001:
                break

        all_hazards = jnp.dot(reprs_f32, beta_curr)

        try:
            train_auroc = sklearn.metrics.roc_auc_score(
                label_values[train_mask], np.array(all_hazards[train_mask])
            )
            valid_auroc = sklearn.metrics.roc_auc_score(
                label_values[valid_mask], np.array(all_hazards[valid_mask])
            )
            test_auroc = sklearn.metrics.roc_auc_score(
                label_values[test_mask], np.array(all_hazards[test_mask])
            )

            logging.info(
                f"L2={l:.6f}: Train={train_auroc:.4f}, Valid={valid_auroc:.4f}, Test={test_auroc:.4f}"
            )

            if valid_auroc > best_valid_auroc:
                best_valid_auroc = valid_auroc
                best_hazards = np.array(all_hazards)
                best_l = l
        except Exception as e:
            logging.warning(f"L2={l:.6f}: Error computing AUROC: {e}")

    if best_hazards is None:
        logging.error("No valid model found!")
        return

    logging.info(f"Best L2: {best_l}")
    test_auroc = sklearn.metrics.roc_auc_score(
        label_values[test_mask], best_hazards[test_mask]
    )
    logging.info(f"Final Test AUROC: {test_auroc:.4f}")

    # Save predictions in same format as original linear_probe.py: [probs, pids, labels, dates]
    predictions = sigmoid(best_hazards.astype(np.float32))

    with open(os.path.join(args.output_dir, "predictions.pkl"), "wb") as f:
        pickle.dump([predictions, label_pids, label_values, all_label_times], f)

    logging.info(f"Saved predictions to {args.output_dir}/predictions.pkl")
    logging.info("Done!")


if __name__ == "__main__":
    main()
