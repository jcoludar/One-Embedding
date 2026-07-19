import sys
from pathlib import Path

import numpy as np
import json
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from runners.per_residue import run_ss3_benchmark, run_ss8_benchmark, run_disorder_benchmark
from runners.protein_level import compute_protein_vectors, run_retrieval_benchmark
from metrics.statistics import paired_bootstrap_retention, paired_cluster_bootstrap_retention
from runners.per_residue import pooled_spearman as _pooled_spearman
from rules import MetricResult
from config import (
    RAW_EMBEDDINGS, SPLITS, LABELS, METADATA, RESULTS_DIR,
    SEEDS, BOOTSTRAP_N, C_GRID, ALPHA_GRID, CV_FOLDS,
)
from src.utils.h5_store import load_residue_embeddings
def load_split(path):
    with open(path) as f:
        return json.load(f)
def metric_to_dict(m):
    """Convert MetricResult to JSON-serializable dict."""
    if isinstance(m, MetricResult):
        return {
            "value": m.value, "ci_lower": m.ci_lower, "ci_upper": m.ci_upper,
            "n": m.n, "seeds_mean": m.seeds_mean, "seeds_std": m.seeds_std,
            "ci_method": m.ci_method,
        }
    return m
def bucket_eval(raw,compressed, bucket_dict,prefix, function:callable,kwargs = {"n_bootstrap":BOOTSTRAP_N, "seed":SEEDS[0]}):
    output_dict = {}
    for bucket_type in bucket_dict.keys():
        for bucket, ids in bucket_dict[bucket_type].items():
            
            bucket_raw = {id: raw[id] for id in ids if id in raw and id in compressed} 
            bucket_comp = {id: compressed[id] for id in ids if id in raw and id in compressed}
            if len(bucket_raw) < 2:
                continue
            bucket_ret_ci = function(
                    bucket_raw, bucket_comp,**kwargs
                    
            )
            bucket_info = {"bucket_type":bucket_type,"bucket":bucket}
            output_dict[f"{prefix}_{bucket_type}_{bucket}"] = metric_to_dict(bucket_ret_ci)#
            output_dict[f"{prefix}_{bucket_type}_{bucket}"].update(bucket_info)
    return output_dict
def load_ss3_labels(prefix,raw_dict):
    from src.evaluation.per_residue_tasks import load_ss_csv
    _, ss3, ss8, _ = load_ss_csv(LABELS[prefix],prefix)
    raw_emb = load_residue_embeddings(raw_dict[prefix])
    
    
    split= load_split(SPLITS[prefix])
    train_ids = split["train_ids"]
    #train_ids = delete_rand_items(train_ids, len(train_ids)/2)
    test_ids = split["test_ids"]
    train_ids = [p for p in train_ids if p in raw_emb and p in ss3]
    test_ids = [p for p in test_ids if p in raw_emb and p in ss8]
        
    return raw_emb, ss3, ss8, train_ids, test_ids
def run_ss3_task(emb,ss3_labels,ss8_labels,train_ids,test_ids,prefix, raw_ss3=None,raw_ss8=None,bucket_dict = None):
    # ── SS3 ──
    print(f"  Running {prefix} SS3...")
    v2_ss3 = run_ss3_benchmark(
        emb, ss3_labels, train_ids, test_ids,
        C_grid=C_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
        n_bootstrap=BOOTSTRAP_N,
    )
    # ── SS8 ──
    print(f"  Running {prefix} SS8...")
    v2_ss8 = run_ss8_benchmark(
        emb, ss8_labels, train_ids, test_ids,
        C_grid=C_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
        n_bootstrap=BOOTSTRAP_N,
    )
    #retention 
    metrics = {
        f"{prefix}_ss3_q3": metric_to_dict(v2_ss3["q3"]),
        f"{prefix}_ss8_q8": metric_to_dict(v2_ss8["q8"]),
        
        f"{prefix}_best_C_ss3": v2_ss3["best_C"],
        f"{prefix}_best_C_ss8": v2_ss8["best_C"],
    }
    bucket_metrics = {}
    if not raw_ss3 is None and not raw_ss8 is None:
        # ── SS3 ──
        ss3_ret_ci = paired_bootstrap_retention(
            raw_ss3["per_protein_scores"], v2_ss3["per_protein_scores"],
            n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        metrics[f"{prefix}_ss3_retention"] = metric_to_dict(ss3_ret_ci)
        print(f"    {prefix} Q3: {v2_ss3['q3'].value:.4f} (retention: {ss3_ret_ci.value:.1f} ± {(ss3_ret_ci.ci_upper - ss3_ret_ci.ci_lower) / 2:.1f}%)")

        # ── SS8 ──
        ss8_ret_ci = paired_bootstrap_retention(
            raw_ss8["per_protein_scores"], v2_ss8["per_protein_scores"],
            n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        metrics[f"{prefix}_ss8_retention"] = metric_to_dict(ss8_ret_ci)
        print(f"    {prefix} Q8: {v2_ss8['q8'].value:.4f} (retention: {ss8_ret_ci.value:.1f} ± {(ss8_ret_ci.ci_upper - ss8_ret_ci.ci_lower) / 2:.1f}%)")
        if not bucket_dict is None:
            bucket_metrics = bucket_eval(raw_ss3["per_protein_scores"],v2_ss3["per_protein_scores"],bucket_dict=bucket_dict,prefix=f"{prefix}_ss3",function=paired_bootstrap_retention,)
            bucket_metrics.update(bucket_eval(raw_ss8["per_protein_scores"],v2_ss8["per_protein_scores"],bucket_dict=bucket_dict,prefix=f"{prefix}_ss8",function=paired_bootstrap_retention,))
    return metrics, v2_ss3, v2_ss8, bucket_metrics
    
def load_chezod_labels():
    from src.evaluation.per_residue_tasks import load_chezod_seth
    _, disorder_scores, train_ids, test_ids = load_chezod_seth(LABELS["chezod_data_dir"])
    return disorder_scores, train_ids, test_ids


def load_trizod_labels():
    from src.evaluation.per_residue_tasks import load_trizod_data
    _, disorder_scores, train_ids, test_ids = load_trizod_data(LABELS["trizod_data_dir"])
    return disorder_scores, train_ids, test_ids
def run_disorder_task(emb,labels,train_ids,test_ids,prefix, raw_results=None,bucket_dict = None):
    # ── Disorder ──
    print(f"  Running {prefix} disorder...")
    v2_dis = run_disorder_benchmark(
        emb, labels, train_ids, test_ids,
        alpha_grid=ALPHA_GRID, cv_folds=CV_FOLDS, seeds=SEEDS,
        n_bootstrap=BOOTSTRAP_N,
    )
    print(f"    {prefix} Pooled rho: {v2_dis['pooled_spearman_rho'].value:.4f}")
    print(f"    {prefix} AUC-ROC: {v2_dis['auc_roc'].value:.4f}")
    metrics = {
        f"{prefix}_disorder_pooled_rho": metric_to_dict(v2_dis["pooled_spearman_rho"]),
        f"{prefix}_disorder_auc_roc": metric_to_dict(v2_dis["auc_roc"]),
        
        f"{prefix}_best_alpha_disorder": v2_dis["best_alpha"],
    }
    bucket_metrics = {}
    if not raw_results is None:
        # Paired cluster bootstrap retention for disorder (pooled rho)
        dis_ret_ci = paired_cluster_bootstrap_retention(
            raw_results["per_protein_predictions"], v2_dis["per_protein_predictions"],
            statistic_fn=_pooled_spearman, n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
        )
        print(f"    Disorder retention: {dis_ret_ci.value:.1f} ± {(dis_ret_ci.ci_upper - dis_ret_ci.ci_lower) / 2:.1f}%")
        metrics[f"{prefix}_disorder_retention"] = metric_to_dict(dis_ret_ci)
        if not bucket_dict is None:
            bucket_metrics = bucket_eval(raw_results["per_protein_predictions"],v2_dis["per_protein_predictions"],bucket_dict=bucket_dict,prefix=prefix,function=paired_cluster_bootstrap_retention,kwargs = {"n_bootstrap":BOOTSTRAP_N, "seed":SEEDS[0],"statistic_fn":_pooled_spearman})
    return metrics, v2_dis, bucket_metrics 
def run_family_task(emb,labels,prefix, raw_results=None,bucket_dict = None,label_key = "family"):
    # ── Retrieval ──
        print(f"  Running {prefix} retrieval...")
        v2_ret = run_retrieval_benchmark(emb, labels, n_bootstrap=BOOTSTRAP_N,label_key=label_key)
        metrics = {
            f"{prefix}_ret1_cosine": metric_to_dict(v2_ret["ret1_cosine"]),
            
        }
        if not raw_results is None:
            ret_ret_ci = paired_bootstrap_retention(
                raw_results["per_query_cosine"], v2_ret["per_query_cosine"],
                n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
            )
            metrics[f"{prefix}_ret1_retention"] =  metric_to_dict(ret_ret_ci)
            print(f"    {prefix} Ret@1 cosine: {v2_ret['ret1_cosine'].value:.4f} (retention: {ret_ret_ci.value:.1f} ± {(ret_ret_ci.ci_upper - ret_ret_ci.ci_lower) / 2:.1f}%)")
            if not bucket_dict is None:
                bucket_metrics = bucket_eval(raw_results["per_query_cosine"], v2_ret["per_query_cosine"],bucket_dict,prefix,paired_bootstrap_retention)
        return metrics, v2_ret, bucket_metrics 
def load_metadata_with_families(prefix = "scope"):
    import csv
    meta = []
    with open(METADATA[prefix]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta.append(row)
    return meta
