#!/usr/bin/env python3
"""V2 Extreme Compression — Rigorous Re-validation.

Runs all V2 codec modes (full, balanced, compact, micro, binary) through
the Exp 43 rigorous benchmark framework with:
- BCa bootstrap CIs
- CV-tuned probes (GridSearchCV)
- Averaged multi-seed predictions
- Pooled Spearman rho for disorder (SETH/CAID standard)
- Fair retrieval baselines

This replaces the Exp 34 results which used hardcoded C=1.0 probes.
"""
from collections import OrderedDict
import json
import sys
import time
import re
import os
#os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import random
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW_EMBEDDINGS, SPLITS, LABELS, METADATA, RESULTS_DIR,
    SEEDS, BOOTSTRAP_N, C_GRID, ALPHA_GRID, CV_FOLDS, BUCKETS
)
from runners.per_residue import run_ss3_benchmark, run_ss8_benchmark, run_disorder_benchmark
from runners.protein_level import compute_protein_vectors, run_retrieval_benchmark
from metrics.statistics import paired_bootstrap_retention, paired_cluster_bootstrap_retention
from rules import MetricResult
from runners.localization import (
    parse_deeploc_metadata_csv,
    run_localization_probe_benchmark,
    macro_f1_from_clusters,
    extract_protein_features_from_codec,
)
from helpers.loaders_runners import load_ss3_labels,run_ss3_task,load_chezod_labels,load_trizod_labels,run_disorder_task,metric_to_dict,load_metadata_with_families,load_split,bucket_eval, run_family_task

from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.utils.h5_store import load_residue_embeddings,load_protein_embeddings
from runners.per_residue import pooled_spearman as _pooled_spearman
context_off = True
max_length= 512
# V2 mode configs (512d for historical reproducibility with Exp 34 / Exp 43 V2 tiers)

V2_CONFIGS = OrderedDict([
    ("PQ512", {"d_out": 512, "quantization": "pq",     "pq_m": 512,"dct_k":1,"pooling_mode":"dct_k",  "desc": "PQ512"}),
    ("PQ256", {"d_out": 512, "quantization": "pq",     "pq_m": 256,"dct_k":1,"pooling_mode":"dct_k",  "desc": "256"}),
    ("PQ128",  {"d_out": 512, "quantization": "pq",     "pq_m": 128,"dct_k":1,"pooling_mode":"dct_k",   "desc": "PQ M=128 "}),
    ("PQ64",  {"d_out": 512, "quantization": "pq",     "pq_m": 64,"dct_k":1,"pooling_mode":"dct_k",   "desc": "PQ M=64"}),
    ("PQ32",  {"d_out": 512, "quantization": "pq",     "pq_m": 32,"dct_k":1,"pooling_mode":"dct_k",   "desc": "PQ M=32"}),
    ("PQ16",  {"d_out": 512, "quantization": "pq",     "pq_m": 16,"dct_k":1,"pooling_mode":"dct_k",   "desc": "PQ M=16"}),
    ("PQ8",  {"d_out": 512, "quantization": "pq",     "pq_m": 8,"dct_k":1,"pooling_mode":"dct_k",   "desc": "PQ M=8"}),
    ("PQ4",  {"d_out": 512, "quantization": "pq",     "pq_m": 4,"dct_k":1,"pooling_mode":"dct_k",   "desc": "PQ M=4"}),
    ("PQ2",  {"d_out": 512, "quantization": "pq",     "pq_m": 2,"dct_k":1,"pooling_mode":"dct_k",   "desc": "PQ M=2"}),
    
    
])


#context_off = False

["chezod","casp12","scope","deeploc"]
def main(config:dict=V2_CONFIGS,tests = ["deeploc"],out_json_name ="v2.json" ):
    results_path = RESULTS_DIR / out_json_name
    print("=" * 70)
    print("V2 Extreme Compression — Rigorous Re-validation")
    print("BCa CIs, CV-tuned probes, averaged seeds, pooled disorder rho")
    print("=" * 70)
    print()
    if os.path.isfile(results_path):
        print("loaded")
        all_results, bucket_results,raw_results_dict = load_json_out(results_path)
        
        
    else:
        all_results = {}
        bucket_results = {}
        raw_results_dict = {}
    t_total = time.time()

    # ── Load data ──
    print("Loading data...")
    raw_dict = RAW_EMBEDDINGS["prot_t5"]
    scope_raw_emb = load_residue_embeddings(raw_dict["scope"])

    scope_splits = load_split(SPLITS["scope"])
    scope_train_ids = scope_splits["train_ids"]
    train_embs = {k: v for k, v in scope_raw_emb.items() if k in set(scope_train_ids)}
    print(f"  SCOPe: {len(scope_raw_emb)} proteins")
    
    
    #secondary structure
    if "cb513" in tests:
        cb513_raw_emb, cb513_ss3_labels,cb513_ss8_labels, cb_train, cb_test = load_ss3_labels("cb513",raw_dict=raw_dict)
        cb513_bucket = load_split(BUCKETS["cb513"])
        print(f"  CB513: {len(cb_train)} train, {len(cb_test)} test")
        metrics, cb513_ss3,cb513_ss8, _ = run_ss3_task(cb513_raw_emb,cb513_ss3_labels,cb513_ss8_labels,cb_train,cb_test,"cb513")
        raw_results_dict.update(metrics)
    if "casp12" in tests:
        casp12_raw_emb, casp12_ss3_labels,casp12_ss8_labels, casp12_train, casp12_test = load_ss3_labels("casp12",raw_dict=raw_dict)
        casp12_bucket = load_split(BUCKETS["casp12"])
        print(f"  casp12: {len(casp12_train)} train, {len(casp12_test)} test")
        metrics, casp12_ss3,casp12_ss8, _ = run_ss3_task(casp12_raw_emb,casp12_ss3_labels,casp12_ss8_labels,casp12_train,casp12_test,"casp12")
        raw_results_dict.update(metrics)
    if "ts115" in tests:
        ts115_raw_emb, ts115_ss3_labels,ts115_ss8_labels, ts115_train, ts115_test = load_ss3_labels("ts115",raw_dict=raw_dict)
        ts115_bucket = load_split(BUCKETS["ts115"])
        print(f"  ts115: {len(ts115_train)} train, {len(ts115_test)} test")
        metrics, ts115_ss3,ts115_ss8, _ = run_ss3_task(ts115_raw_emb,ts115_ss3_labels,ts115_ss8_labels,ts115_train,ts115_test,"ts115")
        raw_results_dict.update(metrics)

    #disorder
    if "chezod" in tests:
        chezod_raw_emb = load_residue_embeddings(raw_dict["chezod"])
        chezod_disorder_scores, chezod_dis_train_ids, chezod_dis_test_ids = load_chezod_labels()
        chezod_dis_train_ids = [p for p in chezod_dis_train_ids if p in chezod_raw_emb and p in chezod_disorder_scores]
        chezod_dis_test_ids = [p for p in chezod_dis_test_ids if p in chezod_raw_emb and p in chezod_disorder_scores]
        chezod_bucket = load_split(BUCKETS["chezod"])
        print(f"  CheZOD: {len(chezod_dis_train_ids)} train, {len(chezod_dis_test_ids)} test")
        metrics, chezod_raw, _ = run_disorder_task(chezod_raw_emb,chezod_disorder_scores,chezod_dis_train_ids,chezod_dis_test_ids,prefix="chezod")
        raw_results_dict.update(metrics)
    if "trizod" in tests:
        raw_trizod_emb = load_residue_embeddings(raw_dict["trizod"])
        trizod_scores, trizod_train_ids, trizod_test_ids = load_trizod_labels()
        trizod_ids = load_split(SPLITS["trizod"])
        # Filter to available
        trizod_train_ids = [p for p in trizod_train_ids if p in raw_trizod_emb and p in trizod_scores and p in trizod_ids["train_ids"]]
        trizod_test_ids  = [p for p in trizod_test_ids  if p in raw_trizod_emb and p in trizod_scores and p in trizod_ids["test_ids"]]
        
        trizod_bucket = load_split(BUCKETS["trizod"])
        print(f"  TriZOD: {len(trizod_train_ids)} train, {len(trizod_test_ids)} test")
        metrics, trizod_raw, _ = run_disorder_task(raw_trizod_emb,trizod_scores,trizod_train_ids,trizod_test_ids,prefix="trizod")
        raw_results_dict.update(metrics)

    #family    
    if "scope" in tests or "scope_superfamily" in tests:
        scope_metadata = load_metadata_with_families()
        scope_raw_ret_vecs = compute_protein_vectors(scope_raw_emb, method="dct_k4")
        
        scope_bucket = load_split(BUCKETS["scope"])
        if "scope" in tests:
            scope_raw_ret = run_retrieval_benchmark(scope_raw_ret_vecs, scope_metadata, n_bootstrap=BOOTSTRAP_N)
            print(f"  Scope Raw Ret@1 cosine: {scope_raw_ret['ret1_cosine'].value:.4f}")
            scope_metrics = {
                "scope_ret1_cosine": metric_to_dict(scope_raw_ret["ret1_cosine"]),
                
            }
            raw_results_dict.update(scope_metrics)
        if "scope_superfamily" in tests:
       
            scope_super_raw_ret = run_retrieval_benchmark(scope_raw_ret_vecs, scope_metadata, n_bootstrap=BOOTSTRAP_N,label_key="superfamily")

            print(f"  Superfamily Scope Raw Ret@1 cosine: {scope_super_raw_ret['ret1_cosine'].value:.4f}")
            scope_metrics = {
                "scope_ret1_cosine": metric_to_dict(scope_super_raw_ret["ret1_cosine"]),
                
            }
            raw_results_dict.update(scope_metrics)
    if "cath20" in tests:
        CATH20_LABEL = "superfamily"
        cath20_metadata = load_metadata_with_families(prefix="cath20")
        cath20_raw_ret_vecs = load_protein_embeddings(raw_dict["cath20"])
        
        cath20_raw_ret = run_retrieval_benchmark(cath20_raw_ret_vecs, cath20_metadata, n_bootstrap=BOOTSTRAP_N,label_key=CATH20_LABEL)
        cath20_bucket = load_split(BUCKETS["cath20"])
        print(f"  Raw cath20 Ret@1 cosine: {cath20_raw_ret['ret1_cosine'].value:.4f}")
        cath20_metrics = {
            "cath20_ret1_cosine": metric_to_dict(cath20_raw_ret["ret1_cosine"]),
            
        }
        raw_results_dict.update(cath20_metrics)

    #localization
    if "deeploc" in tests: 
        t0 = time.time()
        raw_deeploc = load_protein_embeddings(raw_dict["deeploc"],max_length=2000)
        print(time.time() -t0)
        deeploc_csv = METADATA.get("deeploc")
        deeploc_train_labels, deeploc_test_labels = parse_deeploc_metadata_csv(deeploc_csv, max_length=2000)
        raw_ids = set(raw_deeploc.keys())
        deeploc_train_ids = sorted(raw_ids & set(deeploc_train_labels.keys()))
        deeploc_test_ids = sorted(raw_ids & set(deeploc_test_labels.keys()))
        deeploc_train_labels = {pid: deeploc_train_labels[pid] for pid in deeploc_train_ids}
        deeploc_test_labels = {pid: deeploc_test_labels[pid] for pid in deeploc_test_ids}
        deeploc_bucket = load_split(BUCKETS["deeploc"])
        
        raw_deeploc = {
            pid: raw_deeploc[pid]
            for pid in sorted(set(deeploc_train_ids) | set(deeploc_test_ids))
        }
        deeploc_total_ids = set(raw_deeploc.keys())
        print(f"DeepLoc embeddings used: {len(raw_deeploc)}")
        print(f"DeepLoc train labels:    {len(deeploc_train_labels)}")
        print(f"DeepLoc test labels:     {len(deeploc_test_labels)}")
        
        
        raw_loc = run_localization_probe_benchmark(
            embeddings=raw_deeploc,
            train_labels=deeploc_train_labels,
            test_labels=deeploc_test_labels,
            test_name="deeploc1_raw",
            C_grid=C_GRID,
            cv_folds=CV_FOLDS,
            seeds=SEEDS,
            n_bootstrap=BOOTSTRAP_N,
            
        )
        deeploc_metrics = {
            "deeploc_q10": metric_to_dict(raw_loc["q10"]),
            "deeploc_macro_f1": metric_to_dict(raw_loc["macro_f1"])
        }
        raw_results_dict.update(deeploc_metrics)
    
    # ── Benchmark each V2 mode ──
    
    modes = config.keys()
    print(modes)
    
    for mode in modes:
        print("=" * 70)
        print(f"  MODE: {mode} — {config[mode]['desc'] if "desc" in config[mode] else "No Discription"}\nParams:{config[mode]}")
        print("=" * 70)
        t0 = time.time()
        
        # Fit codec on SCOPe train
        cfg = config[mode]
        if not mode in all_results:
            all_results[mode] = {"Params":config[mode]}
            bucket_results[mode] = {"Params":cfg}
        else:
            #just incase not given arg is added to config
            all_results[mode]["Params"] = config[mode]
            bucket_results[mode]["Params"] = config[mode]
        
        
        codec = OneEmbeddingCodec(d_out=cfg["d_out"], quantization=cfg["quantization"], pq_m=cfg["pq_m"],dct_k=(4 if "dct_k" not in cfg else cfg["dct_k"] ),pooling_mode=("dct_k" if "pooling_mode" not in cfg else cfg["pooling_mode"]))
        
        print(f"  Fitting codebook on {len(train_embs)} train proteins...")
        codec.fit(train_embs)
        print("Starting tests")
        if "cb513" in tests:
            if not "cb513_ss3_q3" in all_results[mode]:
                
            # Encode + decode CB513
                decoded = {}
                for pid, emb in cb513_raw_emb.items():
                    enc = codec.encode(emb)
                    decoded[pid] = codec.decode_per_residue(enc)
                metrics,_,_,bucket_metrics = run_ss3_task(decoded,cb513_ss3_labels,cb513_ss8_labels,cb_train,cb_test,"cb513", cb513_ss3,cb513_ss8,cb513_bucket)
                all_results[mode].update(metrics)
                bucket_results[mode].update(bucket_metrics)
                save_json_out(raw_results_dict,all_results,bucket_results,results_path)
        if "casp12" in tests:
            
            if not "casp12_ss3_q3" in all_results[mode]:
                    
            # Encode + decode casp12
                decoded = {}
                for pid, emb in casp12_raw_emb.items():
                    enc = codec.encode(emb)
                    decoded[pid] = codec.decode_per_residue(enc)
                metrics,_,_,bucket_metrics = run_ss3_task(decoded,casp12_ss3_labels,casp12_ss8_labels,casp12_train,casp12_test,"casp12", casp12_ss3,casp12_ss8,casp12_bucket)
                all_results[mode].update(metrics)
                bucket_results[mode].update(bucket_metrics)
                save_json_out(raw_results_dict,all_results,bucket_results,results_path)
        if "ts115" in tests:
            if not "ts115_ss3_q3" in all_results[mode]:
            
            # Encode + decode ts115
                decoded = {}
                for pid, emb in ts115_raw_emb.items():
                    enc = codec.encode(emb)
                    decoded[pid] = codec.decode_per_residue(enc)
                metrics,_,_,bucket_metrics = run_ss3_task(decoded,ts115_ss3_labels,ts115_ss8_labels,ts115_train,ts115_test,"ts115", ts115_ss3,ts115_ss8,ts115_bucket)
                all_results[mode].update(metrics)
                bucket_results[mode].update(bucket_metrics)
                save_json_out(raw_results_dict,all_results,bucket_results,results_path)
        if "chezod" in tests:
            if not "chezod_disorder_pooled_rho" in all_results[mode]:
                
                # Encode + decode CheZOD
                decoded = {}
                
                for pid, emb in chezod_raw_emb.items():
                    if pid in chezod_disorder_scores:
                        enc = codec.encode(emb)
                        decoded[pid] = codec.decode_per_residue(enc)
                metrics,_,bucket_metrics  = run_disorder_task(decoded,chezod_disorder_scores,chezod_dis_train_ids,chezod_dis_test_ids,"chezod",chezod_raw,chezod_bucket)    
                all_results[mode].update(metrics)
                bucket_results[mode].update(bucket_metrics)
                save_json_out(raw_results_dict,all_results,bucket_results,results_path)
        if "trizod" in tests:
            if not "trizod_disorder_pooled_rho" in all_results[mode]:
                
                # Encode + decode TriZOD
                decoded = {}
                for pid, emb in raw_trizod_emb.items():
                    if pid in trizod_scores:
                        enc = codec.encode(emb)
                        decoded[pid] = codec.decode_per_residue(enc)
                # ── Full TriZOD disorder ──
                metrics, _,bucket_metrics = run_disorder_task(decoded,trizod_scores,trizod_train_ids,trizod_test_ids,"trizod",trizod_raw,trizod_bucket)
                all_results[mode].update(metrics)
                bucket_results[mode].update(bucket_metrics)
                save_json_out(raw_results_dict,all_results,bucket_results,results_path)
        if "scope" in tests or "scope_superfamily" in tests:
            if not "scope_ret1_retention" in all_results[mode] or not "scope_super_ret1_retention" in all_results[mode]:
                
                # Encode SCOPe for retrieval (protein vectors)
                decoded = {}
                for pid, emb in scope_raw_emb.items():
                    enc = codec.encode(emb)
                    decoded[pid] = enc["protein_vec"].astype(np.float32)
                if "scope" in tests and not "scope_ret1_retention" in all_results[mode]:
                    metrics, _, bucket_metrics = run_family_task(decoded,scope_metadata,"scope", raw_results=scope_raw_ret,bucket_dict = scope_bucket)
                    all_results[mode].update(metrics)
                    bucket_results[mode].update(bucket_metrics)
                    save_json_out(raw_results_dict,all_results,bucket_results,results_path)
                if "scope_superfamily" in tests and not "scope_super_ret1_retention" in all_results[mode]:
                    metrics, _, bucket_metrics = run_family_task(decoded,scope_metadata,"scope_super", raw_results=scope_super_raw_ret,bucket_dict = scope_bucket,label_key="superfamily")
                    all_results[mode].update(metrics)
                    bucket_results[mode].update(bucket_metrics)
                    save_json_out(raw_results_dict,all_results,bucket_results,results_path)
        if "cath20" in tests:
            if not "cath20_ret1_retention" in all_results[mode]:
                
                # Encode CATH20 for retrieval (protein vectors)
                decoded = load_protein_embeddings(raw_dict["cath20"],codec=codec)
                metrics, _, bucket_metrics = run_family_task(decoded,cath20_metadata,"cath20", raw_results=cath20_raw_ret,bucket_dict = cath20_bucket,label_key=CATH20_LABEL)
                all_results[mode].update(metrics)
                bucket_results[mode].update(bucket_metrics)
                save_json_out(raw_results_dict,all_results,bucket_results,results_path)
        if "deeploc" in tests:
            if not "deeploc_q10" in all_results[mode]:
                
                # Encode + decode Deeploc
                decoded = load_protein_embeddings(raw_dict["deeploc"],max_length=2000,codec=codec,keys=deeploc_total_ids)
                v2_loc = run_localization_probe_benchmark(
                    embeddings=decoded,
                    train_labels=deeploc_train_labels,
                    test_labels=deeploc_test_labels,
                    test_name=f"deeploc1_{mode}",
                    C_grid=C_GRID,
                    cv_folds=CV_FOLDS,
                    seeds=SEEDS,
                    n_bootstrap=BOOTSTRAP_N,
                    
                )
                loc_q10_ret_ci = paired_bootstrap_retention(
                    raw_loc["per_protein_scores"],
                    v2_loc["per_protein_scores"],
                    n_bootstrap=BOOTSTRAP_N,
                    seed=SEEDS[0],
                )

                loc_macro_ret_ci = paired_cluster_bootstrap_retention(
                    raw_loc["per_protein_predictions"],
                    v2_loc["per_protein_predictions"],
                    statistic_fn=macro_f1_from_clusters,
                    n_bootstrap=BOOTSTRAP_N,
                    seed=SEEDS[0],
                )
                deeploc_metrics = {
                    "deeploc_q10": metric_to_dict(v2_loc["q10"]),
                    "deeploc_macro_f1": metric_to_dict(v2_loc["macro_f1"]),
                    "deeploc_q10_retention": metric_to_dict(loc_q10_ret_ci),
                    "deeploc_macro_f1_retention": metric_to_dict(loc_macro_ret_ci)
                }
                bucket_metrics = bucket_eval(raw_loc["per_protein_predictions"], v2_loc["per_protein_predictions"],deeploc_bucket,"deeploc",paired_cluster_bootstrap_retention, kwargs={"n_bootstrap":BOOTSTRAP_N, "seed":SEEDS[0],"statistic_fn":macro_f1_from_clusters})
                all_results[mode].update(deeploc_metrics)
                bucket_results[mode].update(bucket_metrics)
                save_json_out(raw_results_dict,all_results,bucket_results,results_path)
        elapsed = time.time() - t0
        print(f"  Mode {mode} took {elapsed:.1f}s")
        
        all_results[mode]["time_s"]=  elapsed
        save_json_out(raw_results_dict,all_results,bucket_results,results_path)
    print(f"Total time: {time.time() - t_total:.1f}s")  
            
           
            
           
        
def load_json_out(json_path):
    d = load_split(json_path)
    return d["modes"], d.get("bucket_modes",{}),d["raw_baselines"]
def save_json_out(raw_results_dict,all_results,bucket_results,results_path):
    # ── Save results ──
    output = {
        "raw_baselines": raw_results_dict,
        "modes": all_results,
        "bucket_modes": bucket_results,
        "_meta": {
            "script": "run_v2_validation.py",
            "methodology": "BCa bootstrap, CV-tuned probes, averaged 3-seed, pooled disorder rho",
            "seeds": SEEDS,
            "n_bootstrap": BOOTSTRAP_N,
            "C_grid": C_GRID,
            "alpha_grid": ALPHA_GRID,
        },
        
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_path}")
    

def read_input_config(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)["config"]
    return OrderedDict(config)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="run tests")
    parser.add_argument("--tests",required=True,nargs="+",  help="tests to run")
    parser.add_argument("--out_json_name","-o",required=True,  help="just filename no path (test.json)")
    parser.add_argument("--input_config","-c",help="V2 config input json, see /input_configs/pq_m_sweep.json as example")
    args = parser.parse_args()
    config = V2_CONFIGS
    if args.input_config:
        config = read_input_config(args.input_config)
    main(tests=args.tests,out_json_name=args.out_json_name,config=config)
