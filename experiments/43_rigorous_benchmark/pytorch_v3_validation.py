#!/usr/bin/env python3
""" V3 validation
Only per residue tasks
grabs data from h5s, limiting memory use 
"""
from collections import OrderedDict
import json
import sys
import time
from pathlib import Path
import random
import numpy as np
import os
import h5py

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW_EMBEDDINGS, SPLITS, LABELS, METADATA, RESULTS_DIR,
    SEEDS, BOOTSTRAP_N, C_GRID, ALPHA_GRID, CV_FOLDS,ENCODED_FOLDER,CHECKPOINT_FOLDER
)
from runners.per_residue import run_ss3_benchmark, run_ss8_benchmark, run_disorder_benchmark
from runners.protein_level import compute_protein_vectors, run_retrieval_benchmark
from metrics.statistics import paired_bootstrap_retention, paired_cluster_bootstrap_retention
from rules import MetricResult
from pytorch_based.runners import run_classification_task,run_regression_task,MultiClassTrainer
from metrics.statistics import bootstrap_ci, averaged_multi_seed, cluster_bootstrap_ci

from src.one_embedding.codec_v2 import OneEmbeddingCodec
from src.utils.h5_store import load_residue_embeddings
from pytorch_based.dataloaders import PerResidueRegressionProteinBatchDataset,PerResidueClassificationProteinBatchDataset,split_dataset


#Run settings
#CONTEXT = False
OVERWRITE = False
TESTS = set(["cb513","chezod"])
#TESTS = ["chezod"]
"""("int4_d896_dck_1_no_context",     {"d_out": 896, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d896_dck_1"}),
    ("int4_d768_dck_1_no_context",     {"d_out": 768, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d768_dck_1"}),
    ("int4_d640_dck_1_no_context",     {"d_out": 640, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d640_dck_1"}),
    ("int4_d512_dck_1_no_context",     {"d_out": 512, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d512_dck_1"}),
    ("int4_d384_dck_1_no_context",     {"d_out": 384, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d384_dck_1"}),
    ("int4_d256_dck_1_no_context",     {"d_out": 256, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d256_dck_1"}),
    ("int4_d128_dck_1_no_context",     {"d_out": 128, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d128_dck_1"}),
    ("int4_d64_dck_1_no_context",     {"d_out": 64, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d64_dck_1"}),
    ("int4_d32_dck_1_no_context",     {"d_out": 32, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d32_dck_1"}),
    ("int4_d16_dck_1_no_context",     {"d_out": 16, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d16_dck_1"}),
    ("int4_d8_dck_1_no_context",     {"d_out": 8, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d8_dck_1"}),
    ("int4_d4_dck_1_no_context",     {"d_out": 4, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int4_d4_dck_1"}),
    ("int4_d896_dck_1",     {"d_out": 896, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d896_dck_1"}),
    ("int4_d768_dck_1",     {"d_out": 768, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d768_dck_1"}),
    ("int4_d640_dck_1",     {"d_out": 640, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d640_dck_1"}),
    ("int4_d512_dck_1",     {"d_out": 512, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d512_dck_1"}),
    ("int4_d384_dck_1",     {"d_out": 384, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d384_dck_1"}),
    ("int4_d256_dck_1",     {"d_out": 256, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d256_dck_1"}),
    ("int4_d128_dck_1",     {"d_out": 128, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d128_dck_1"}),
    ("int4_d64_dck_1",     {"d_out": 64, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d64_dck_1"}),
    ("int4_d32_dck_1",     {"d_out": 32, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d32_dck_1"}),
    ("int4_d16_dck_1",     {"d_out": 16, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d16_dck_1"}),
    ("int4_d8_dck_1",     {"d_out": 8, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d8_dck_1"}),
    ("int4_d4_dck_1",     {"d_out": 4, "quantization": "int4",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int4_d4_dck_1"}),
    ("int4_d896_dck_4",     {"d_out": 896, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d896_dck_4"}),
    ("int4_d768_dck_4",     {"d_out": 768, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d768_dck_4"}),
    ("int4_d640_dck_4",     {"d_out": 640, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d640_dck_4"}),
    ("int4_d512_dck_4",     {"d_out": 512, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d512_dck_4"}),
    ("int4_d384_dck_4",     {"d_out": 384, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d384_dck_4"}),
    ("int4_d256_dck_4",     {"d_out": 256, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d256_dck_4"}),
    ("int4_d128_dck_4",     {"d_out": 128, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d128_dck_4"}),
    ("int4_d64_dck_4",     {"d_out": 64, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d64_dck_4"}),
    ("int4_d32_dck_4",     {"d_out": 32, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d32_dck_4"}),
    ("int4_d16_dck_4",     {"d_out": 16, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d16_dck_4"}),
    ("int4_d8_dck_4",     {"d_out": 8, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d8_dck_4"}),
    ("int4_d4_dck_4",     {"d_out": 4, "quantization": "int4",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int4_d4_dck_4"}),"""
""""""
V2_CONFIGS = OrderedDict([
    ("lossless",     {"d_out": 1024, "quantization": None,   "pq_m": None,"dct_k":1,"context":False,"prefix":"baseline_lossless"}),
    ("int2_d896_dck_1_no_context",     {"d_out": 896, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d896_dck_1"}),
    ("int2_d768_dck_1_no_context",     {"d_out": 768, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d768_dck_1"}),
    ("int2_d640_dck_1_no_context",     {"d_out": 640, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d640_dck_1"}),
    ("int2_d512_dck_1_no_context",     {"d_out": 512, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d512_dck_1"}),
    ("int2_d384_dck_1_no_context",     {"d_out": 384, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d384_dck_1"}),
    ("int2_d256_dck_1_no_context",     {"d_out": 256, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d256_dck_1"}),
    ("int2_d128_dck_1_no_context",     {"d_out": 128, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d128_dck_1"}),
    ("int2_d64_dck_1_no_context",     {"d_out": 64, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d64_dck_1"}),
    ("int2_d32_dck_1_no_context",     {"d_out": 32, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d32_dck_1"}),
    ("int2_d16_dck_1_no_context",     {"d_out": 16, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d16_dck_1"}),
    ("int2_d8_dck_1_no_context",     {"d_out": 8, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d8_dck_1"}),
    ("int2_d4_dck_1_no_context",     {"d_out": 4, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":False,"prefix":"int2_d4_dck_1"}),
    ("int2_d896_dck_1",     {"d_out": 896, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d896_dck_1"}),
    ("int2_d768_dck_1",     {"d_out": 768, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d768_dck_1"}),
    ("int2_d640_dck_1",     {"d_out": 640, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d640_dck_1"}),
    ("int2_d512_dck_1",     {"d_out": 512, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d512_dck_1"}),
    ("int2_d384_dck_1",     {"d_out": 384, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d384_dck_1"}),
    ("int2_d256_dck_1",     {"d_out": 256, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d256_dck_1"}),
    ("int2_d128_dck_1",     {"d_out": 128, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d128_dck_1"}),
    ("int2_d64_dck_1",     {"d_out": 64, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d64_dck_1"}),
    ("int2_d32_dck_1",     {"d_out": 32, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d32_dck_1"}),
    ("int2_d16_dck_1",     {"d_out": 16, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d16_dck_1"}),
    ("int2_d8_dck_1",     {"d_out": 8, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d8_dck_1"}),
    ("int2_d4_dck_1",     {"d_out": 4, "quantization": "int2",   "pq_m": None,"dct_k":1,"context":True,"prefix":"int2_d4_dck_1"}),
    ("int2_d896_dck_4",     {"d_out": 896, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d896_dck_4"}),
    ("int2_d768_dck_4",     {"d_out": 768, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d768_dck_4"}),
    ("int2_d640_dck_4",     {"d_out": 640, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d640_dck_4"}),
    ("int2_d512_dck_4",     {"d_out": 512, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d512_dck_4"}),
    ("int2_d384_dck_4",     {"d_out": 384, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d384_dck_4"}),
    ("int2_d256_dck_4",     {"d_out": 256, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d256_dck_4"}),
    ("int2_d128_dck_4",     {"d_out": 128, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d128_dck_4"}),
    ("int2_d64_dck_4",     {"d_out": 64, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d64_dck_4"}),
    ("int2_d32_dck_4",     {"d_out": 32, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d32_dck_4"}),
    ("int2_d16_dck_4",     {"d_out": 16, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d16_dck_4"}),
    ("int2_d8_dck_4",     {"d_out": 8, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d8_dck_4"}),
    ("int2_d4_dck_4",     {"d_out": 4, "quantization": "int2",   "pq_m": None,"dct_k":4,"context":True,"prefix":"int2_d4_dck_4"}),
    
])
results_path = RESULTS_DIR / "quick_test.json"
#SEED = 42


def add_time(dict_,start_time):
    dict_["time_s"] = time.time() - start_time
    return dict_
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


from runners.per_residue import pooled_spearman as _pooled_spearman


def main(config:dict=V2_CONFIGS,tests = ["chezod"],out_json_name ="v2.json"):
    print("=" * 70)
    print("V2 Extreme Compression — Rigorous Re-validation")
    print("BCa CIs, CV-tuned probes, averaged seeds, pooled disorder rho")
    print("=" * 70)
    print()
    out_json_name = RESULTS_DIR / out_json_name
    t_total = time.time()
    
    # ── Load scope for train ──
    emb_dict = RAW_EMBEDDINGS["prot_t5"]
    #maybe should be conditional if something needs to be encoded
    
    
    
    # Trainers
    init_model_params ={
        "num_classes":1, 
        "embed_size":1024, 
        "hidden_dim1":512,  
        "dropout_rate": .1,
        
    } 
    #nominal_grid = { "lr": [0.05,0.0005],"wd": [0.0005, 0.000005],"dropout": [0.1, 0.3] }
    nominal_grid = { "lr": [0.05,0.0005],"wd": [0.0005],"dropout": [0.1, 0.3] }
    #nominal_grid = { "lr": [0.0005],"wd": [0.0005]}
    print(f"Param grid:\n{nominal_grid}")
    nominal_trainer = MultiClassTrainer(init_model_params,)
    
    
   
    
    # ── Encode everything first so codec and scope emb can be removed from memory ──
    # Downside is all h5s need to be written at once, which is issue if hardrive space is limited
    modes = config.keys()
    print(modes)
    
    encoded_h5_dict = {}
    pq_codebook_dict = {}
    train_embs = None
    print("=" * 70)
    print(f"Encoding All Modes")
    print("=" * 70)
   
    for mode in modes:
        
        t0 = time.time()

        
        cfg = config[mode]

        
        
        codec = None
        pq_codebook = None
        encoded_h5_dict[mode] = {}
        if cfg["quantization"] == "pq":
            pq_codebook = ENCODED_FOLDER / f"{cfg["prefix"]}_pq_codebook.h5"
        pq_codebook_dict[mode] = pq_codebook
        for item in tests:
            h5 = ENCODED_FOLDER / f"{cfg["prefix"]}_{item}.h5"
            encoded_h5_dict[mode][item] = h5
            
            if not os.path.isfile(h5) or OVERWRITE:
                print(f"Encoding {h5}")
                if codec == None:
                    codec = OneEmbeddingCodec(d_out=cfg["d_out"], quantization=cfg["quantization"], 
                                        pq_m=cfg["pq_m"],dct_k=(4 if "dct_k" not in cfg else cfg["dct_k"] ),pooling_mode=("dct_k" if "pooling_mode" not in cfg else cfg["pooling_mode"]))
                
                    
                    if train_embs is None:
                        raw_scope = load_residue_embeddings(emb_dict["scope"])
                        scope_split = load_split(SPLITS["scope"])
                        scope_train_ids = scope_split["train_ids"]
                        train_embs = {k: v for k, v in raw_scope.items() if k in set(scope_train_ids)}
                    print(f"  Fitting codebook on {len(train_embs)} train proteins...")
                    codec.fit(train_embs)
                    if pq_codebook and not os.path.isfile(pq_codebook):
                        codec.save_codebook(pq_codebook)
                codec.encode_h5_to_h5(emb_dict[item],h5)
            else:
               print(f"Found encoded h5: {h5}")  
        pq_model= None
        if cfg["quantization"] == "pq":
            if not codec is None:
                pq_model = codec._pq_model   
            elif os.path.isfile(pq_codebook):
                with h5py.File(pq_codebook, "r") as cb:
                    pq_model = {
                        "codebook": cb["pq_codebook"][:],
                        "M": int(cb.attrs["pq_M"]),
                        "n_centroids": int(cb.attrs["pq_K"]),
                        "sub_dim": int(cb.attrs["pq_sub_dim"]),
                        "D": int(cb.attrs["pq_D"]),
                    }
            else:
                raise ValueError("Missing pq_model codebook and no codec")
    if os.path.isfile(out_json_name):
        raw_results, all_results, raw_compare = load_json(out_json_name)
    else:   
        all_results = {}
        
        raw_compare = {}
        raw_results = {}
    print("=" * 70)
    print(f"Eval All Modes")
    print("=" * 70)
    #am lazy, dont want to make sep loader for raw h5s, will use lossless one-embedding :(. also reusing loop, really wierd :(
    tests = set(tests)
    for mode in modes:
        print("=" * 70)
        print(f"starting {mode}\nParams={config[mode]}")
        print("=" * 70)
        if not mode in all_results:
            all_results[mode] = {"Params":config[mode]}
        else:
            #just incase not given arg is added to config
            all_results[mode]["Params"] = config[mode]
        #goal was to simplify adding database, dont think i achieved that lol  
        ss_tests = set(["cb513","casp12","ts115"]) 
        pq_model = pq_codebook_dict.get(mode,None)
        for prefix in tests.intersection(ss_tests):
            start_instance = time.time()
            print(f"running {prefix}")
            if f"{prefix}_q3" in all_results[mode] and f"{prefix}_q8" in all_results[mode]:
                print(f"{prefix}_q3 exists in {out_json_name}")
                continue
            #ss3
            dataset = PerResidueClassificationProteinBatchDataset(encoded_h5_dict[mode][prefix],METADATA[prefix],label_col="ss3",context=(False if not "context" in config[mode] else config[mode]["context"]),codebook=pq_model)
            print(len(dataset))
            train_sub, test_sub = split_dataset(dataset,SPLITS[prefix])
            ss3_metric,ss3_per_protein_acc =run_classification_task(train_sub,test_sub,nominal_trainer,
                                                                    checkpoint_folder=CHECKPOINT_FOLDER,parameter_search_dict=nominal_grid,
                                                                    n_bootstrap=BOOTSTRAP_N,seeds=SEEDS,
                                                                    batch_size=16)
            all_results[mode][f"{prefix}_q3"] = add_time(metric_to_dict(ss3_metric),start_time=start_instance)
            if f"{prefix}_ss3" in raw_compare:
                ss3_ret_ci= paired_bootstrap_retention(
                    raw_compare[f"{prefix}_ss3"], ss3_per_protein_acc,
                    n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
                )
                all_results[mode][f"{prefix}_ss3_retention"] = add_time(metric_to_dict(ss3_ret_ci),start_time=start_instance)
            else:
                raw_compare[f"{prefix}_ss3"] = ss3_per_protein_acc
                raw_results[f"{prefix}_q3"] = all_results[mode][f"{prefix}_q3"]
            
            #ss8
            start_instance = time.time()
            dataset = PerResidueClassificationProteinBatchDataset(encoded_h5_dict[mode][prefix],METADATA[prefix],label_col="ss8",context=(False if not "context" in config[mode] else config[mode]["context"]),codebook=pq_model)
            train_sub, test_sub = split_dataset(dataset,SPLITS[prefix])
            ss8_metric,ss8_per_protein_acc =run_classification_task(train_sub,test_sub,nominal_trainer,
                                                                    checkpoint_folder=CHECKPOINT_FOLDER,
                                                                    parameter_search_dict=nominal_grid,
                                                                    n_bootstrap=BOOTSTRAP_N,seeds=SEEDS,
                                                                    batch_size=16)
            all_results[mode][f"{prefix}_q8"] = add_time(metric_to_dict(ss8_metric),start_time=start_instance)
            if f"{prefix}_ss8" in raw_compare:
                ss8_ret_ci= paired_bootstrap_retention(
                    raw_compare[f"{prefix}_ss8"], ss8_per_protein_acc,
                    n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
                )
                all_results[mode][f"{prefix}_ss8_retention"] = add_time(metric_to_dict(ss8_ret_ci),start_time=start_instance)
            else:
                raw_compare[f"{prefix}_ss8"] = ss8_per_protein_acc
                raw_results[f"{prefix}_q8"] = all_results[mode][f"{prefix}_q8"]
            
            save_output(raw_results,raw_compare,all_results,nominal_grid,out_json_name)
        dis_tests = set(["chezod","trizod"]) 
        for prefix in tests.intersection(dis_tests):
            if f"{prefix}_spearman" in all_results[mode]:
                print(f"{prefix}_spearman exists in {out_json_name}")
                continue
            print(f"running {prefix}")
            start_instance = time.time()
            dataset = PerResidueRegressionProteinBatchDataset(encoded_h5_dict[mode][prefix],METADATA[prefix],label_col="disorder",context=(False if not "context" in config[mode] else config[mode]["context"]),codebook=pq_model)
           
            train_sub, test_sub = split_dataset(dataset,SPLITS[prefix])
            spearman_metric,cluster_avg =run_regression_task(train_sub,test_sub,nominal_trainer,
                                                                    checkpoint_folder=CHECKPOINT_FOLDER,
                                                                    parameter_search_dict=nominal_grid,
                                                                    n_bootstrap=BOOTSTRAP_N,seeds=SEEDS,
                                                                    batch_size=16)
            all_results[mode][f"{prefix}_spearman"] = add_time(metric_to_dict(spearman_metric),start_time=start_instance)
            if f"{prefix}_cluster" in raw_compare:
                spearman_ret_ci= paired_cluster_bootstrap_retention(
                    raw_compare[f"{prefix}_cluster"], cluster_avg,_pooled_spearman,
                    n_bootstrap=BOOTSTRAP_N, seed=SEEDS[0],
                )
                all_results[mode][f"{prefix}_spearman_retention"] = add_time(metric_to_dict(spearman_ret_ci),start_time=start_instance)
            else:
                raw_compare[f"{prefix}_cluster"] = cluster_avg
                raw_results[f"{prefix}_spearman"] = all_results[mode][f"{prefix}_spearman"]
            
            save_output(raw_results,raw_compare,all_results,nominal_grid,out_json_name)

    #output
def save_output(raw_results,raw_compare,all_results,nominal_grid,results_path): 
    output = {
        "raw_baselines": raw_results,
        "modes": all_results,
        "_meta": {
            "script": "test.py",
            "methodology": "BCa bootstrap, CV-tuned probes, averaged 3-seed, pooled disorder rho",
            "seeds": SEEDS,
            "n_bootstrap": BOOTSTRAP_N,
            "param_grid": nominal_grid,
            "alpha_grid": ALPHA_GRID,
        },
        "raw_compare": raw_compare
    }
    
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_path}")
def load_json(json):
    d = load_split(json)
    return d["raw_baselines"], d["modes"], d["raw_compare"]
    
    
        
        
        
       
        
        

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
