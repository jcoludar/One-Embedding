






import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    LABELS,METADATA
    
)

def load_ss_labels(prefix):
    from src.evaluation.per_residue_tasks import load_ss_csv
    _, ss3, ss8, _ = load_ss_csv(LABELS[prefix],prefix)
    return ss3, ss8


def load_chezod_labels():
    from src.evaluation.per_residue_tasks import load_chezod_seth
    _, disorder_scores, train_ids, test_ids = load_chezod_seth(LABELS["chezod_data_dir"])
    return disorder_scores, train_ids, test_ids
def load_trizod_labels():
    from src.evaluation.per_residue_tasks import load_trizod_data
    _, disorder_scores, train_ids, test_ids = load_trizod_data(LABELS["trizod_data_dir"])
    return disorder_scores, train_ids, test_ids

def dict_to_per_residue_df(in_dict, id_col ="id", label_col = "label",idx_col ="idx"):
    df = pd.Series(in_dict).to_frame(label_col)
    df[id_col] = df.index
    df = df.reset_index()
    df = df.explode(label_col)
    
    df[idx_col] = df.groupby(id_col).cumcount()
    return df
def dict_to_per_residue_csv(in_dict,csv, id_col ="id", label_col = "label",idx_col ="idx"):

    df = dict_to_per_residue_df(in_dict,id_col=id_col,label_col=label_col,idx_col=idx_col)
    
    df = df[[id_col,idx_col,label_col]]
    #print(df)
    df.to_csv(csv,index=False)   
def dict_to_per_residue_csv_ss3_8(in_dicts,csv, id_col ="id", label_cols = ["ss3","ss8"],idx_col ="idx"):

    df = dict_to_per_residue_df(in_dicts[0],id_col=id_col,label_col=label_cols[0],idx_col=idx_col)
    for x in range(1,len(in_dicts)):
        df_temp = dict_to_per_residue_df(in_dicts[x],id_col=id_col,label_col=label_cols[x],idx_col=idx_col)
        df = df.merge(df_temp, how="inner",on = [id_col,idx_col])
    df = df[([id_col,idx_col]+label_cols)]
    print(df)
    df.to_csv(csv,index=False)   
def break_ss_to_list(in_dict):
    out_dict = {}
    for key, item in in_dict.items():
        out_dict[key] = list(str(item))
    return out_dict
ss3, ss8 = load_ss_labels("cb513")
ss3 = break_ss_to_list(ss3)
ss8 = break_ss_to_list(ss8)
dict_to_per_residue_csv_ss3_8([ss3,ss8],METADATA["cb513"])
ss3, ss8 = load_ss_labels("casp12")
ss3 = break_ss_to_list(ss3)
ss8 = break_ss_to_list(ss8)
dict_to_per_residue_csv_ss3_8([ss3,ss8],METADATA["casp12"])
ss3, ss8 = load_ss_labels("ts115")
ss3 = break_ss_to_list(ss3)
ss8 = break_ss_to_list(ss8)
dict_to_per_residue_csv_ss3_8([ss3,ss8],METADATA["ts115"])
disorder_scores, train_ids, test_ids =load_chezod_labels()
#print(disorder_scores["26672"])
dict_to_per_residue_csv(disorder_scores,METADATA["chezod"],label_col="disorder")
disorder_scores, train_ids, test_ids =load_trizod_labels()
#print(disorder_scores["26672"])
dict_to_per_residue_csv(disorder_scores,METADATA["trizod"],label_col="disorder")

