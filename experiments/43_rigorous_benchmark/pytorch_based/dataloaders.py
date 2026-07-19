import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import h5py
import sys 
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.one_embedding.codec_v2 import OneEmbeddingCodec


#this class is never used, and is redundent, but is inherited from by used datasets, should fix
class PerResidueRegressionDataset(Dataset):
    def __init__(self, embeddings_path: str, csv_path: str,
                 id_col: str="id",label_col: str = "family",
                 idx_col = "idx",context:bool = False,
                 codebook:str = None,pq_model:dict = None):
        super().__init__()
        #core data
        self.embeddings_file_path = embeddings_path
        self.metadata_df = pd.read_csv(csv_path)
        self.h5f = h5py.File(self.embeddings_file_path, 'r')
        file_meta = json.loads(self.h5f.attrs["metadata"])
        self.quantization = file_meta.get("quantization")
        self.d_out = file_meta["d_out"]
        self.dct_k = file_meta["dct_k"]
        self.pq_model = None
        if self.quantization == "pq":
            self.pq_model = pq_model
            if pq_model is None and not codebook is None :
                with h5py.File(codebook, "r") as cb:
                    self.pq_model = {
                        "codebook": cb["pq_codebook"][:],
                        "M": int(cb.attrs["pq_M"]),
                        "n_centroids": int(cb.attrs["pq_K"]),
                        "sub_dim": int(cb.attrs["pq_sub_dim"]),
                        "D": int(cb.attrs["pq_D"]),
                    }
            else:
                raise ValueError("codebook_path or pq_model dict required for PQ modes")
        #calc output dim for embeddings
        self.embedding_dim = self.d_out
        if context:
            self.embedding_dim=self.embedding_dim*(self.dct_k+1)
        self.context = context
        self.id_column = id_col
        self.label_column = label_col
        self.idx_column = idx_col
        self.h5f = None
        if self.id_column not in self.metadata_df.columns or self.label_column not in self.metadata_df.columns:
            raise ValueError(f"Required columns '{self.id_column}' or '{self.label_column}' not found in {csv_path}.")
        
        
        
        
        with h5py.File(self.embeddings_file_path, "r") as h5f:
            embedding_ids = set(h5f.keys())
        
        original_len = len(self.metadata_df)
        self.metadata_df = self.metadata_df[
            self.metadata_df[self.id_column].isin(embedding_ids)
        ].reset_index(drop=True)
        csv_filter_length = len(self.metadata_df)
        if original_len != csv_filter_length:
            print(f"INFO: Filtered metadata for HDF5 keys. Kept {len(self.metadata_df)}/{original_len} entries.")
        self.metadata_df =  self.metadata_df.dropna()
        nan_length = len(self.metadata_df)
        if nan_length!= csv_filter_length:
            print(f"INFO: Removed NAN lables. Kept {nan_length}/{csv_filter_length} entries.")
        
        
    def __getitem__(self, index):
        if self.h5f is None:
            self.h5f = h5py.File(self.embeddings_file_path, 'r')
        row = self.metadata_df.iloc[index]
        
        grp = self.h5f[row[self.id_column]]
        #len = int(grp.attrs["seq_len"])
        
        #print(f"{row[self.id_column]} {row[self.idx_column]} {row[self.label_column]}")
        emb = OneEmbeddingCodec.read_single_residue_from_h5(grp,self.quantization,self.d_out,row[self.idx_column],pq_model=self.pq_model)
        if self.context:
            protein_vec = grp["protein_vec"][:]
            emb = np.concatenate([emb,protein_vec],axis=None)
        #print(emb.shape)
        #print(self.embedding_dim)
        if emb.shape[0] != self.embedding_dim:
            raise ValueError("Incorrect embedding shape")
        label = torch.tensor(row[self.label_column],dtype=torch.float32)
        emb = torch.tensor(emb,dtype=torch.float32)
        id = np.array([str(row[self.id_column])])
        idx = np.array([str(row[self.idx_column])])
        return emb, label, id, idx
    def __len__(self):
        return len(self.metadata_df)
    def get_labels(self):
        return None
    def get_classes(self):
        return None
    def __del__(self):
        """Close the HDF5 handle when the dataset is destroyed."""
        if getattr(self, 'h5f', None) is not None:
            try:
                self.h5f.close()
            except Exception:
                pass
            self.h5f = None

class PerResidueRegressionProteinBatchDataset(PerResidueRegressionDataset):
    def __init__(self, embeddings_path: str, csv_path: str,
                 id_col: str="id",label_col: str = "family",
                 idx_col = "idx",context:bool = False,
                 codebook:str = None,pq_model:dict = None):
        super().__init__(
            embeddings_path,csv_path,id_col,label_col,idx_col,context,codebook,pq_model=pq_model
        )
        self.unique_protein = list(set(self.metadata_df[id_col]))

    def __len__(self):
        return len(self.unique_protein)
    def __getitem__(self, index):
        if self.h5f is None:
            self.h5f = h5py.File(self.embeddings_file_path, 'r')
        
        id = self.unique_protein[index]
        
        rows = self.metadata_df[self.metadata_df[self.id_column]==id ]
        
        
        grp = self.h5f[id]
        lenght_prot = int(grp.attrs["seq_len"])
        
        #print(f"{row[self.id_column]} {row[self.idx_column]} {row[self.label_column]}")
        emb = OneEmbeddingCodec._read_per_residue_from_h5(grp,self.quantization,lenght_prot,self.d_out,pq_model=self.pq_model)[rows[self.
        idx_column]]
        lenght_prot = len(rows)
        if self.context:
            protein_vec = grp["protein_vec"][:][np.newaxis,:]
            long_protein_vec = np.repeat(protein_vec,lenght_prot,axis=0)
            emb = np.hstack([emb,long_protein_vec])
            
            
        #print(emb.shape)
        #print(self.embedding_dim)
        if emb.shape[1] != self.embedding_dim:
            raise ValueError("Incorrect embedding shape")
        label = torch.tensor(np.vstack(rows[self.label_column]),dtype=torch.float32)
        emb = torch.tensor(emb,dtype=torch.float32)
        ids= [id]*lenght_prot
        ids = np.array(ids)
        idxs = np.array(rows[self.idx_column])
        return emb, label, ids,idxs
    #for stratifying#
    def get_labels(self):
        #return list(self.metadata_df.groupby(self.id_column)[self.label_column].mean())
        return None
class PerResidueClassificationProteinBatchDataset(PerResidueRegressionProteinBatchDataset):
    def __init__(self, embeddings_path: str, csv_path: str,
                 id_col: str="id",label_col: str = "family",
                 idx_col = "idx",context:bool = False,
                 codebook:str = None,pq_model:dict = None,
                 encoder:OneHotEncoder = None):
        super().__init__(
            embeddings_path,csv_path,id_col,label_col,idx_col,context,codebook,pq_model=pq_model
        )
        
        if encoder is None:
            self.encoder = OneHotEncoder(sparse_output=False).fit(
                self.metadata_df[[self.label_column]].values
            )
        else:
            self.encoder = encoder
            known_genes = self.encoder.categories_[0]
            self.metadata_df = self.metadata_df[self.metadata_df[self.label_column].isin(known_genes)]
        encodings = self.encoder.transform(
            self.metadata_df[[self.label_column]].values
        )
        self.metadata_df[label_col] = list(encodings)
    def __len__(self):
        return len(self.unique_protein)
    def get_classes(self):
        return self.encoder.categories_[0]
    #for stratifying
    def get_labels(self):
        """
        #get random sample, based on percentage, idea being if a
        encodings = self.metadata_df.groupby(self.id_column).apply(lambda x: x.sample(1,random_state=42))[self.label_column]
        encodings = np.vstack(encodings)
        return np.argmax(encodings,axis=1)
        """
        return None

#subsets with get functions
class PerResidueRegressionSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.data = dataset.metadata_df.iloc[indices]
        self.embedding_dim = dataset.embedding_dim
        self.label_column = dataset.label_column
        
    def get_embedding_dim(self):
        return self.embedding_dim
    def get_labels(self):
        return None#
    def get_classes(self):
        return None
    def get_data(self):
        return self.data
class PerResidueRegressionProteinBatchSubset(PerResidueRegressionSubset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)  
        self.id_column = dataset.id_column
    def get_labels(self):
        #return list(self.data.groupby(self.id_column)[self.label_column].mean())
        return None
class PerResidueClassificationProteinBatchSubset(PerResidueRegressionProteinBatchSubset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.encoder = dataset.encoder
    def get_classes(self):
        return self.encoder.categories_[0]
    def get_labels(self):
        return None

   

    
    
def auto_subset(dataset,idx):
   
    if type(dataset) is PerResidueRegressionDataset:
        return PerResidueRegressionSubset(dataset,idx)
    elif type(dataset) is PerResidueClassificationProteinBatchDataset:
        return PerResidueClassificationProteinBatchSubset(dataset,idx)
    elif type(dataset) is PerResidueRegressionProteinBatchDataset:
        return PerResidueRegressionProteinBatchSubset(dataset,idx)
    else:
        raise TypeError("Unkown dataset instance")
def split_dataset(dataset, split_json:str):
    with open(split_json) as f:
        splits_dict = json.load(f)
    
    if isinstance(dataset,PerResidueRegressionProteinBatchDataset):
        #dont know y i make series #
        column = pd.Series(dataset.unique_protein)
        train_dataset = auto_subset(dataset, np.where(column.isin(set(splits_dict["train_ids"])))[0])
        test_dataset = auto_subset(dataset, np.where(column.isin(set(splits_dict["test_ids"])))[0])
    else:
        df = dataset.metadata_df
    #print(np.where(df[dataset.id_column].isin(splits_dict["train_ids"])))
        train_dataset = auto_subset(dataset, np.where(df[dataset.id_column].isin(set(splits_dict["train_ids"])))[0])
        test_dataset = auto_subset(dataset, np.where(df[dataset.id_column].isin(set(splits_dict["test_ids"])))[0])
    print(f"Length train: {len(train_dataset)}\nLength test: {len(test_dataset)}")
    return train_dataset, test_dataset
        
def _h5_worker_init_fn(worker_id):
    """Each DataLoader worker clears any inherited handle and re-opens
    its own on first __getitem__ call (inside the worker process)."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, 'h5f'):
        info.dataset.h5f = None   
        
def collate_protein_fn(data):
    embeds, labels, ids,idxs = zip(*data)
    embeds = torch.vstack(embeds)
    labels = torch.vstack(labels)
    ids = np.concatenate(ids)
    ids = ids.flatten()
    idxs = np.concatenate(idxs).flatten()
    return  embeds,labels,ids,idxs
    