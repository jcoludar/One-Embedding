

from collections import Counter
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold,KFold, train_test_split
from scipy.stats import spearmanr
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast,GradScaler
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pytorch_based.dataloaders import _h5_worker_init_fn,collate_protein_fn
from pytorch_based.models import SimpleClassifier, TransformerClassifier,DeepSimpleClassifier

from metrics.statistics import averaged_multi_seed, cluster_bootstrap_ci
class SigmoidMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid()
        self.mse = nn.MSELoss()
    def forward(self,pred,labels):
        pred = self.sig(pred)
        return self.mse(pred,labels)
class MultiClassTrainer:
    def __init__(self, model_config, learning_rate=.0005, weight_decay=.0005,
                device = None,num_epochs = 50, patience = 10,mode = "classification"):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"INFO: Using device: {self.device}")
        #defualt Loss function for nominal classification
        
        
        self.lr = learning_rate
        self.wd = weight_decay
        self.cfg = model_config
        #self.model = SimpleClassifier(**model_config).to(self.device)
        self.num_epochs = num_epochs
        self.patience = patience 
        self.T_0 = 8
        self.dropout = self.cfg["dropout_rate"]
        self.output_dim = self.cfg["num_classes"]
        self.mode = mode
        self.criterion = None
        self.set_criterion()
        self.batch_size = 256
    def set_criterion(self):
        if self.mode == "classification":
            if not isinstance(self.criterion, nn.CrossEntropyLoss): 
                print("switched to CrossEntropyLoss" )
                self.model_type = SimpleClassifier
                self.criterion = nn.CrossEntropyLoss(weight=None)
        elif self.mode == "regression":
            if not isinstance(self.criterion, nn.L1Loss): 
                print("switched to L1Loss" )
                #print("switched to MSELoss" )
                self.model_type = SimpleClassifier
                #self.criterion = SigmoidMSE()
                #self.criterion = nn.MSELoss()
                self.criterion = nn.L1Loss()
    #want to keep trainer for simplicity but unsure what needs to acctually be reset so just do all 
    def init(self):
        self.cfg["dropout_rate"] = self.dropout
        
        self.model = self.model_type(**self.cfg).to(self.device)
        if self.device.type == 'cuda':
            self.model = torch.compile(self.model)
            self.model = self.model.to(self.device.type)
        
        #seperate weight decay, improvement over Adam
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        #Makes learning rate initial val less important
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0 = self.T_0,T_mult = 1, eta_min = self.lr/100)
        #Still not 100% sure what this does
        self.use_amp = self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
    #model train step
    def train_and_validate(self, train_loader, val_loader, checkpoint_path, 
                      classes_list):
        #stop by val accuracy
        best_val_acc = -10000.0 
        epochs_without_improvement = 0
        last_epoch = 0

        for epoch in range(self.num_epochs):
            last_epoch = epoch
            #train
            self.model.train()
            train_loss, train_acc = self._run_epoch(train_loader)
            
            # Validation phase: get loss and full report
            self.model.eval()
            with torch.no_grad():
                val_report, _, _, _,_ = self.evaluate_on_loader(val_loader, classes_list)
            
            # extract the F1 score
            
            
            
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            #current_val_f1 = val_report['macro avg']['f1-score'] 
            if self.mode == "classification":
                current_val_acc = val_report['accuracy'] * 100  #for logging
            elif self.mode == "regression":
                current_val_acc = val_report['spearman_rho']* 100
            print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Acc: {current_val_acc:.2f}% | LR: {current_lr:.2e}")
            
            

            # checkpointing based on best Acc score
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'val_acc': current_val_acc,
                    'label_encoder_classes': classes_list
                }, checkpoint_path)
                #print(f"  -> Saved best model (val_acc: {current_val_acc:.4f})")
                epochs_without_improvement = 0
            #early stop
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    #print(f'INFO: Early stopping at epoch {epoch+1}. Best val_Acc: {best_val_acc:.4f}')
                    break
              
        
        if os.path.exists(checkpoint_path):
            checkpoint= self.load_checkpoint(checkpoint_path)
            #print(f"INFO: Loaded best model from checkpoint (val_acc: {checkpoint.get('val_acc', 0):.4f})")

        val_metrics, all_preds, all_ids, all_labels,_ = self.evaluate_on_loader(val_loader, classes_list)
        #not sure whats efficient, dataframe seems easy to do but seems really ineffiecent, most likely would be better to only use np. only do it once at end so effieceny is not super important.
        if self.mode == "classification":
            correct = all_preds ==all_labels
            df = pd.DataFrame(data = {"ids":all_ids,"acc":correct})
            per_prot_series = df.groupby("ids").mean()
            per_prot_acc = per_prot_series.to_dict()
            val_metrics["per_protein"] = per_prot_acc
        return val_metrics, last_epoch + 1
    #pretty default _run_epoch function, but only handles training, validation done on evaluate_on_loader
    def _run_epoch(self, dataloader):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        for embeddings, labels, _, _ in dataloader:
            #load set num proteins then put batches of 64 on gpu
            
            permutation = np.random.RandomState(seed=42).permutation(len(labels))
            remaining = len(labels) % self.batch_size
            length_truncated = len(labels)-remaining
            for i in range(0,length_truncated,self.batch_size):
                if i == length_truncated - self.batch_size:
                    
                    current_batch = permutation[i:]
                else:  
                    current_batch = permutation[i:i+self.batch_size]
                #load to device
                gpu_embeddings = embeddings[current_batch].to(self.device)
                gpu_labels = labels[current_batch].float().to(self.device)
                
                
                
                
                #compute predictions
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(gpu_embeddings)
                    loss = self.criterion(outputs, gpu_labels)
                    
                #scale grad
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                #clip grad if want
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                #format predictions
                if self.mode == "classification":
                    predictions = torch.argmax(outputs, dim=1)
                    actual = torch.argmax(gpu_labels, dim=1)
                    total_correct += (predictions == actual).sum().item()
                    
                elif self.mode == "regression":
                    all_labels.extend(gpu_labels.cpu().numpy())
                    all_preds.extend(outputs.cpu().detach().numpy())
                total_samples += gpu_labels.size(0)
                total_loss += loss.item() * gpu_labels.size(0)
            
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy=0
        if self.mode == "classification":
            accuracy = (100 * total_correct / total_samples) if total_samples > 0 else 0
        elif self.mode == "regression":
            accuracy =  100 * spearmanr(all_labels,all_preds)[0]
        return avg_loss, float(accuracy)
    def load_checkpoint(self, checkpoint_path):
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only = False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    #
    
    def evaluate_on_loader(self, data_loader, classes_list):
        self.model.eval()
        all_labels = []
        all_preds = []
        all_ids = []
        all_idxs = []
        #all_preds_raw = torch.empty((0,len(label_encoder.categories_[0])),dtype=torch.float32)

        with torch.no_grad():
            for embeddings, labels, ids_batch,idxs_batch in data_loader:
                #load set num proteins then put batches of 64 on gpu
                permutation = np.random.RandomState(seed=42).permutation(len(ids_batch))
                remaining = len(ids_batch) % self.batch_size
                length_truncated = len(ids_batch)-remaining
                for i in range(0,length_truncated,self.batch_size):
                    if i == length_truncated - self.batch_size:
                        
                        current_batch = permutation[i:]
                    else:  
                        current_batch = permutation[i:i+self.batch_size]
                    #load to device
                    gpu_embeddings = embeddings[current_batch].to(self.device)
                    gpu_labels = labels[current_batch].float().to(self.device)
                    
                    
                    #compute predictions               
                    with autocast(device_type=self.device.type, enabled=self.use_amp):
                        outputs = self.model(gpu_embeddings)
                    #format predictions
                    if self.mode == "classification":
                        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                        actual = torch.argmax(gpu_labels, dim=1).cpu().numpy()
                    elif self.mode == "regression":
                        predictions = outputs.cpu().detach().numpy()
                        actual = gpu_labels.cpu().numpy()
                    #save predictions
                    
                    all_labels.extend(actual)
                    all_preds.extend(predictions)
                    all_ids.extend(list(ids_batch[current_batch]))
                    all_idxs.extend(list(idxs_batch[current_batch]))
                
        #make stats
        
        if self.mode == "classification":
            
            report = classification_report(
                all_labels, all_preds, 
                labels=list(range(len(classes_list))),
                target_names=classes_list, 
                output_dict=True, 
                zero_division=0
            )
        elif self.mode == "regression":
            rho, p_value = spearmanr(all_labels,all_preds)
            report = {"spearman_rho":float(rho),"p_value":float(p_value)}
        
        return report,  all_preds, all_ids, all_labels, all_idxs

def run_kfold(train_dataset,trainer: MultiClassTrainer, k_folds=5,random_seed = 42,batch_size = 64,checkpoint_folder = "checkpoints"):
    
    labels = train_dataset.get_labels()
    
    if labels is None: 
        skf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
        labels = np.zeros(len(train_dataset)) 
    
    else:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    class_list = train_dataset.get_classes()
    #val_accuracy_metrics = []
    val_fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)),labels )):
        train_loader = DataLoader(
            Subset(train_dataset, train_idx),
            batch_size=batch_size,
            num_workers=8,
            worker_init_fn=_h5_worker_init_fn, 
            persistent_workers=True,
            shuffle=True,
            collate_fn = collate_protein_fn
        )
        val_loader = DataLoader(
            Subset(train_dataset, val_idx),
            batch_size=batch_size,
            num_workers=8,
            worker_init_fn=_h5_worker_init_fn, 
            persistent_workers=True,
            shuffle=False,
            collate_fn = collate_protein_fn
        )
        trainer.init()
        checkpoint = os.path.join(checkpoint_folder, f"temp_trial_{fold}.pt")
        val_metrics, epochs_ran = trainer.train_and_validate(
            train_loader, val_loader,
            checkpoint, class_list, 
        )
        if os.path.exists(checkpoint):
            os.remove(checkpoint)
        #val_accuracy_metrics.append(val_metrics["accuracy"])
        if trainer.mode == "classification":
            val_fold_metrics.append(val_metrics["macro avg"]["f1-score"])
        elif trainer.mode == "regression":
            val_fold_metrics.append(val_metrics["spearman_rho"]) 

    val_fold_metrics = np.array(val_fold_metrics)
    return val_fold_metrics.mean()
def parameter_grid_search(train_dataset,trainer: MultiClassTrainer,batch_size:int = 64,checkpoint_folder: str = "checkpoints",parameter_search_dict:dict = {"lr":[0.0005,0.000005],"wd":[0.0005,0.000005]}):
    #method to grid search the dict doesnt seem right
    #setup for grid search 
    
    param_lengths = {}
    param_list = []
    total_length = 1
    for key, item in parameter_search_dict.items():
        param_list.append(key)
        length = len(item)
        param_lengths[key] = length
        total_length *= length
    best_metric = -1000
    best_params = {}
    for search in range(total_length):
        #get current params
        step = 1
        cur_param = {}
        for param in param_list:
            cur_length = param_lengths[param]
            value = parameter_search_dict[param][(search//step)%cur_length]
            cur_param[param] = value
            #idk why i insist on using one trainer object :(
            setattr(trainer,param,value)
            step = step *cur_length
        print(f"current params: {cur_param}")
        average_metric = run_kfold(train_dataset,trainer,k_folds=3,checkpoint_folder=checkpoint_folder,batch_size=batch_size)
        if average_metric > best_metric:
            best_metric = average_metric
            best_params = cur_param
    print(f"best params: {best_params}")
    return best_params
def train_val_test_loader(train_labels, train_dataset, test_dataset,batch_size = 64):
    strat = train_labels
    if train_labels is None:
        train_labels = np.zeros(len(train_dataset))
        
    train_idx, val_idx, _, _ = train_test_split(
        range(len(train_labels)),
        train_labels,
        train_size = 0.9,
        stratify = strat,
        random_state=42
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size,
        num_workers=8,
        worker_init_fn=_h5_worker_init_fn, 
        persistent_workers=True,
        shuffle=True,
        collate_fn = collate_protein_fn, 
    )
    val_loader = DataLoader(
        Subset(train_dataset, val_idx),
        batch_size=batch_size,
        num_workers=8,
        worker_init_fn=_h5_worker_init_fn, 
        persistent_workers=True,
        shuffle=True,
        collate_fn = collate_protein_fn, 
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=8,
        worker_init_fn=_h5_worker_init_fn, 
        persistent_workers=True,
        shuffle=True,
        collate_fn = collate_protein_fn, 
    )
    return train_loader,val_loader,test_loader
def run_classification_task(train_dataset,test_dataset,trainer: MultiClassTrainer,checkpoint_folder="checkpoint",parameter_search_dict={"lr":[0.0005,0.000005],"wd":[0.0005,0.000005],"dropout": [0.1,0.3]},seeds = [41,42,43],batch_size=16, n_bootstrap=10000):
    #setup trainer
    class_labels = train_dataset.get_classes()
    trainer.cfg["num_classes"] = len(class_labels)
    trainer.cfg["embed_size"] = train_dataset.get_embedding_dim()
    trainer.mode = "classification"
    trainer.set_criterion()
    #ensure correct simple params
    trainer.num_epochs = 2
    trainer.patience = 1
    trainer.T_0 = 3
    
    params = parameter_grid_search(train_dataset,trainer,checkpoint_folder=checkpoint_folder,parameter_search_dict=parameter_search_dict,batch_size=batch_size)
    
    for param in params.keys():
        setattr(trainer,param,params[param])
    #train_labels =train_dataset.get_labels() 
    train_loader,val_loader,test_loader = train_val_test_loader(None, train_dataset, test_dataset,batch_size=batch_size)
    #trainer for longer 
    trainer.num_epochs = 4
    trainer.patience = 2
    trainer.T_0 = 4
    seed_acc = []
    seed_per_prot_acc = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = True
        trainer.init()
        checkpoint = os.path.join(checkpoint_folder, f"temp_trial_{seed}.pt")
        trainer.train_and_validate(
            train_loader, val_loader,
            checkpoint, class_labels, 
        )
        test_metrics, all_preds, all_ids, all_labels,_ = trainer.evaluate_on_loader(test_loader, class_labels)
        #not sure whats efficient, dataframe seems easy to do but seems really ineffiecent, most likely would be better to only use np. only do it once at end so effieceny is not super important.
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        correct = all_preds == all_labels
        
        df = pd.DataFrame(data = {"ids":all_ids,"acc":list(correct.flatten())})
        
        per_prot_series = df.groupby("ids").mean()
        
        per_prot_acc = per_prot_series.to_dict()
        seed_per_prot_acc.append(per_prot_acc["acc"])
        seed_acc.append(test_metrics["accuracy"]) 
        
        if os.path.exists(checkpoint):
            os.remove(checkpoint)
    q3 = averaged_multi_seed(seed_per_prot_acc,n_bootstrap=n_bootstrap)
    median_idx = int(np.argsort(seed_acc)[len(seed_acc) // 2])
    return q3, seed_per_prot_acc[median_idx]

def pooled_spearman(cluster_data: list[dict]) -> float:
    """Compute pooled residue-level Spearman rho across all clusters (proteins).

    This is the standard metric used by SETH/ODiNPred/ADOPT/UdonPred.

    Args:
        cluster_data: list of {"y_true": ndarray, "y_pred": ndarray} dicts.

    Returns:
        Pooled Spearman rho, or 0.0 if computation fails.
    """
    
    all_true = np.concatenate([d["y_true"] for d in cluster_data])
    all_pred = np.concatenate([d["y_pred"] for d in cluster_data])
    rho, _ = spearmanr(all_true, all_pred)
    return float(rho) if not np.isnan(rho) else 0.0

def run_regression_task(train_dataset,test_dataset,trainer: MultiClassTrainer,checkpoint_folder="checkpoint",parameter_search_dict={"lr":[0.0005,0.000005],"wd":[0.0005,0.000005],"dropout": [0.1,0.3]},seeds = [41,42,43],batch_size=16, n_bootstrap=10000):
    #setup trainer
    #class_labels = train_dataset.get_classes()
    trainer.cfg["num_classes"] = 1
    trainer.cfg["embed_size"] = train_dataset.get_embedding_dim()
    trainer.mode = "regression"
    trainer.set_criterion()
    #ensure correct simple params
    trainer.num_epochs = 3
    trainer.patience = 1
    trainer.T_0 = 3
    
    params = parameter_grid_search(train_dataset,trainer,checkpoint_folder=checkpoint_folder,parameter_search_dict=parameter_search_dict,batch_size=batch_size)
    
    for param in params.keys():
        setattr(trainer,param,params[param])
    train_labels =train_dataset.get_labels() 
    train_loader,val_loader,test_loader = train_val_test_loader(train_labels, train_dataset, test_dataset,batch_size=batch_size)
    #trainer for longer 
    trainer.num_epochs = 5
    trainer.patience = 3
    trainer.T_0 = 4
    
    seed_per_prot_rho = []
    seed_cluster_dicts = []
    all_labels = None
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = True
        trainer.init()
        checkpoint = os.path.join(checkpoint_folder, f"temp_trial_{seed}.pt")
        
        trainer.train_and_validate(
            train_loader, val_loader,
            checkpoint, None, 
        )
        test_metrics, all_preds, all_ids, all_labels,all_idxs = trainer.evaluate_on_loader(test_loader, None)
        #not sure whats efficient, dataframe seems easy to do but seems really ineffiecent, most likely would be better to only use np. only do it once at end so effieceny is not super important.
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        #no clue y i used dataframe here, dont think groupby aproach with new init df is that good
        df = pd.DataFrame(data = {"ids":all_ids,"idxs":all_idxs,"all_preds": list(all_preds)})
        cluster_dict = df.sort_values("idxs").groupby("ids")["all_preds"].apply(list).to_dict()
        seed_cluster_dicts.append(cluster_dict)
        seed_per_prot_rho.append(test_metrics["spearman_rho"])
       
        
        if os.path.exists(checkpoint):
            os.remove(checkpoint)
    print(seed_per_prot_rho)
    df = pd.DataFrame(data = {"ids":all_ids,"idxs":all_idxs,"all_labels": list(all_labels)})
    
    cluster_labels = df.sort_values("idxs").groupby("ids")["all_labels"].apply(list).to_dict()
    #this part right here is so slow wow
    #here
    
    median_idx = int(np.argsort(seed_per_prot_rho)[len(seed_per_prot_rho) // 2])
    median_cluster = {}  # {pid: {"y_true": arr, "y_pred": arr}}
    for pid in cluster_labels.keys():
        y_true = np.array(cluster_labels[pid]).flatten()  # same across seeds
        y_pred_median = np.array(seed_cluster_dicts[median_idx][pid]).flatten()
        median_cluster[pid] = {"y_true": y_true.tolist(), "y_pred": y_pred_median.tolist()}
       
    averaged_clusters = {}  # {pid: {"y_true": arr, "y_pred": arr}}
    for pid in seed_cluster_dicts[0].keys():
        y_true = np.array(cluster_labels[pid]).flatten()  # same across seeds
        y_pred_stacked = np.stack(
            [seed_cluster_dicts[s][pid] for s in range(len(seeds))],
            axis=0,
        )
        y_pred_avg = np.mean(y_pred_stacked, axis=0).flatten()
        averaged_clusters[pid] = {"y_true": y_true.tolist(), "y_pred": y_pred_avg.tolist()}    
    pooled_rho_result = cluster_bootstrap_ci(
        median_cluster, pooled_spearman,
        n_bootstrap=n_bootstrap, seed=seeds[0],
    )
    #to here

    return pooled_rho_result, median_cluster


        


    




    