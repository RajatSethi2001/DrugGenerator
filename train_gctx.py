import h5py
import numpy as np
import os
import pandas as pd
import random
import selfies as sf
import torch
import torch.nn as nn
import torch.optim as optim
from gene_expr_autoencoder import GeneExprEncoder, GeneExprDecoder
from scipy.stats import pearsonr
from selfies_autoencoder import SelfiesEncoder, SelfiesDecoder
from torch.utils.data import Dataset, DataLoader, Subset
from utils import get_zscore_minmax, get_zscores, clean_dose_unit, smiles_to_embedding, set_seeds
"""
/0
/0/DATA
/0/DATA/0
/0/DATA/0/matrix
/0/META
/0/META/COL
/0/META/COL/cell_id
/0/META/COL/distil_id
/0/META/COL/id
/0/META/COL/pert_dose
/0/META/COL/pert_dose_unit
/0/META/COL/pert_id
/0/META/COL/pert_idose
/0/META/COL/pert_iname
/0/META/COL/pert_itime
/0/META/COL/pert_time
/0/META/COL/pert_time_unit
/0/META/COL/pert_type
/0/META/ROW
/0/META/ROW/id
/0/META/ROW/pr_gene_symbol
/0/META/ROW/pr_gene_title
/0/META/ROW/pr_is_bing
/0/META/ROW/pr_is_lm
"""
class PearsonCorrLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(PearsonCorrLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred and target shape: (batch_size, num_genes)
        pred_mean = torch.mean(pred, dim=1, keepdim=True)
        target_mean = torch.mean(target, dim=1, keepdim=True)

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        numerator = torch.sum(pred_centered * target_centered, dim=1)
        denominator = torch.sqrt(torch.sum(pred_centered**2, dim=1) * 
                                 torch.sum(target_centered**2, dim=1) + self.eps)

        r = numerator / denominator
        return 1 - r.mean()

class GenePertDataset(Dataset):
    def __init__(self, gctx_file, compound_file, data_limit=80000, max_selfies_len=50, gene_ae_hidden_size=512, selfies_ae_hidden_size=512):
        self.gctx_fp = h5py.File(gctx_file, "r")
        
        data_idx = random.sample(range(0, len(self.gctx_fp["0/META/COL/id"]) // 2), data_limit)
        data_idx.sort()
        data_idx = np.array(data_idx)

        self.gene_symbols = [s.decode("utf-8") for s in self.gctx_fp["/0/META/ROW/pr_gene_symbol"]]
        self.important_genes = pd.read_csv("Data/important_genes.csv", header=None)[0].to_list()
        self.gene_idx = np.array([self.gene_symbols.index(gene) for gene in self.important_genes])

        data_list = []
        for data_start in range(0, data_limit, 1000):
            print(f"Processing Data Point {data_start}")
            data_end = data_start + min(1000, data_limit - data_start)
            data_idx_window = data_idx[data_start:data_end]
            data_matrix = np.array(self.gctx_fp["0/DATA/0/matrix"][data_idx_window, :])[:, self.gene_idx]
            data_list.append(data_matrix)
        data_matrix = np.concatenate(data_list)
        print(data_matrix.shape)
        print(data_matrix)

        self.ids = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/id"][data_idx]]
        self.pert_types = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/pert_type"][data_idx]]
        self.pert_id = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/pert_id"][data_idx]]

        print("Metadata Collected")

        compound_df = pd.read_csv(compound_file, sep="\t")
        self.smiles_lookup = compound_df.set_index("pert_id")["canonical_smiles"].to_dict()
        self.max_selfies_len = max_selfies_len

        with open("Data/selfies_alphabet.txt", "r") as f:
            self.selfies_alphabet = f.read().splitlines()

        data_map = {}
        for idx in range(data_limit):
            id_parts = self.ids[idx].split(":")
        
            if len(id_parts) < 2:
                continue

            condition = id_parts[0]
            pert_type = self.pert_types[idx]
            
            if condition not in data_map:
                data_map[condition] = {"ctl_idx": [], "trt_idx": []}
            
            if pert_type == "ctl_untrt" or pert_type == "ctl_vehicle":
                data_map[condition]["ctl_idx"].append(idx)

            elif pert_type == "trt_cp":
                pert_id = self.pert_id[idx]
                if pert_id not in self.smiles_lookup:
                    continue
                smiles = self.smiles_lookup[pert_id]
                try:
                    selfies = sf.encoder(smiles)
                    selfies_tokens = list(sf.split_selfies(selfies))
                except:
                    continue

                if len(selfies_tokens) <= self.max_selfies_len:
                    data_map[condition]["trt_idx"].append(idx)
        
        valid_conditions = {}
        for condition, data in data_map.items():
            if len(data["ctl_idx"]) > 0 and len(data["trt_idx"]) > 0:
                valid_conditions[condition] = data
                print(f"Condition {condition}: {len(data['ctl_idx'])} controls, {len(data['trt_idx'])} treatments")
            else:
                print(f"Condition {condition} has no valid mappings")

        self.gene_expr_encoder = GeneExprEncoder(len(self.important_genes),
                                                 hidden_size=gene_ae_hidden_size)

        gene_expr_ae_checkpoint = torch.load("Models/gene_expr_autoencoder.pth")
        self.gene_expr_encoder.load_state_dict(gene_expr_ae_checkpoint["encoder_model"])
        self.gene_expr_encoder.eval()

        self.selfies_encoder = SelfiesEncoder(len(self.selfies_alphabet),
                                              max_selfies_len=max_selfies_len,
                                              hidden_size=selfies_ae_hidden_size)

        selfies_ae_checkpoint = torch.load("Models/selfies_autoencoder.pth")
        self.selfies_encoder.load_state_dict(selfies_ae_checkpoint["encoder_model"])
        self.selfies_encoder.eval()

        self.gctx_data = []
        data_map_items = list(valid_conditions.items())
        random.shuffle(data_map_items)
        for condition_id, gene_data in data_map_items:
            ctl_exprs = []
            for ctl_idx in gene_data["ctl_idx"]:
                ctl_expr_total = np.array(data_matrix[ctl_idx, :])
                ctl_expr = get_zscore_minmax(ctl_expr_total)
                ctl_exprs.append(ctl_expr)
            
            if len(ctl_exprs) == 0:
                continue
                
            ctl_exprs = np.array(ctl_exprs)
            ctl_expr_median = np.median(ctl_exprs, axis=0)
            
            if np.isnan(ctl_expr_median).any():
                continue
                
            ctl_expr_median = torch.tensor(ctl_expr_median, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                ctl_expr_median = self.gene_expr_encoder(ctl_expr_median)[0]

            for trt_idx in gene_data["trt_idx"]:
                try:
                    trt_expr_total = np.array(data_matrix[trt_idx, :])
                    trt_expr = torch.tensor(get_zscore_minmax(trt_expr_total), dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        trt_expr = self.gene_expr_encoder(trt_expr)[0]
                        smiles = self.smiles_lookup[self.pert_id[trt_idx]]
                        smiles_embedding = torch.tensor(smiles_to_embedding(smiles, self.selfies_alphabet, self.selfies_encoder), dtype=torch.float32)

                    self.gctx_data.append((ctl_expr_median, trt_expr, smiles_embedding))
                except Exception as e:
                    print(f"Error processing treatment {trt_idx}: {e}")
                    continue
            
            print(f"Conditions Processed: {len(self.gctx_data)}")
        
        del self.ids
        del self.pert_types
        del self.pert_id
        self.gctx_fp.close()

    def __len__(self):
        return len(self.gctx_data)

    def __getitem__(self, idx):
        return self.gctx_data[idx]

    def get_selfies_alphabet(self):
        return self.selfies_alphabet

    def get_gene_symbols(self):
        return self.important_genes

class GenePertModel(nn.Module):
    def __init__(self, gene_embedding_len, selfies_embedding_len, hidden_size=512, dropout_prob=0.2, activation_fn=nn.GELU):
        super().__init__()
        self.input_size = gene_embedding_len + selfies_embedding_len
        self.input_layer = nn.Linear(self.input_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.LayerNorm(hidden_size)
        
        self.output_layer = nn.Linear(hidden_size, gene_embedding_len)

        self.activation = activation_fn()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, ctl_expr, smiles_embedding):
        x = torch.cat((ctl_expr, smiles_embedding), dim=-1)
        x = self.input_layer(x)
        residual = x
        x = self.dropout(self.activation(self.bn1(self.fc1(x))))
        x = x + residual

        residual = x
        x = self.dropout(self.activation(self.bn2(self.fc2(x))))
        x = x + residual

        x = self.dropout(self.activation(self.bn3(self.fc3(x))))

        output = self.sigmoid(self.output_layer(x))
        return output

def main():
    set_seeds(3333)

    train_test_split = 0.2
    dataset = GenePertDataset("Data/annotated_GSE92742_Broad_LINCS_Level5_COMPZ_n473647x12328.gctx", "Data/compoundinfo_beta.txt")
    model_savefile = "Models/gctx.pth"

    train_size = int(len(dataset) * (1 - train_test_split))
    indices = list(range(len(dataset)))
    # random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.MSELoss()
    training_noise = 0.02

    model = GenePertModel(512, 512, 512, dropout_prob=0.2, activation_fn=nn.GELU)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3, betas=(0.9, 0.999))

    if os.path.exists(model_savefile):
        checkpoint = torch.load(model_savefile, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    epochs = 500

    for epoch in range(epochs):
        print(f"Training Epoch {epoch}")
        model.train()
        train_loss = 0
        batch = 0
        for ctl_expr, trt_expr, smiles_embedding in train_loader:
            optimizer.zero_grad()
            noisy_ctl_expr = ctl_expr + training_noise * torch.randn_like(ctl_expr)  # small Gaussian noise
            noisy_smiles_embedding = smiles_embedding + training_noise * torch.randn_like(smiles_embedding)
            pred_expr = model(noisy_ctl_expr, noisy_smiles_embedding)

            loss = criterion(pred_expr, trt_expr)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch += 1
            
            # print(f"Train Batch {batch}: Loss = {loss.item()}")
        
        train_loss /= batch
        print(f"Training Loss = {train_loss}")
        print(f"Saving Model Data: {model_savefile}")
        torch.save({
            "model_state_dict": model.state_dict()
        }, model_savefile)

        print(f"Testing Epoch {epoch}")
        model.eval()
        test_loss = 0
        batch = 0
        total_pred_expr = torch.tensor([], dtype=torch.float32)
        total_trt_expr = torch.tensor([], dtype=torch.float32)
        for ctl_expr, trt_expr, smiles_embedding in test_loader:
            pred_expr = model(ctl_expr, smiles_embedding)
            loss = criterion(pred_expr, trt_expr)
            test_loss += loss.item()

            batch += 1

            total_pred_expr = torch.cat([total_pred_expr, pred_expr])
            total_trt_expr = torch.cat([total_trt_expr, trt_expr])

            # pred_std = pred_expr.std(dim=1).mean().item()
            # tgt_std  = trt_expr.std(dim=1).mean().item()
            # print("pred_std:", pred_std, "tgt_std:", tgt_std)
            
            # r_list = []
            # for i in range(pred_expr.shape[0]):
            #     r_list.append(pearsonr(pred_expr[i].detach().numpy(), trt_expr[i].detach().numpy())[0])
            # print(np.percentile(r_list,[5,25,50,75,95]), np.mean(r_list))

            # per_gene_r = [pearsonr(pred_expr[:,g].detach().numpy(), trt_expr[:,g].detach().numpy())[0] for g in range(len(dataset.get_gene_symbols()))]
            # print(np.percentile(per_gene_r,[5,25,50,75,95]), np.mean(per_gene_r))

            # input()

            # print(f"Test Batch {batch}: Loss = {loss.item()}")

        test_loss /= batch
        pearson_corr_list = [pearsonr(total_pred_expr.detach().numpy()[idx], total_trt_expr.detach().numpy()[idx])[0] for idx in range(len(total_pred_expr))]
        print(f"Testing Loss = {test_loss}, Pearson-R = {np.median(pearson_corr_list)}")

if __name__=="__main__":
    main()