import h5py
import mygene
import numpy as np
import os
import pandas as pd
import random
import selfies as sf
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from selfies_autoencoder import SelfiesDecoder, SelfiesEncoder
from torch.utils.data import DataLoader, Dataset, Subset
from train_gctx import GenePertModel
from utils import clean_dose_unit, get_zscores, get_minmax, get_zscore_minmax, set_seeds, smiles_to_embedding

class AllGenePertDataset(Dataset):
    def __init__(self, gctx_file, compound_file, data_limit=20000, max_selfies_len=50):
        self.gctx_fp = h5py.File(gctx_file, "r")
        
        data_idx = random.sample(range(0, len(self.gctx_fp["0/META/COL/id"])), data_limit)
        data_idx.sort()
        data_idx = np.array(data_idx)

        self.ids = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/id"][data_idx]]
        self.pert_types = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/pert_type"][data_idx]]
        self.pert_id = [s.decode('utf-8') for s in self.gctx_fp["0/META/COL/pert_id"][data_idx]]
        self.gene_symbols = np.array([s.decode("utf-8") for s in self.gctx_fp["/0/META/ROW/pr_gene_symbol"]])

        print("Data Collected")

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
        
        enc_hidden_size = 1024
        enc_layers = 2
        enc_activation = nn.GELU

        dec_hidden_size = 1024
        dec_layers = 2
        dec_activation = nn.GELU

        self.encoder = SelfiesEncoder(len(self.selfies_alphabet),
                            max_selfies_len=max_selfies_len,
                            hidden_size=enc_hidden_size,
                            num_layers=enc_layers,
                            activation_fn=enc_activation)
    
        self.decoder = SelfiesDecoder(len(self.selfies_alphabet),
                                max_selfies_len=max_selfies_len,
                                embedding_size=dec_hidden_size,
                                hidden_size=dec_hidden_size,
                                num_layers=dec_layers,
                                activation_fn=dec_activation)

        ae_checkpoint = torch.load("Models/selfies_autoencoder.pth")
        self.encoder.load_state_dict(ae_checkpoint["encoder_model"])
        self.decoder.load_state_dict(ae_checkpoint["decoder_model"])
        self.encoder.eval()
        self.decoder.eval()

        self.gctx_data = []
        data_map_items = list(valid_conditions.items())
        random.shuffle(data_map_items)
        for condition_id, gene_data in data_map_items:
            ctl_exprs = []
            for ctl_idx in gene_data["ctl_idx"]:
                ctl_expr_total = np.array(self.gctx_fp["0/DATA/0/matrix"][ctl_idx, :])
                ctl_expr = get_zscore_minmax(ctl_expr_total)
                ctl_exprs.append(ctl_expr)
            
            if len(ctl_exprs) == 0:
                continue
                
            ctl_exprs = np.array(ctl_exprs)
            ctl_expr_median = np.median(ctl_exprs, axis=0)
            
            if np.isnan(ctl_expr_median).any():
                continue
                
            ctl_expr_median = torch.tensor(ctl_expr_median, dtype=torch.float32)

            for trt_idx in gene_data["trt_idx"]:
                try:
                    trt_expr_total = np.array(self.gctx_fp["0/DATA/0/matrix"][trt_idx, :])
                    trt_expr = torch.tensor(get_zscore_minmax(trt_expr_total), dtype=torch.float32)
                    smiles = self.smiles_lookup[self.pert_id[trt_idx]]
                    smiles_embedding = torch.tensor(smiles_to_embedding(smiles, self.selfies_alphabet, self.encoder), dtype=torch.float32)

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
        return self.gene_symbols

def main():
    set_seeds(1111)

    train_test_split = 0.2
    dataset = AllGenePertDataset("Data/annotated_GSE92742_Broad_LINCS_Level5_COMPZ_n473647x12328.gctx", "Data/compoundinfo_beta.txt")
    mg = mygene.MyGeneInfo()

    train_size = int(len(dataset) * (1 - train_test_split))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.L1Loss()
    training_noise = 0.02

    model = GenePertModel(len(dataset.get_gene_symbols()), 1024, 1600, dropout_prob=0.0, activation_fn=nn.GELU)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3, betas=(0.9, 0.999))

    sample_salmon_file = "Conditions/Healthy/SRR16316900.csv"
    with open(sample_salmon_file, "r") as f:
        salmon_gene_data = f.read().splitlines()
        salmon_gene_data.pop(0)
        ensembl_ids = {gene_expr.split(".")[0] for gene_expr in salmon_gene_data}

    epochs = 100

    for epoch in range(epochs):
        print(f"Training Epoch {epoch}")
        model.train()
        train_loss = 0
        batch = 0
        for ctl_expr, trt_expr, smiles_embedding in train_loader:
            optimizer.zero_grad()
            noisy_ctl_expr = ctl_expr + training_noise * torch.randn_like(ctl_expr)  # small Gaussian noise
            noisy_smiles_embedding = smiles_embedding + training_noise * torch.randn_like(smiles_embedding)
            pred_delta_expr = model(noisy_ctl_expr, noisy_smiles_embedding)
            pred_expr = ctl_expr + pred_delta_expr

            loss = criterion(pred_expr, trt_expr)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch += 1
            
            # print(f"Train Batch {batch}: Loss = {loss.item()}")
        
        train_loss /= batch
        print(f"Training Loss = {train_loss}")

        print(f"Testing Epoch {epoch}")
        model.eval()
        test_loss = 0
        batch = 0
        total_pred_expr = torch.tensor([])
        total_trt_expr = torch.tensor([])
        for ctl_expr, trt_expr, smiles_embedding in test_loader:
            pred_delta_expr = model(ctl_expr, smiles_embedding)
            pred_expr = ctl_expr + pred_delta_expr
            loss = criterion(pred_expr, trt_expr)
            test_loss += loss.item()

            total_pred_expr = torch.cat([total_pred_expr, pred_expr])
            total_trt_expr = torch.cat([total_trt_expr, trt_expr])

            batch += 1
        per_gene_r = np.array([criterion(total_pred_expr[:,g], total_trt_expr[:,g]).item() for g in range(len(dataset.get_gene_symbols()))])
        top_genes_idx = np.argsort(per_gene_r)
        top_genes = dataset.get_gene_symbols()[top_genes_idx]
        top_genes_loss = per_gene_r[top_genes_idx]

        with open("Data/important_genes.csv", "w") as f:
            for idx in range(len(top_genes)):
                symbol = top_genes[idx]
                gene_loss = top_genes_loss[idx]
                print(f"{symbol}, {gene_loss}")
                if gene_loss > 0.13:
                    break
                result = mg.query(symbol, scopes="symbol", fields="ensembl.gene", species="human")

                # Extract Ensembl IDs
                if result and 'hits' in result and result['hits']:
                    hit = result["hits"][0]
                    if 'ensembl' in hit:
                        # Handle cases where ensembl might be a list or single dict
                        ensembl_data = hit['ensembl']
                        if isinstance(ensembl_data, list):
                            # Take the first one if multiple
                            ensembl_id = ensembl_data[0].get('gene')
                        else:
                            ensembl_id = ensembl_data.get('gene')
                    else:
                        ensembl_id = None
                    
                    if ensembl_id is not None and ensembl_id in ensembl_ids:
                        f.write(f"{symbol},{ensembl_id}\n")

            # input()

            # print(f"Test Batch {batch}: Loss = {loss.item()}")

        test_loss /= batch   
        print(f"Testing Loss = {test_loss}, Pearson-R = {pearsonr(pred_expr.detach().numpy()[0], trt_expr.detach().numpy()[0])[0]}")

if __name__=="__main__":
    main()