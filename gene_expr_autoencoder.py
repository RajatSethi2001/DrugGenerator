import h5py
import numpy as np
import os
import pandas as pd
import random
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset
from utils import set_seeds, get_zscore_minmax

class GeneExprEncoder(nn.Module):
    def __init__(self, num_genes, hidden_size=256, num_layers=2, activation_fn=nn.GELU):
        super().__init__()
        self.input_layer = nn.Linear(num_genes, hidden_size)

        self.fc_layers = []
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            self.fc_layers.append(nn.LayerNorm(hidden_size))
            self.fc_layers.append(activation_fn())
        
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gene_expr):
        x = self.input_layer(gene_expr)
        for layer in self.fc_layers:
            x = layer(x)    
        output = self.sigmoid(self.output_layer(x))
        return output
    
class GeneExprDecoder(nn.Module):
    def __init__(self, num_genes, embedding_size=256, hidden_size=256, num_layers=2, activation_fn=nn.GELU):
        super().__init__()
        
        self.input_layer = nn.Linear(embedding_size, hidden_size)
        self.fc_layers = []
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            self.fc_layers.append(nn.LayerNorm(hidden_size))
            self.fc_layers.append(activation_fn())
        
        self.output_layer = nn.Linear(hidden_size, num_genes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedding):
        x = self.input_layer(embedding)
        for layer in self.fc_layers:
            x = layer(x)    
        output = self.sigmoid(self.output_layer(x))
        return output

class GeneExprDataset(Dataset):
    def __init__(self, gctx_file, data_limit=10000):
        self.gctx_fp = h5py.File(gctx_file, "r")
        
        data_idx = random.sample(range(0, len(self.gctx_fp["0/META/COL/id"])), data_limit)
        data_idx.sort()
        data_idx = np.array(data_idx)

        self.gene_symbols = [s.decode("utf-8") for s in self.gctx_fp["/0/META/ROW/pr_gene_symbol"]]
        self.important_genes = pd.read_csv("Data/important_genes.csv", header=None)[0].to_list()
        self.gene_idx = np.array([self.gene_symbols.index(gene) for gene in self.important_genes])

        data_list = []
        for idx in range(data_limit):
            data_list.append(get_zscore_minmax(np.array(self.gctx_fp["0/DATA/0/matrix"][idx, :])[self.gene_idx]))
        
        self.data_arr = np.array(data_list)
        print(self.data_arr)

        self.gctx_fp.close()

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, idx):
        return self.data_arr[idx]

    def get_gene_symbols(self):
        return self.important_genes

def main():
    set_seeds(111)
    train_test_split = 0.2
    save_dir = "Models"
    gctx_file = "Data/annotated_GSE92742_Broad_LINCS_Level5_COMPZ_n473647x12328.gctx"

    batch_size = 128
    input_noise = 0.01

    enc_hidden_size = 512
    enc_layers = 2
    enc_activation = nn.GELU
    enc_lr = 1e-3
    enc_weight_decay = 1e-3

    dec_hidden_size = 512
    dec_layers = 2
    dec_activation = nn.GELU
    dec_lr = 1e-3
    dec_weight_decay = 1e-3
    
    dataset = GeneExprDataset(gctx_file)
    train_size = int(len(dataset) * (1 - train_test_split))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.MSELoss()
    encoder = GeneExprEncoder(len(dataset.get_gene_symbols()),
                              hidden_size=enc_hidden_size,
                              num_layers=enc_layers,
                              activation_fn=enc_activation)
    
    decoder = GeneExprDecoder(len(dataset.get_gene_symbols()),
                              embedding_size=enc_hidden_size,
                              hidden_size=dec_hidden_size,
                              num_layers=dec_layers,
                              activation_fn=dec_activation)

    encoder_optim = optim.AdamW(encoder.parameters(), lr=enc_lr, weight_decay=enc_weight_decay)
    decoder_optim = optim.AdamW(decoder.parameters(), lr=dec_lr, weight_decay=dec_weight_decay)

    if os.path.exists(f"{save_dir}/gene_expr_autoencoder.pth"):
        checkpoint = torch.load(f"{save_dir}/gene_expr_autoencoder.pth")
        encoder.load_state_dict(checkpoint["encoder_model"])
        decoder.load_state_dict(checkpoint["decoder_model"])

    encode_scheduler = ReduceLROnPlateau(encoder_optim, mode='min', factor=0.3, patience=3, threshold=1e-4)
    decode_scheduler = ReduceLROnPlateau(decoder_optim, mode='min', factor=0.3, patience=3, threshold=1e-4)

    for epoch in range(250):
        print(f"Epoch {epoch}")
        encoder.train()
        decoder.train()
        train_loss = 0.0
        batch_idx = 0
        for gene_expr in train_loader:
            noisy_gene_expr = gene_expr + input_noise * torch.randn_like(gene_expr)
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            gene_expr_encoding = encoder(noisy_gene_expr)
            gene_expr_decoding = decoder(gene_expr_encoding)
            
            loss = criterion(gene_expr_decoding, gene_expr)
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            encoder_optim.step()
            decoder_optim.step()

            batch_idx += 1
            # if batch_idx % 5 == 0:
            #     print(f"Training Batch Loss = {loss.item()}")

        print(f"Training Loss = {train_loss / (len(train_dataset) // batch_size)}")

        print("Saving Encoder and Decoder")
        checkpoint = {
            "encoder_model": encoder.state_dict(),
            "decoder_model": decoder.state_dict(),
        }
        torch.save(checkpoint, f"{save_dir}/gene_expr_autoencoder.pth")
        
        encoder.eval()
        decoder.eval()
        test_loss = 0.0
        for gene_expr in test_loader:
            gene_expr_encoding = encoder(gene_expr)
            gene_expr_decoding = decoder(gene_expr_encoding)

            loss = criterion(gene_expr_decoding, gene_expr)
            test_loss += loss.item()

        test_loss = test_loss / (len(test_dataset) // batch_size)
        print(f"Testing Loss = {test_loss}")
        encode_scheduler.step(test_loss)
        decode_scheduler.step(test_loss)

        encoder_lr = encoder_optim.param_groups[0]['lr']
        decoder_lr = decoder_optim.param_groups[0]['lr']
        print(f"Encoder LR = {encoder_lr}, Decoder LR = {decoder_lr}")

if __name__=="__main__":
    main()