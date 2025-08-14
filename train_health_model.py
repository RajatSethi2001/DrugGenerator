import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from gene_expr_autoencoder import GeneExprEncoder
from torch.utils.data import Dataset, DataLoader, Subset
from utils import set_seeds, get_minmax, get_zscores, get_zscore_minmax

class HealthDataset(Dataset):
    def __init__(self, df: pd.DataFrame, genes, encoder):
        self.gene_data = torch.tensor(df[genes].values, dtype=torch.float32)
        with torch.no_grad():
            self.gene_encodings = encoder(self.gene_data)
        self.health_data = torch.tensor(df["Healthy"].values, dtype=torch.float32).unsqueeze(1)
        print("Gene Encodings Shape = ", self.gene_encodings.shape)
        print("Health Data Shape = ", self.health_data.shape)
        self.genes = genes

    def __len__(self):
        return len(self.gene_encodings)

    def __getitem__(self, idx):
        return self.gene_encodings[idx, :], self.health_data[idx, :]

class HealthModel(nn.Module):
    def __init__(self, gene_embedding_len, hidden_neurons=32):
        super().__init__()
        self.input_layer = nn.Linear(gene_embedding_len, hidden_neurons)
        self.fc1 = nn.Linear(hidden_neurons, hidden_neurons)
        self.bn1 = nn.LayerNorm(hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.bn2 = nn.LayerNorm(hidden_neurons)
        self.output_layer = nn.Linear(hidden_neurons, 1)
    
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.0)
    
    def forward(self, x):
        x = self.dropout(self.activation(self.input_layer(x)))
        x = self.dropout(self.activation(self.bn1(self.fc1(x))))
        x = self.dropout(self.activation(self.bn2(self.fc2(x))))
        x = self.sigmoid(self.output_layer(x))
        return x

class HealthModelLinear(nn.Module):
    def __init__(self, num_genes):
        super().__init__()
        self.dropout = nn.Dropout(0.4)
        self.network = nn.Linear(num_genes, 1, bias=False)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.activation(self.network(x))
        return x

    def get_network_weights(self):
        return self.network.weight.data

def main():
    set_seeds(111)
    healthy_dir = "Conditions/Healthy"
    unhealthy_dir = "Conditions/Unhealthy"
    savefile = "Models/health_model.pth"
    train_test_split = 0.2
    batch_size = 16
    genes = pd.read_csv("Data/important_genes.csv", header=None)[1].to_list()

    encoder_file = "Models/gene_expr_autoencoder.pth"
    encoder_data = torch.load(encoder_file, weights_only=False)
    encoder_state_dict = encoder_data["encoder_model"]
    encoder = GeneExprEncoder(256, 256, 0.0)
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    total_df = pd.DataFrame()
    for filename in os.listdir(healthy_dir):
        path = f"{healthy_dir}/{filename}"
        print(f"Processing File: {path}")

        df = pd.read_csv(path, index_col=0)
        df.index = [index.split(".")[0] for index in df.index]
        df = np.log2(df + 1)
        df = df.apply(get_zscores, axis=0)
        df = df.apply(get_zscore_minmax, axis=0)
        df = df.loc[genes, :]
        df = df.transpose()
        df["Healthy"] = 1
        total_df = pd.concat([total_df, df])
    
    for filename in os.listdir(unhealthy_dir):
        path = f"{unhealthy_dir}/{filename}"
        print(f"Processing File: {path}")

        df = pd.read_csv(path, index_col=0)
        df.index = [index.split(".")[0] for index in df.index]
        df = np.log2(df + 1)
        df = df.apply(get_zscores, axis=0)
        df = df.apply(get_zscore_minmax, axis=0)
        df = df.loc[genes, :]
        df = df.transpose()
        df["Healthy"] = 0
        total_df = pd.concat([total_df, df])

    print(total_df)

    dataset = HealthDataset(total_df, genes, encoder)
    train_size = int(len(dataset) * (1 - train_test_split))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = HealthModel(len(genes), hidden_neurons=20)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if os.path.exists(savefile):
        checkpoint = torch.load(savefile, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.L1Loss()
    for epoch in range(501):
        print(f"Epoch: {epoch}")
        train_metrics = {"tp": np.float32(0), "tn": np.float32(0), "fp": np.float32(0), "fn": np.float32(0)}
        train_loss = 0
        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            for batch_idx in range(len(outputs)):
                output_batch = outputs[batch_idx]
                label_batch = labels[batch_idx]
                
                output_choice = round(output_batch[0].item())
                label_choice = round(label_batch[0].item())

                if output_choice == 1 and label_choice == 1:
                    train_metrics["tp"] += 1
                
                elif output_choice == 0 and label_choice == 0:
                    train_metrics["tn"] += 1
                
                elif output_choice == 1 and label_choice == 0:
                    train_metrics["fp"] += 1
                
                elif output_choice == 0 and label_choice == 1:
                    train_metrics["fn"] += 1
        
        if epoch % 20 == 0:
            print(train_metrics)
            tp = train_metrics["tp"]
            tn = train_metrics["tn"]
            fp = train_metrics["fp"]
            fn = train_metrics["fn"]
            with np.errstate(invalid='ignore', divide='ignore'):
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f_score = 2 * precision * recall / (precision + recall)
                print(f"Train Metrics: Precision = {round(precision, 3)}, Recall = {round(recall, 3)}, F-Score = {round(f_score, 3)}")
        
            train_loss /= len(train_dataset)
            print(f"Train Loss = {train_loss}")    
            print("Saving Model")
            print()
            torch.save({
                "model_state_dict": model.state_dict(),
                "genes": genes,
            }, savefile)

        model.eval()
        test_metrics = {"tp": np.float32(0), "tn": np.float32(0), "fp": np.float32(0), "fn": np.float32(0)}
        test_loss = 0
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            for batch_idx in range(len(outputs)):
                output_batch = outputs[batch_idx]
                label_batch = labels[batch_idx] 
                
                output_choice = round(output_batch[0].item())
                label_choice = round(label_batch[0].item())

                if output_choice == 1 and label_choice == 1:
                    test_metrics["tp"] += 1
                
                elif output_choice == 0 and label_choice == 0:
                    test_metrics["tn"] += 1
                
                elif output_choice == 1 and label_choice == 0:
                    test_metrics["fp"] += 1
                
                elif output_choice == 0 and label_choice == 1:
                    test_metrics["fn"] += 1
        
        if epoch % 20 == 0:
            tp = test_metrics["tp"]
            tn = test_metrics["tn"]
            fp = test_metrics["fp"]
            fn = test_metrics["fn"]
            with np.errstate(invalid='ignore', divide='ignore'):
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f_score = 2 * precision * recall / (precision + recall)
                print(f"Test Metrics: Precision = {round(precision, 3)}, Recall = {round(recall, 3)}, F-Score = {round(f_score, 3)}")
            
            test_loss /= len(test_dataset)
            print(f"Test Loss = {test_loss}")
            print()

if __name__=="__main__":
    main()