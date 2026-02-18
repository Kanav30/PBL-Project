# Generated from: Federated_Spatio_Temporal_Graph_Network.ipynb
# Converted at: 2026-02-18T20:29:09.372Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# <a href="https://colab.research.google.com/github/Kanav30/Federated-Spatio-Temporal-Graph-Network/blob/main/Federated_Spatio_Temporal_Graph_Network.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import os

#Loading the file
file_path = "/content/drive/MyDrive/metr-la.h5"

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure you have uploaded the 'metr-la (1).h5' file to your Colab environment's /content/ directory.")
else:
    with h5py.File(file_path, "r") as f:
        print(list(f.keys()))

#read traffic speed data
with h5py.File(file_path, "r") as f:
    df_group = f["df"]
    speed_data = df_group["block0_values"][:]

print(speed_data.shape)

#One row
print(speed_data[0])

#Missing value

print("NaNs before:", np.isnan(speed_data).sum())
mean_value = np.nanmean(speed_data)
speed_data = np.nan_to_num(speed_data, nan=mean_value)

print("NaNs after:", np.isnan(speed_data).sum())


#Normalizing
mean = speed_data.mean()
std = speed_data.std()

speed_data = (speed_data - mean) / std

#Samples of the data

INPUT_WINDOW = 12
OUTPUT_WINDOW = 12

def create_samples(data, input_window, output_window):
    X, Y = [], []
    T = data.shape[0]

    for t in range(input_window, T - output_window):
        past = data[t - input_window:t]      #past 12 steps
        future = data[t:t + output_window]   #next 12 steps

        X.append(past)
        Y.append(future)

    return np.array(X), np.array(Y)


X, Y = create_samples(speed_data, INPUT_WINDOW, OUTPUT_WINDOW)

print("X shape:", X.shape)
print("Y shape:", Y.shape)


X = X[..., np.newaxis]
Y = Y[..., np.newaxis]

print(X.shape)
print(Y.shape)


X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)


class METRLADataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = METRLADataset(X, Y)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for xb, yb in dataloader:
    print("Input batch:", xb.shape)
    print("Target batch:", yb.shape)
    break

import pickle

with open("/content/drive/MyDrive/adj_mx.pkl", "rb") as f:
    sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
print(adj_mx.shape)

#Convert to PyTorch Sensor
import torch

adj = torch.tensor(adj_mx, dtype=torch.float32)


def normalize_adj(adj):
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    degree = torch.sum(adj, dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

adj_norm = normalize_adj(adj)

import torch
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj):
        # x: [batch, time, nodes, features]
        x = torch.einsum("btnf,nm->btmf", x, adj)
        x = self.linear(x)
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0)
        )

    def forward(self, x):
        # x: [batch, time, nodes, features]
        x = x.permute(0, 3, 1, 2)  # [batch, features, time, nodes]
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # back to [batch, time, nodes, features]
        return x


class STGCN(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()

        self.gc1 = GraphConv(1, 32)
        self.tc1 = TemporalConv(32, 32)

        self.gc2 = GraphConv(32, 32)
        self.tc2 = TemporalConv(32, 32)

        self.output_layer = nn.Linear(32, 1)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = torch.relu(x)

        x = self.tc1(x)
        x = torch.relu(x)

        x = self.gc2(x, adj)
        x = torch.relu(x)

        x = self.tc2(x)
        x = torch.relu(x)

        x = self.output_layer(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STGCN(num_nodes=207).to(device)
adj_norm = adj_norm.to(device)

x_batch, y_batch = next(iter(dataloader))

x_batch = x_batch.to(device)

with torch.no_grad():
    output = model(x_batch, adj_norm)

print("Input shape :", x_batch.shape)
print("Output shape:", output.shape)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
model = model.to(device)
adj_norm = adj_norm.to(device)

#The training loop

def train_model(model, dataloader, adj, criterion, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for x_batch, y_batch in dataloader:
            # Move data to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(x_batch, adj)

            # Compute loss
            loss = criterion(output, y_batch)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

train_model(
    model,
    dataloader,
    adj_norm,
    criterion,
    optimizer,
    epochs=15
)

model.eval()

with torch.no_grad():
    x_batch, y_batch = next(iter(dataloader))
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    predictions = model(x_batch, adj_norm)

    mae = torch.mean(torch.abs(predictions - y_batch))
    print("MAE:", mae.item())

import matplotlib.pyplot as plt

sensor_id = 50  # pick any sensor

true_values = y_batch[0, :, sensor_id, 0].cpu().numpy()
pred_values = predictions[0, :, sensor_id, 0].cpu().numpy()

plt.plot(true_values, label="True")
plt.plot(pred_values, label="Predicted")
plt.legend()
plt.title("Prediction vs True (One Sensor)")
plt.show()

import numpy as np
print("Correlation:", np.corrcoef(true_values, pred_values)[0,1])
print("True:", true_values)
print("Pred:", pred_values)

#Four regions for now
import numpy as np

NUM_CLIENTS = 4

#Splitting sensors into clusters
num_sensors = speed_data.shape[1]

sensor_indices = np.arange(num_sensors)

clusters = np.array_split(sensor_indices, NUM_CLIENTS)

for i, cluster in enumerate(clusters):
    print(f"Client {i} has {len(cluster)} sensors")

#Local speed data per client
client_speed_data = []

for cluster in clusters:
    client_speed_data.append(speed_data[:, cluster])
    print(f"Client speed data shape: {client_speed_data[-1].shape}")

#Local adjacency subgraphs
client_adj_matrices = []

for cluster in clusters:
    sub_adj = adj_norm[np.ix_(cluster, cluster)]
    client_adj_matrices.append(sub_adj)
    print(client_adj_matrices[0].shape)

#Sliding windows per client
client_dataloaders = []

for i in range(NUM_CLIENTS):

    X_client, Y_client = create_samples(
        client_speed_data[i],
        INPUT_WINDOW,
        OUTPUT_WINDOW
    )

    X_client = X_client[..., np.newaxis]
    Y_client = Y_client[..., np.newaxis]

    X_client = torch.tensor(X_client, dtype=torch.float32)
    Y_client = torch.tensor(Y_client, dtype=torch.float32)

    dataset = METRLADataset(X_client, Y_client)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    client_dataloaders.append(dataloader)

for i in range(NUM_CLIENTS):
  xb, yb = next(iter(client_dataloaders[i]))
  print(f"Client {i} batch shape:", xb.shape)