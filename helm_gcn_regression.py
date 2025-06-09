import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
import ast

# === Step 1: Load data ===
df = pd.read_csv(
    "/Users/kevinhou/Documents/CY Lab/relevant papers/CycPeptMPDB_Peptide_All.csv")  # Update with your file path
df = df.dropna(subset=['HELM', 'Same_Peptides_Permeability'])


# Step 3: Define a function to parse and compute the average permeability
def compute_avg_permeability(value):
    try:
        # Parse the string into a nested list
        parsed = ast.literal_eval(value)
        # Flatten the nested list and convert all elements to float
        values = [float(x) for sublist in parsed for x in sublist]
        if values:
            return sum(values) / len(values)
    except Exception:
        return None
    return None

# Step 4: Apply the function to compute average permeability for each row
df['Avg_Permeability'] = df['Same_Peptides_Permeability'].apply(compute_avg_permeability)

print(df.head())


# === Step 2: Build residue vocabulary ===
def extract_residues(helm_str):
    return helm_str.split("{")[1].split("}")[0].split(".")

residue_set = set()
for helm in df["HELM"]:
    residue_set.update(extract_residues(helm))

residue_list = sorted(residue_set)
residue2id = {res: i for i, res in enumerate(residue_list)}

# === Step 3: HELM to PyG Data ===
def helm_to_pyg_data(helm_str, target, residue2id):
    residues = extract_residues(helm_str)
    residue_ids = [residue2id[r] for r in residues]
    edge_list = []

    for i in range(len(residue_ids) - 1):
        edge_list.append((i, i + 1))
        edge_list.append((i + 1, i))

    if "$" in helm_str:
        parts = helm_str.split("$")
        if len(parts) > 1 and ":" in parts[1]:
            try:
                cross = parts[1].split(",")[2]
                p1 = int(cross.split("-")[0].split(":")[0]) - 1
                p2 = int(cross.split("-")[1].split(":")[0]) - 1
                edge_list.append((p1, p2))
                edge_list.append((p2, p1))
            except:
                pass

    x = torch.tensor(residue_ids, dtype=torch.long).unsqueeze(1)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    y = torch.tensor([target], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)

# === Step 4: Dataset class ===
class PeptideHelmDataset(InMemoryDataset):
    def __init__(self, dataframe, residue2id):
        self.df = dataframe
        data_list = [helm_to_pyg_data(row["HELM"], row["Avg_Permeability"], residue2id)
                     for _, row in dataframe.iterrows()]
        super().__init__(".")
        self.data, self.slices = self.collate(data_list)

# === Step 5: Define GCN Model ===
class GCNResidueEmbedding(nn.Module):
    def __init__(self, num_residues, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_residues, emb_dim)
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.embedding(x.squeeze())
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze()

# === Step 6: Split data ===
df_trainval, df_test = train_test_split(df, test_size=0.1, random_state=42)
df_train, df_val = train_test_split(df_trainval, test_size=0.2, random_state=42)

train_dataset = PeptideHelmDataset(df_train, residue2id)
val_dataset = PeptideHelmDataset(df_val, residue2id)
test_dataset = PeptideHelmDataset(df_test, residue2id)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# === Step 7: Train model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNResidueEmbedding(num_residues=len(residue2id), emb_dim=32, hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


def evaluate(loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            y_true.append(batch.y)
            y_pred.append(out)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    mse = F.mse_loss(y_pred, y_true).item()
    return mse

for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    val_loss = evaluate(val_loader)
    print(f"Epoch {epoch:03d}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

def compute_relative_accuracy(loader, relative_threshold=0.1):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            relative_error = (out - batch.y).abs() / (batch.y.abs() + 1e-8)
            correct += (relative_error < relative_threshold).sum().item()
            total += batch.y.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy


def compute_accuracy(loader, threshold=1):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            diff = (out - batch.y).abs()
            correct += (diff < threshold).sum().item()
            total += batch.y.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy
# === Step 8: Final evaluation ===
test_loss = evaluate(test_loader)
print(f"Test MSE: {test_loss:.4f}")
print(compute_relative_accuracy(test_loader, relative_threshold=0.1))
print(compute_accuracy(test_loader, threshold=1))
