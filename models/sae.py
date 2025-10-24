import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- Sparse Autoencoder ---
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, sparsity=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity = sparsity

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon = F.mse_loss(x_hat, x)
        sparse = self.sparsity * torch.mean(torch.abs(z))
        return recon + sparse

# --- Training function ---
def train_sae(model, data, epochs=5, lr=1e-3, device="cuda"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for batch in data:   # data: list or DataLoader of tensors
            x = batch.to(device)
            x_hat, z = model(x)
            loss = model.loss(x, x_hat, z)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}: loss={total_loss/len(data):.4f}")
    return model

# --- Save / Load ---
def save_sae(model, path="models/sae_model.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_sae(path="models/sae_model.pt", input_dim=768, hidden_dim=256, sparsity=1e-3, device="cuda"):
    model = SparseAutoencoder(input_dim, hidden_dim, sparsity)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
