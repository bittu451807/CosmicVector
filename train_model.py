import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, gru_out):
        scores = self.V(torch.tanh(self.W1(gru_out)))
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * gru_out, dim=1)
        return context

class CosmicVectorModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_classes=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.attention = BahdanauAttention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        ctx = self.attention(out)
        return self.classifier(ctx)

def train_ai():
    X = np.load('data/X_train.npy')
    y = np.load('data/y_train.npy')
    
    class_counts = np.bincount(y, minlength=4)
    weights = len(y) / (4.0 * (class_counts + 1e-5)) 
    
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)), batch_size=1024, shuffle=True)
    model = CosmicVectorModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    
    model.train()
    for epoch in range(3):
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/3 Complete.")
        
    torch.save(model.state_dict(), 'data/solar_flare_gru.pt')
    print("💾 Model saved as 'data/solar_flare_gru.pt'")

if __name__ == "__main__":
    train_ai()
