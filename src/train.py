import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader  # Исправленный импорт
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import yaml
from pathlib import Path

class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

def load_graph_dataset(data_path):
    """Загрузка данных с проверкой наличия файлов"""
    from utils import GraphDataset
    
    train_path = data_path / 'train'
    val_path = data_path / 'val'
    
    # Проверка наличия данных
    if not os.path.exists(train_path) or len(os.listdir(train_path)) == 0:
        raise FileNotFoundError(f"No training data found in {train_path}")
    if not os.path.exists(val_path) or len(os.listdir(val_path)) == 0:
        raise FileNotFoundError(f"No validation data found in {val_path}")
    
    return GraphDataset(train_path), GraphDataset(val_path)

def train_model(config):
    # Инициализация
    base_dir = Path(__file__).parent.parent
    processed_path = base_dir / config['data']['processed_path']
    
    try:
        train_dataset, val_dataset = load_graph_dataset(processed_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please run data processing first!")
        sys.exit(1)
    
    # Проверка данных
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: No data available for training")
        sys.exit(1)
    
    # DataLoader с новым импортом
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['model']['batch_size']
    )
    
    # Инициализация модели
    model = GraphSAGEModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config['model']['lr'])
    criterion = nn.BCELoss()
    
    # Обучение
    best_val_loss = float('inf')
    for epoch in range(config['model']['epochs']):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        val_loss = validate_model(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{config['model']['epochs']}, "
              f"Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), base_dir / config['model']['model_save_path'])

def validate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index)
            total_loss += criterion(out, batch.y).item()
    return total_loss / len(loader)

if __name__ == "__main__":
    try:
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / 'config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Создание директорий
        os.makedirs(base_dir / 'models', exist_ok=True)
        os.makedirs(base_dir / 'logs', exist_ok=True)
        os.makedirs(base_dir / config['data']['processed_path'] / 'train', exist_ok=True)
        os.makedirs(base_dir / config['data']['processed_path'] / 'val', exist_ok=True)
        
        train_model(config)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)