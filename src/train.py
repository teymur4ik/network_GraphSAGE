import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.data import DataLoader
from .utils import load_graph_dataset

class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGEModel, self).__init__()
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

def train_model(config):
    # Загрузка данных
    train_dataset, val_dataset = load_graph_dataset(config['data_path'])
    
    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Инициализация модели
    model = GraphSAGEModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim']
    )
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.BCELoss()
    
    # Обучение
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Валидация
        val_loss = validate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader)}, Val Loss: {val_loss}")
    
    # Сохранение модели
    torch.save(model.state_dict(), config['model_save_path'])

def validate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)