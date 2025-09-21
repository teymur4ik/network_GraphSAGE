import os
import json
import torch
from torch_geometric.data import Data, Dataset

class GraphDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.files = [f for f in os.listdir(root) if f.endswith('.graph')]
    
    def len(self):
        return len(self.files)
    
    def get(self, idx):
        path = os.path.join(self.root, self.files[idx])
        with open(path, 'r') as f:
            graph_dict = json.load(f)
        
        x = torch.tensor([node['features'] for node in graph_dict['nodes']], dtype=torch.float)
        edge_index = torch.tensor([[edge['from'], edge['to']] for edge in graph_dict['edges']], dtype=torch.long).t().contiguous()
        y = torch.tensor([graph_dict.get('has_error', 0)], dtype=torch.float)  # Метка (0/1)
        return Data(x=x, edge_index=edge_index, y=y)

def load_graph_dataset(data_path):
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    return GraphDataset(train_path), GraphDataset(val_path)