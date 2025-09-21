import torch
from torch_geometric.data import Data
from .graph_builder import PrincipleSchemeGraphBuilder

class SchemeErrorDetector:
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.model = self._load_model(model_path)
        self.graph_builder = PrincipleSchemeGraphBuilder(config)
    
    def detect_errors(self, scheme_data: Dict) -> Dict:
        """Обнаружение ошибок в принципиальной схеме"""
        # Построение графа
        graph_dict = self.graph_builder.build_graph(scheme_data)
        graph_data = self._convert_to_pyg_data(graph_dict)
        
        # Предсказание
        with torch.no_grad():
            predictions = self.model(graph_data.x, graph_data.edge_index)
        
        # Анализ предсказаний
        errors = self._analyze_predictions(predictions, graph_dict)
        return errors
    
    def _load_model(self, model_path: str):
        """Загрузка обученной модели"""
        model = GraphSAGEModel(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim']
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def _convert_to_pyg_data(self, graph_dict: Dict) -> Data:
        """Конвертация графа в формат PyTorch Geometric"""
        from torch_geometric.utils import from_networkx
        import networkx as nx

        G = nx.Graph()
        for node in graph_dict['nodes']:
            G.add_node(node['id'], **node)
        
        for edge in graph_dict['edges']:
            G.add_edge(edge['from'], edge['to'], **edge)
        
        # Преобразование в Data
        pyg_data = from_networkx(G)
        pyg_data.x = torch.tensor([node['features'] for node in graph_dict['nodes']], dtype=torch.float)
        pyg_data.edge_index = torch.tensor([[edge['from'], edge['to']] for edge in graph_dict['edges']], dtype=torch.long).t().contiguous()
        return pyg_data

    def _analyze_predictions(self, predictions: torch.Tensor, graph: Dict) -> Dict:
        """Анализ предсказаний (пример: поиск подозрительных узлов)"""
        errors = {}
        for i, (node, pred) in enumerate(zip(graph['nodes'], predictions)):
            if pred.item() > 0.5:  # Порог для ошибки
                errors[node['id']] = {
                    'type': node['type'],
                    'confidence': float(pred.item())
                }
        return {'errors': errors}