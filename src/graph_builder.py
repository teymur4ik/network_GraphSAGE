from typing import Dict, List, Optional
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from dataclasses import dataclass
import json

@dataclass
class ComponentFeatures:
    """Контейнер для признаков компонента с валидацией"""
    voltage: float = 0.0
    power: float = 0.0
    current: float = 0.0
    is_grounded: int = 0
    is_dimmable: int = 0

class PrincipleSchemeGraphBuilder:
    def __init__(self, config: Dict):
        self.config = config
        self.component_rules = config['components']
        self.connection_rules = config['connections']
        self.feature_size = self._calculate_feature_size()

    def build_graph(self, scheme_data: Dict) -> Dict:
        """Построение графа из данных схемы с валидацией"""
        G = nx.Graph()
        
        # Добавление узлов (компонентов)
        for component in scheme_data['components']:
            features, errors = self._process_component(component)
            G.add_node(
                component['id'],
                type=component['type'],
                features=features,
                errors=errors,
                pos=(component.get('x', 0), component.get('y', 0))
            )

        # Добавление рёбер (соединений)
        for connection in scheme_data['connections']:
            features, errors = self._process_connection(connection)
            G.add_edge(
                connection['from'],
                connection['to'],
                type=connection['type'],
                features=features,
                errors=errors,
                length=connection.get('length', 0.0)
            )

        return self._convert_to_dict(G)

    def _process_component(self, component: Dict) -> (List[float], List[str]):
        """Извлечение признаков и валидация компонента"""
        comp_type = component['type']
        features = ComponentFeatures()
        errors = []

        # Заполнение признаков
        if comp_type in self.component_rules:
            for feature in self.component_rules[comp_type]['features']:
                if hasattr(features, feature):
                    setattr(features, feature, component.get(feature, 0))
        
        # Валидация
        if comp_type in self.component_rules:
            rules = self.component_rules[comp_type].get('rules', {})
            if 'min_voltage' in rules and features.voltage < rules['min_voltage']:
                errors.append(f"Low voltage: {features.voltage} < {rules['min_voltage']}")
            if 'max_power' in rules and features.power > rules['max_power']:
                errors.append(f"High power: {features.power} > {rules['max_power']}")

        # Конвертация в список признаков
        feature_list = [
            features.voltage,
            features.power,
            features.current,
            features.is_grounded,
            features.is_dimmable
        ]
        return feature_list, errors

    def _process_connection(self, connection: Dict) -> (List[float], List[str]):
        """Обработка соединения между компонентами"""
        features = []
        errors = []
        conn_type = connection['type']
        
        # One-hot encoding типа соединения
        features.extend([
            1 if conn_type == t else 0 
            for t in self.connection_rules['types']
        ])
        
        # Добавление дополнительных признаков
        features.append(connection.get('length', 0.0))
        features.append(1 if connection.get('is_ground', False) else 0)
        
        # Валидация
        if conn_type not in self.connection_rules['types']:
            errors.append(f"Invalid connection type: {conn_type}")
        
        max_len = self.connection_rules['rules'].get('max_length', float('inf'))
        if 'length' in connection and connection['length'] > max_len:
            errors.append(f"Connection too long: {connection['length']} > {max_len}")
        
        return features, errors

    def _calculate_feature_size(self) -> int:
        """Вычисление размерности признаков на основе конфига"""
        # Максимальное кол-во признаков компонента
        max_comp_features = max(
            len(props['features']) 
            for props in self.component_rules.values()
        ) if self.component_rules else 5
        
        # Признаки соединений: типы + длина + is_ground
        conn_features = len(self.connection_rules['types']) + 2
        
        return max(max_comp_features, conn_features)

    def _convert_to_dict(self, graph: nx.Graph) -> Dict:
        """Конвертация NetworkX графа в словарь"""
        return {
            'nodes': [{
                'id': n,
                'type': graph.nodes[n]['type'],
                'features': graph.nodes[n]['features'],
                'errors': graph.nodes[n]['errors'],
                'x': graph.nodes[n]['pos'][0],
                'y': graph.nodes[n]['pos'][1]
            } for n in graph.nodes],
            'edges': [{
                'from': u,
                'to': v,
                'type': graph.edges[u, v]['type'],
                'features': graph.edges[u, v]['features'],
                'errors': graph.edges[u, v]['errors'],
                'length': graph.edges[u, v]['length']
            } for u, v in graph.edges],
            'feature_dim': self.feature_size
        }

    def save_graph(self, graph: Dict, filepath: str):
        """Сохранение графа в JSON файл"""
        with open(filepath, 'w') as f:
            json.dump(graph, f, indent=2)

    def load_graph(self, filepath: str) -> Dict:
        """Загрузка графа из JSON файла"""
        with open(filepath, 'r') as f:
            return json.load(f)

    def to_pyg_data(self, graph_dict: Dict) -> Data:
        """Конвертация в формат PyTorch Geometric"""
        x = torch.tensor(
            [node['features'] for node in graph_dict['nodes']],
            dtype=torch.float
        )
        
        edge_index = torch.tensor(
            [[edge['from'], edge['to']] for edge in graph_dict['edges']],
            dtype=torch.long
        ).t().contiguous()
        
        edge_attr = torch.tensor(
            [edge['features'] for edge in graph_dict['edges']],
            dtype=torch.float
        )
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([graph_dict.get('has_error', 0)], dtype=torch.float)
        )


# Пример использования
if __name__ == "__main__":
    import yaml
    import torch
    
    # Загрузка конфига
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Пример данных схемы
    sample_scheme = {
        "components": [
            {"id": "l1", "type": "lamp", "voltage": 220, "power": 60, "x": 10, "y": 20},
            {"id": "s1", "type": "switch", "current": 5, "is_dimmable": 1, "x": 10, "y": 40}
        ],
        "connections": [
            {"from": "l1", "to": "s1", "type": "wire", "length": 2.5}
        ]
    }
    
    # Построение графа
    builder = PrincipleSchemeGraphBuilder(config)
    graph = builder.build_graph(sample_scheme)
    
    # Сохранение и загрузка
    builder.save_graph(graph, "sample.graph")
    loaded = builder.load_graph("sample.graph")
    
    # Конвертация в PyG
    pyg_data = builder.to_pyg_data(loaded)
    print(f"PyG Data: {pyg_data}")