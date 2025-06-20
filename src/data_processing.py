import os
import json
import numpy as np
from typing import List, Dict
from .graph_builder import PrincipleSchemeGraphBuilder

class SchemeDataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.graph_builder = PrincipleSchemeGraphBuilder(config)
        
    def process_all_schemes(self, input_dir: str, output_dir: str):
        """Обработка всех схем в директории"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for filename in os.listdir(input_dir):
            if filename.endswith('.json') or filename.endswith('.xml'):  # предполагаем JSON/XML формат
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.graph")
                self.process_single_scheme(input_path, output_path)
    
    def process_single_scheme(self, input_path: str, output_path: str):
        """Обработка одной схемы и сохранение графа"""
        # Загрузка и парсинг схемы
        scheme_data = self._load_scheme(input_path)
        
        # Построение графа
        graph = self.graph_builder.build_graph(scheme_data)
        
        # Сохранение графа
        self._save_graph(graph, output_path)
    
    def _load_scheme(self, filepath: str) -> Dict:
        """Загрузка схемы из файла"""
        # Реализация загрузки в зависимости от формата
        pass
    
    def _save_graph(self, graph: Dict, filepath: str):
        """Сохранение графа в файл"""
        with open(filepath, 'w') as f:
            json.dump(graph, f)