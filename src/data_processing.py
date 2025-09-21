import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
import yaml
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataValidator:
    @staticmethod
    def check_project_structure():
        """Проверяет базовую структуру проекта"""
        base_dir = Path(__file__).parent.parent
        required_dirs = {
            'data/raw': "Исходные SVG-схемы",
            'data/processed/train': "Обработанные данные для обучения",
            'data/processed/val': "Обработанные данные для валидации"
        }

        logger.info("=== Проверка структуры проекта ===")
        
        missing = []
        for dir_path, desc in required_dirs.items():
            full_path = base_dir / dir_path
            if not full_path.exists():
                missing.append(f"- {full_path}: {desc}")
                full_path.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Создана отсутствующая папка: {full_path}")

        if missing:
            logger.warning("Обнаружены отсутствующие папки:\n" + "\n".join(missing))
        
        # Проверка наличия исходных данных
        svg_files = list((base_dir / 'data/raw').glob('*.svg'))
        if not svg_files:
            logger.error("В data/raw/ не найдено SVG-файлов! Добавьте схемы для обработки.")
            sys.exit(1)
            
        logger.info(f"Найдено {len(svg_files)} SVG-файлов в data/raw/")
        logger.info(f"Примеры файлов: {[f.name for f in svg_files[:3]]}")

        return base_dir

class SchemeDataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.base_dir = Path(__file__).parent.parent
        self.raw_path = self.base_dir / config['data']['raw_path']
        self.processed_path = self.base_dir / config['data']['processed_path']
        
        # Создаем папки при инициализации
        os.makedirs(self.processed_path / 'train', exist_ok=True)
        os.makedirs(self.processed_path / 'val', exist_ok=True)

    def _validate_svg(self, filepath: Path) -> bool:
        """Проверяет валидность SVG файла"""
        try:
            ET.parse(filepath)
            return True
        except ET.ParseError:
            logger.error(f"Ошибка парсинга SVG: {filepath}")
            return False

    def process_all_schemes(self):
        """Обрабатывает все схемы в директории"""
        DataValidator.check_project_structure()
        
        svg_files = list(self.raw_path.glob('*.svg'))
        if not svg_files:
            logger.error("Не найдено SVG-файлов для обработки!")
            return

        logger.info(f"Начало обработки {len(svg_files)} схем...")
        
        processed_count = 0
        for svg_file in tqdm(svg_files, desc="Обработка схем"):
            if not self._validate_svg(svg_file):
                continue
                
            try:
                output_filename = f"{svg_file.stem}.graph"
                # Распределяем по train/val
                subset = 'train' if (processed_count % 10) < 8 else 'val'  # 80/20
                output_path = self.processed_path / subset / output_filename
                
                # Здесь должна быть ваша логика обработки
                graph_data = self._process_single_scheme(svg_file)
                
                with open(output_path, 'w') as f:
                    json.dump(graph_data, f, indent=2)
                
                processed_count += 1
            except Exception as e:
                logger.error(f"Ошибка обработки {svg_file.name}: {str(e)}")

        logger.info(f"Успешно обработано {processed_count}/{len(svg_files)} схем")
        logger.info(f"Результаты сохранены в: {self.processed_path}")

    def _process_single_scheme(self, filepath: Path) -> Dict:
        """Обрабатывает одну схему и возвращает граф"""
        # Ваша реализация парсинга SVG и преобразования в граф
        return {
            "nodes": [],
            "edges": [],
            "metadata": {
                "source_file": filepath.name,
                "processed_at": datetime.now().isoformat()
            }
        }

def load_config() -> Dict:
    """Загружает конфигурацию с проверкой"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Валидация обязательных параметров
        required_keys = ['data.raw_path', 'data.processed_path']
        for key in required_keys:
            if not config.get(*key.split('.')):
                raise ValueError(f"В конфиге отсутствует обязательный параметр: {key}")
                
        return config
    except Exception as e:
        logger.error(f"Ошибка загрузки конфига: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        logger.info("=== Запуск обработки данных ===")
        
        # Проверка структуры перед началом
        DataValidator.check_project_structure()
        
        # Загрузка конфига
        config = load_config()
        
        # Инициализация и запуск обработчика
        processor = SchemeDataProcessor(config)
        processor.process_all_schemes()
        
        logger.info("Обработка завершена успешно!")
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}", exc_info=True)
        sys.exit(1)