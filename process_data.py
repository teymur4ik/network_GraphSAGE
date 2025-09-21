import yaml
from src.data_processing import SchemeDataProcessor

def main():
    # Загрузка конфига
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Инициализация процессора
    processor = SchemeDataProcessor(config)
    
    # Обработка всех схем из data/raw и сохранение в data/processed
    processor.process_all_schemes(
        input_dir=config['data']['raw_path'],
        output_dir=config['data']['processed_path']
    )
    print("Обработка данных завершена!")

if __name__ == "__main__":
    main()