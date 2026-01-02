"""
Модуль для загрузки и подготовки данных
Загружает данные с API и сохраняет их локально для повторного использования
"""

import os
import requests
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import Config


class DataLoader:
    """
    Класс для загрузки и обработки данных
    Сохраняет данные локально, чтобы избежать повторных обращений к API
    """

    def __init__(self):
        self.config = Config()
        self._create_directories()

    def _create_directories(self):
        """Создать необходимые директории"""
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.PLOTS_DIR, exist_ok=True)
        os.makedirs(self.config.MODELS_DIR, exist_ok=True)
        print(f"Директории созданы: {self.config.DATA_DIR}, {self.config.OUTPUT_DIR}, {self.config.PLOTS_DIR}, {self.config.MODELS_DIR}")

    def download_wine_dataset(self):
        """
        Скачать датасет Wine Quality с UCI ML Repository
        Сохраняет данные локально
        """
        filepath = os.path.join(self.config.DATA_DIR, self.config.WINE_DATASET_NAME)

        # Проверка существования файла
        if os.path.exists(filepath):
            print(f"Файл {self.config.WINE_DATASET_NAME} уже существует. Загрузка из локального файла.")
            return filepath

        print(f"Загрузка датасета Wine Quality с {self.config.WINE_DATASET_URL}...")
        try:
            response = requests.get(self.config.WINE_DATASET_URL, stream=True, timeout=30)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Датасет успешно загружен и сохранен в {filepath}")
            return filepath

        except Exception as e:
            print(f"Ошибка при загрузке датасета: {str(e)}")
            return None

    def load_wine_data(self):
        """
        Загрузить и обработать датасет Wine Quality
        Возвращает DataFrame с данными
        """
        filepath = self.download_wine_dataset()

        if filepath is None:
            raise Exception("Не удалось загрузить датасет")

        try:
            # Загрузка данных (CSV с разделителем точка с запятой)
            df = pd.read_csv(filepath, sep=';')
            print(f"\nДатасет Wine Quality загружен успешно!")
            print(f"Размер датасета: {df.shape}")
            print(f"\nПервые строки данных:\n{df.head()}")
            print(f"\nОписание данных:\n{df.describe()}")
            print(f"\nРаспределение качества вина:\n{df['quality'].value_counts().sort_index()}")

            # Сохранить информацию о датасете в файл
            self._save_dataset_info(df)

            return df

        except Exception as e:
            print(f"Ошибка при загрузке данных: {str(e)}")
            raise

    def prepare_classification_data(self, force_reload=False):
        """
        Подготовить данные для задачи бинарной классификации
        Сохраняет данные локально для повторного использования

        Args:
            force_reload: Принудительно перезагрузить данные с API

        Returns:
            Словарь с данными для обучения и тестирования
        """
        data_file = os.path.join(self.config.DATA_DIR, self.config.CLASSIFICATION_DATA_FILE)

        # Проверка существования сохраненных данных
        if os.path.exists(data_file) and not force_reload:
            print(f"\nЗагрузка данных классификации из файла {data_file}...")
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            print("Данные классификации загружены из локального файла.")
            return data

        print("\nПодготовка данных для бинарной классификации...")

        # Загрузка данных
        df = self.load_wine_data()

        # Создание бинарной целевой переменной
        # quality >= 6 -> 1 (хорошее вино), quality < 6 -> 0 (плохое вино)
        df['quality_binary'] = (df['quality'] >= self.config.CLASSIFICATION_THRESHOLD).astype(int)

        # Разделение на признаки и целевую переменную
        X = df.drop(['quality', 'quality_binary'], axis=1)
        y = df['quality_binary']

        print(f"\nРаспределение классов:")
        print(f"Класс 0 (плохое вино): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
        print(f"Класс 1 (хорошее вино): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")

        # Разделение на train и test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )

        # Стандартизация данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Сохранение данных
        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': scaler,
            'feature_names': list(X.columns)
        }

        # Сохранить данные локально
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Данные классификации сохранены в {data_file}")

        print(f"\nДанные классификации подготовлены:")
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        return data

    def prepare_regression_data(self, force_reload=False):
        """
        Подготовить данные для задачи регрессии
        Сохраняет данные локально для повторного использования

        Args:
            force_reload: Принудительно перезагрузить данные с API

        Returns:
            Словарь с данными для обучения и тестирования
        """
        data_file = os.path.join(self.config.DATA_DIR, self.config.REGRESSION_DATA_FILE)

        # Проверка существования сохраненных данных
        if os.path.exists(data_file) and not force_reload:
            print(f"\nЗагрузка данных регрессии из файла {data_file}...")
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            print("Данные регрессии загружены из локального файла.")
            return data

        print("\nПодготовка данных для регрессии...")

        # Загрузка данных
        df = self.load_wine_data()

        # Разделение на признаки и целевую переменную
        X = df.drop('quality', axis=1)
        y = df['quality']

        print(f"\nСтатистика целевой переменной (quality):")
        print(y.describe())

        # Разделение на train и test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )

        # Стандартизация данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Сохранение данных
        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': scaler,
            'feature_names': list(X.columns)
        }

        # Сохранить данные локально
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Данные регрессии сохранены в {data_file}")

        print(f"\nДанные регрессии подготовлены:")
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        return data

    def _save_dataset_info(self, df):
        """Сохранить информацию о датасете в файл"""
        info_file = os.path.join(self.config.OUTPUT_DIR, "dataset_info.txt")

        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ИНФОРМАЦИЯ О ДАТАСЕТЕ WINE QUALITY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Размер датасета: {df.shape}\n")
            f.write(f"Количество записей: {df.shape[0]}\n")
            f.write(f"Количество признаков: {df.shape[1]}\n\n")

            f.write("Признаки:\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"  {i}. {col}\n")
            f.write("\n")

            f.write("Типы данных:\n")
            f.write(str(df.dtypes) + "\n\n")

            f.write("Статистическое описание:\n")
            f.write(str(df.describe()) + "\n\n")

            f.write("Пропущенные значения:\n")
            missing = df.isnull().sum()
            if missing.sum() == 0:
                f.write("  Пропущенных значений нет\n\n")
            else:
                f.write(str(missing[missing > 0]) + "\n\n")

            f.write("Распределение качества вина:\n")
            f.write(str(df['quality'].value_counts().sort_index()) + "\n\n")

        print(f"Информация о датасете сохранена в {info_file}")


if __name__ == "__main__":
    # Тестирование загрузки данных
    loader = DataLoader()

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАГРУЗКИ ДАННЫХ")
    print("=" * 80)

    # Загрузка данных для классификации
    classification_data = loader.prepare_classification_data(force_reload=True)
    print("\n✓ Данные классификации загружены успешно")

    # Загрузка данных для регрессии
    regression_data = loader.prepare_regression_data(force_reload=True)
    print("\n✓ Данные регрессии загружены успешно")

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
