"""
Файл конфигурации для лабораторной работы №3
Получение объяснений от моделей машинного обучения
"""

class Config:
    # ==================== НАСТРОЙКИ ДАТАСЕТА ====================

    # URL для скачивания датасета Wine Quality (для классификации и регрессии)
    WINE_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    WINE_DATASET_NAME = "winequality-red.csv"

    # Директории проекта
    DATA_DIR = "data"
    OUTPUT_DIR = "results"
    PLOTS_DIR = "plots"
    MODELS_DIR = "models"

    # Файлы с данными (для повторного использования без обращения к API)
    CLASSIFICATION_DATA_FILE = "classification_data.pkl"
    REGRESSION_DATA_FILE = "regression_data.pkl"

    # ==================== НАСТРОЙКИ ОБУЧЕНИЯ ====================

    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2  # Из тренировочной выборки

    # ==================== ПАРАМЕТРЫ МОДЕЛЕЙ ====================

    # Параметры для моделей классификации
    CLASSIFICATION_MODELS = {
        'GradientBoosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.8,
            'random_state': 42
        },
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        },
        'LogisticRegression': {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        }
    }

    # Параметры для моделей регрессии
    REGRESSION_MODELS = {
        'GradientBoosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.8,
            'random_state': 42
        },
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
    }

    # Параметры для нейронных сетей
    NEURAL_NETWORK_PARAMS = {
        'classifier': {
            'hidden_layers': [64, 32, 16],
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 10
        },
        'regressor': {
            'hidden_layers': [64, 32, 16],
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 10
        }
    }

    # ==================== НАСТРОЙКИ ОБЪЯСНЕНИЙ ====================

    # Настройки для SHAP
    SHAP_PARAMS = {
        'max_display': 10,  # Максимальное количество признаков для отображения
        'sample_size': 100,  # Количество образцов для вычисления SHAP
        'check_additivity': False  # Отключить проверку аддитивности для ускорения
    }

    # Настройки для Partial Dependence Plots
    PDP_PARAMS = {
        'n_jobs': -1,
        'grid_resolution': 50,  # Разрешение сетки для PDP
        'n_cols': 2  # Количество колонок в сетке графиков
    }

    # Настройки для Permutation Importance
    PERMUTATION_PARAMS = {
        'n_repeats': 10,
        'n_jobs': -1,
        'random_state': 42
    }

    # ==================== НАСТРОЙКИ ВИЗУАЛИЗАЦИИ ====================

    FIGURE_SIZE = (12, 8)
    FIGURE_SIZE_LARGE = (16, 12)
    DPI = 300
    FONT_SIZE = 10

    # Цветовая палитра
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff9800',
        'info': '#17a2b8'
    }

    # ==================== НАСТРОЙКИ ОТЧЕТА ====================

    REPORT_FILE = "LAB3_REPORT.txt"
    REPORT_MD_FILE = "LAB3_REPORT.md"
    METRICS_FILE = "model_metrics.csv"

    # ==================== НАСТРОЙКИ ЗАДАЧИ ====================

    # Для классификации: бинарная классификация (хорошее вино / плохое вино)
    # quality >= 6 -> 1 (хорошее), quality < 6 -> 0 (плохое)
    CLASSIFICATION_THRESHOLD = 6

    # Для регрессии: предсказание качества вина (quality)
    # Целевая переменная: quality (от 3 до 8)

    # Количество примеров для индивидуальных объяснений
    N_INDIVIDUAL_EXAMPLES = 5

    # Включить/выключить сохранение моделей
    SAVE_MODELS = True

    # Включить/выключить подробный вывод
    VERBOSE = True
