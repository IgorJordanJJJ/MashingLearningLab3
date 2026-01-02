"""
Главный скрипт для выполнения лабораторной работы №3
Получение объяснений от моделей машинного обучения

Автор: Студент группы [ГРУППА]
Дата: 2024
"""

import os
import warnings

# Установка backend для matplotlib ДО всех импортов
os.environ['MPLBACKEND'] = 'Agg'  # Установка через переменную окружения
import matplotlib
matplotlib.use('Agg')  # Дополнительная установка для надежности

import numpy as np

# Отключение предупреждений для чистоты вывода
warnings.filterwarnings('ignore')
# Подавление предупреждений joblib о временных файлах
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключение логов TensorFlow

from config import Config
from data_loader import DataLoader
from model_trainer import ClassificationTrainer, RegressionTrainer
from model_explainer import ModelExplainer
from report_generator import ReportGenerator


def print_banner(text):
    """Красивый баннер для вывода"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def main():
    """
    Главная функция для выполнения всей лабораторной работы
    """
    print_banner("ЛАБОРАТОРНАЯ РАБОТА №3")
    print_banner("Получение объяснений от моделей машинного обучения")

    config = Config()

    # ==================== ЭТАП 1: ЗАГРУЗКА ДАННЫХ ====================
    print_banner("ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")

    loader = DataLoader()

    # Загрузка данных для классификации
    print("\n>>> Загрузка данных для классификации...")
    classification_data = loader.prepare_classification_data(force_reload=False)

    # Загрузка данных для регрессии
    print("\n>>> Загрузка данных для регрессии...")
    regression_data = loader.prepare_regression_data(force_reload=False)

    print("\n✓ Данные успешно загружены и подготовлены!")

    # ==================== ЭТАП 2: ОБУЧЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ ====================
    print_banner("ЭТАП 2: ОБУЧЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ")

    classification_trainer = ClassificationTrainer()
    classification_models = {}
    classification_results = {}

    # Обучение Gradient Boosting
    print("\n>>> Обучение Gradient Boosting Classifier...")
    model_gb, metrics_gb = classification_trainer.train_gradient_boosting(
        classification_data['X_train'],
        classification_data['y_train'],
        classification_data['X_test'],
        classification_data['y_test']
    )
    classification_models['GradientBoosting'] = model_gb
    classification_results['GradientBoosting'] = metrics_gb

    # Обучение Random Forest
    print("\n>>> Обучение Random Forest Classifier...")
    model_rf, metrics_rf = classification_trainer.train_random_forest(
        classification_data['X_train'],
        classification_data['y_train'],
        classification_data['X_test'],
        classification_data['y_test']
    )
    classification_models['RandomForest'] = model_rf
    classification_results['RandomForest'] = metrics_rf

    # Обучение Logistic Regression
    print("\n>>> Обучение Logistic Regression...")
    model_lr, metrics_lr = classification_trainer.train_logistic_regression(
        classification_data['X_train_scaled'],
        classification_data['y_train'],
        classification_data['X_test_scaled'],
        classification_data['y_test']
    )
    classification_models['LogisticRegression'] = model_lr
    classification_results['LogisticRegression'] = metrics_lr

    # Обучение нейронной сети
    print("\n>>> Обучение Neural Network Classifier...")
    model_nn, metrics_nn = classification_trainer.train_neural_network(
        classification_data['X_train_scaled'],
        classification_data['y_train'],
        classification_data['X_test_scaled'],
        classification_data['y_test']
    )
    classification_models['NeuralNetwork'] = model_nn
    classification_results['NeuralNetwork'] = metrics_nn

    # Сохранение моделей
    classification_trainer.save_models()

    print("\n✓ Все модели классификации обучены!")

    # ==================== ЭТАП 3: ОБУЧЕНИЕ МОДЕЛЕЙ РЕГРЕССИИ ====================
    print_banner("ЭТАП 3: ОБУЧЕНИЕ МОДЕЛЕЙ РЕГРЕССИИ")

    regression_trainer = RegressionTrainer()
    regression_models = {}
    regression_results = {}

    # Обучение Gradient Boosting
    print("\n>>> Обучение Gradient Boosting Regressor...")
    model_gb_reg, metrics_gb_reg = regression_trainer.train_gradient_boosting(
        regression_data['X_train'],
        regression_data['y_train'],
        regression_data['X_test'],
        regression_data['y_test']
    )
    regression_models['GradientBoosting'] = model_gb_reg
    regression_results['GradientBoosting'] = metrics_gb_reg

    # Обучение Random Forest
    print("\n>>> Обучение Random Forest Regressor...")
    model_rf_reg, metrics_rf_reg = regression_trainer.train_random_forest(
        regression_data['X_train'],
        regression_data['y_train'],
        regression_data['X_test'],
        regression_data['y_test']
    )
    regression_models['RandomForest'] = model_rf_reg
    regression_results['RandomForest'] = metrics_rf_reg

    # Обучение нейронной сети
    print("\n>>> Обучение Neural Network Regressor...")
    model_nn_reg, metrics_nn_reg = regression_trainer.train_neural_network(
        regression_data['X_train_scaled'],
        regression_data['y_train'],
        regression_data['X_test_scaled'],
        regression_data['y_test']
    )
    regression_models['NeuralNetwork'] = model_nn_reg
    regression_results['NeuralNetwork'] = metrics_nn_reg

    # Сохранение моделей
    regression_trainer.save_models()

    print("\n✓ Все модели регрессии обучены!")

    # ==================== ЭТАП 4: ПОЛУЧЕНИЕ ОБЪЯСНЕНИЙ ДЛЯ КЛАССИФИКАЦИИ ====================
    print_banner("ЭТАП 4: ПОЛУЧЕНИЕ ОБЪЯСНЕНИЙ ДЛЯ МОДЕЛЕЙ КЛАССИФИКАЦИИ")

    for model_name, model in classification_models.items():
        print(f"\n{'=' * 80}")
        print(f"Получение объяснений для модели: {model_name}")
        print(f"{'=' * 80}")

        # Выбор данных (для LogisticRegression и NeuralNetwork используем нормализованные)
        if model_name in ['LogisticRegression', 'NeuralNetwork']:
            X_train = classification_data['X_train_scaled']
            X_test = classification_data['X_test_scaled']
        else:
            X_train = classification_data['X_train']
            X_test = classification_data['X_test']

        # Создание explainer
        explainer = ModelExplainer(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_test=classification_data['y_test'],
            feature_names=classification_data['feature_names'],
            task_type='classification'
        )

        # Генерация всех объяснений
        explainer.generate_all_explanations()

    print("\n✓ Объяснения для всех моделей классификации получены!")

    # ==================== ЭТАП 5: ПОЛУЧЕНИЕ ОБЪЯСНЕНИЙ ДЛЯ РЕГРЕССИИ ====================
    print_banner("ЭТАП 5: ПОЛУЧЕНИЕ ОБЪЯСНЕНИЙ ДЛЯ МОДЕЛЕЙ РЕГРЕССИИ")

    for model_name, model in regression_models.items():
        print(f"\n{'=' * 80}")
        print(f"Получение объяснений для модели: {model_name}")
        print(f"{'=' * 80}")

        # Выбор данных (для NeuralNetwork используем нормализованные)
        if model_name == 'NeuralNetwork':
            X_train = regression_data['X_train_scaled']
            X_test = regression_data['X_test_scaled']
        else:
            X_train = regression_data['X_train']
            X_test = regression_data['X_test']

        # Создание explainer
        explainer = ModelExplainer(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_test=regression_data['y_test'],
            feature_names=regression_data['feature_names'],
            task_type='regression'
        )

        # Генерация всех объяснений
        explainer.generate_all_explanations()

    print("\n✓ Объяснения для всех моделей регрессии получены!")

    # ==================== ЭТАП 6: ГЕНЕРАЦИЯ ОТЧЕТА ====================
    print_banner("ЭТАП 6: ГЕНЕРАЦИЯ ОТЧЕТА")

    report = ReportGenerator()

    # Титульная страница
    report.generate_title_page()

    # Описание датасета
    dataset_info = """
Название: Wine Quality Dataset (Red Wine)
Источник: UCI Machine Learning Repository

Описание:
Датасет содержит информацию о физико-химических свойствах красного вина и его качестве.
Каждая запись представляет собой образец вина с измеренными характеристиками.

Признаки (всего 11):
1. fixed acidity - фиксированная кислотность
2. volatile acidity - летучая кислотность
3. citric acid - лимонная кислота
4. residual sugar - остаточный сахар
5. chlorides - хлориды
6. free sulfur dioxide - свободный диоксид серы
7. total sulfur dioxide - общий диоксид серы
8. density - плотность
9. pH - уровень pH
10. sulphates - сульфаты
11. alcohol - содержание алкоголя

Целевая переменная:
- Для классификации: бинарная метка (0 - плохое вино (quality < 6), 1 - хорошее вино (quality >= 6))
- Для регрессии: quality - качество вина (оценка от 3 до 8)

Размер датасета: ~1599 записей
Разделение: 80% train, 20% test
"""
    report.add_dataset_description(dataset_info)

    # Параметры моделей
    report.add_model_parameters(
        config.CLASSIFICATION_MODELS,
        config.REGRESSION_MODELS,
        config.NEURAL_NETWORK_PARAMS
    )

    # Результаты классификации
    report.add_classification_results(classification_results)

    # Результаты регрессии
    report.add_regression_results(regression_results)

    # Секция объяснений
    report.add_explanations_section()

    # Выводы
    report.add_conclusions()

    # Сохранение отчета
    report.save_report()

    # Сохранение метрик в CSV
    report.save_metrics_csv(classification_results, regression_results)

    # ==================== ЗАВЕРШЕНИЕ ====================
    print_banner("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА!")

    print("""
✓ Все задачи выполнены успешно!

Результаты сохранены в директории:
  - data/         - загруженные данные
  - results/      - отчеты и метрики
  - plots/        - графики и визуализации
  - models/       - обученные модели

Основные файлы:
  - results/LAB3_REPORT.txt    - полный отчет о работе
  - results/model_metrics.csv  - метрики моделей в CSV
  - results/dataset_info.txt   - информация о датасете

Следующие шаги:
  1. Просмотрите отчет в results/LAB3_REPORT.txt
  2. Изучите графики в директории plots/
  3. Проанализируйте результаты объяснений
  4. Подготовьте презентацию результатов
""")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"ОШИБКА: {str(e)}")
        print(f"{'=' * 80}\n")
        import traceback
        traceback.print_exc()
    finally:
        # Явное закрытие всех фигур matplotlib при завершении
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
