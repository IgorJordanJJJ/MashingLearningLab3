# Лабораторная работа №3
## Получение объяснений от моделей машинного обучения

### Описание
Проект для получения и интерпретации объяснений моделей машинного обучения с использованием различных методов: SHAP, Partial Dependence Plots, Permutation Importance.

### Структура проекта
```
Lab3/
├── config.py                 # Конфигурация проекта
├── data_loader.py           # Загрузка и подготовка данных
├── model_trainer.py         # Обучение моделей
├── model_explainer.py       # Получение объяснений
├── report_generator.py      # Генерация отчетов
├── main.py                  # Главный скрипт
├── requirements.txt         # Зависимости
├── README.md               # Документация
│
├── data/                   # Данные
│   ├── winequality-red.csv
│   ├── classification_data.pkl
│   └── regression_data.pkl
│
├── results/                # Результаты
│   ├── LAB3_REPORT.txt
│   ├── model_metrics.csv
│   └── dataset_info.txt
│
├── plots/                  # Графики
│   ├── classification_*.png
│   └── regression_*.png
│
└── models/                 # Обученные модели
    ├── *_classifier.pkl
    └── *_regressor.pkl
```

### Установка зависимостей

```bash
# Создание виртуального окружения (опционально)
python -m venv .venv

# Активация виртуального окружения
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### Использование

#### Быстрый старт
Запустите главный скрипт для выполнения всей лабораторной работы:

```bash
python main.py
```

#### Поэтапное выполнение

1. **Загрузка данных**
```python
from data_loader import DataLoader

loader = DataLoader()
classification_data = loader.prepare_classification_data()
regression_data = loader.prepare_regression_data()
```

2. **Обучение моделей**
```python
from model_trainer import ClassificationTrainer, RegressionTrainer

# Классификация
clf_trainer = ClassificationTrainer()
model_gb, metrics = clf_trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
model_rf, metrics = clf_trainer.train_random_forest(X_train, y_train, X_test, y_test)
model_lr, metrics = clf_trainer.train_logistic_regression(X_train, y_train, X_test, y_test)

# Регрессия
reg_trainer = RegressionTrainer()
model_gb, metrics = reg_trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
model_rf, metrics = reg_trainer.train_random_forest(X_train, y_train, X_test, y_test)
```

3. **Получение объяснений**
```python
from model_explainer import ModelExplainer

explainer = ModelExplainer(
    model=model,
    model_name='GradientBoosting',
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    task_type='classification'
)

# Генерация всех объяснений
explainer.generate_all_explanations()

# Или по отдельности:
explainer.get_feature_importance()
explainer.get_permutation_importance()
explainer.get_partial_dependence()
explainer.calculate_shap_values()
explainer.plot_shap_summary()
```

4. **Генерация отчета**
```python
from report_generator import ReportGenerator

report = ReportGenerator()
report.generate_title_page()
report.add_classification_results(classification_results)
report.add_regression_results(regression_results)
report.save_report()
```

### Методы объяснений

#### 1. Feature Importance
- **Для моделей на основе деревьев**: Критерий Information Gain (IG)
- **Для логистической регрессии**: Коэффициенты модели
- **Визуализация**: Bar plots

#### 2. Permutation Importance
- Измеряет ухудшение качества модели при случайной перестановке значений признака
- Работает для любых моделей
- **Визуализация**: Bar plots с error bars

#### 3. Partial Dependence Plots (PDP)
- Показывает зависимость предсказания от значений признака
- Усредняет влияние остальных признаков
- **Визуализация**: Line plots

#### 4. SHAP Values
- **TreeExplainer**: Для моделей на основе деревьев
- **LinearExplainer**: Для линейных моделей
- **KernelExplainer**: Для нейронных сетей
- **Визуализации**:
  - Summary Plot - общая важность признаков
  - Waterfall Plot - пошаговое влияние на конкретное предсказание
  - Force Plot - интерактивная визуализация

### Обученные модели

#### Классификация (бинарная)
- Gradient Boosting Classifier
- Random Forest Classifier
- Logistic Regression
- Neural Network Classifier

#### Регрессия
- Gradient Boosting Regressor
- Random Forest Regressor
- Neural Network Regressor

### Датасет

**Wine Quality Dataset (Red Wine)**
- Источник: UCI Machine Learning Repository
- Размер: ~1599 записей
- Признаков: 11
- Целевая переменная:
  - Классификация: quality >= 6 (хорошее вино) vs quality < 6 (плохое вино)
  - Регрессия: quality (оценка от 3 до 8)

### Результаты

Все результаты сохраняются в соответствующие директории:
- `results/LAB3_REPORT.txt` - полный отчет
- `results/model_metrics.csv` - метрики в CSV
- `plots/` - все графики в PNG
- `models/` - обученные модели

### Настройка параметров

Все параметры можно изменить в файле `config.py`:
- Параметры моделей
- Настройки SHAP
- Параметры PDP
- Параметры Permutation Importance
- Настройки визуализации

### Примечания

1. **Кэширование данных**: После первой загрузки данные сохраняются локально в `data/` и при повторном запуске используются сохраненные данные (не обращается к API)

2. **SHAP для нейронных сетей**: Вычисление SHAP values для нейронных сетей может занять длительное время. По умолчанию используется 100 примеров.

3. **Графики**: Все графики сохраняются в высоком разрешении (DPI=300) в формате PNG.

4. **Модели**: Обученные модели автоматически сохраняются в директории `models/`.

### Требования

- Python 3.8+
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Scikit-learn >= 1.3.0
- TensorFlow >= 2.13.0
- SHAP >= 0.43.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0

### Автор

Студент группы 8ПМ41 Иордан Игорь

### Лицензия

Проект создан в образовательных целях для выполнения лабораторной работы.
