"""
Модуль для генерации отчетов о работе моделей
Создает текстовые отчеты с результатами обучения и объяснениями
"""

import os
from datetime import datetime
import pandas as pd
from config import Config


class ReportGenerator:
    """
    Класс для генерации отчетов о результатах работы
    """

    def __init__(self):
        self.config = Config()
        self.report_lines = []

    def add_header(self, text, level=1):
        """Добавить заголовок в отчет"""
        if level == 1:
            self.report_lines.append("\n" + "=" * 80)
            self.report_lines.append(text.upper())
            self.report_lines.append("=" * 80 + "\n")
        elif level == 2:
            self.report_lines.append("\n" + "-" * 80)
            self.report_lines.append(text)
            self.report_lines.append("-" * 80 + "\n")
        else:
            self.report_lines.append(f"\n{text}\n")

    def add_text(self, text):
        """Добавить текст в отчет"""
        self.report_lines.append(text)

    def add_section(self, title, content):
        """Добавить секцию с содержимым"""
        self.add_header(title, level=2)
        self.add_text(content)

    def generate_title_page(self):
        """Генерация титульной страницы"""
        self.add_header("ЛАБОРАТОРНАЯ РАБОТА №3", level=1)
        self.add_text("Тема: Получение объяснений от моделей машинного обучения\n")
        self.add_text(f"Дата выполнения: {datetime.now().strftime('%d.%m.%Y')}\n")
        self.add_text("Цель работы:")
        self.add_text("Получить навыки получения и интерпретации объяснений моделей машинного обучения.\n")
        self.add_text("Задачи:")
        self.add_text("1. Обучить классические модели для задач классификации и регрессии")
        self.add_text("2. Получить графики частичной зависимости (PDP)")
        self.add_text("3. Получить важность признаков на основе перестановок")
        self.add_text("4. Вычислить значения Шепли (SHAP)")
        self.add_text("5. Визуализировать результаты объяснений")

    def add_dataset_description(self, dataset_info):
        """Добавить описание датасета"""
        self.add_header("1. ОПИСАНИЕ НАБОРА ДАННЫХ", level=1)
        self.add_text(dataset_info)

    def add_model_parameters(self, classification_params, regression_params, neural_params):
        """Добавить параметры моделей"""
        self.add_header("2. ПАРАМЕТРЫ МОДЕЛЕЙ", level=1)

        self.add_header("2.1. Параметры моделей классификации", level=2)
        for model_name, params in classification_params.items():
            self.add_text(f"\n{model_name}:")
            for param, value in params.items():
                self.add_text(f"  - {param}: {value}")

        self.add_header("2.2. Параметры моделей регрессии", level=2)
        for model_name, params in regression_params.items():
            self.add_text(f"\n{model_name}:")
            for param, value in params.items():
                self.add_text(f"  - {param}: {value}")

        self.add_header("2.3. Параметры нейронных сетей", level=2)
        self.add_text("\nКлассификатор:")
        for param, value in neural_params['classifier'].items():
            self.add_text(f"  - {param}: {value}")

        self.add_text("\nРегрессор:")
        for param, value in neural_params['regressor'].items():
            self.add_text(f"  - {param}: {value}")

    def add_classification_results(self, results):
        """Добавить результаты классификации"""
        self.add_header("3. РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ", level=1)

        # Создание таблицы с метриками
        metrics_data = []
        for model_name, metrics in results.items():
            metrics_data.append({
                'Модель': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })

        df = pd.DataFrame(metrics_data)
        self.add_text("\nТаблица 1. Метрики качества моделей классификации\n")
        self.add_text(df.to_string(index=False))

        # Подробные результаты для каждой модели
        for model_name, metrics in results.items():
            self.add_header(f"3.{list(results.keys()).index(model_name) + 1}. {model_name}", level=2)
            self.add_text(f"\nAccuracy: {metrics['accuracy']:.4f}")
            self.add_text(f"Precision: {metrics['precision']:.4f}")
            self.add_text(f"Recall: {metrics['recall']:.4f}")
            self.add_text(f"F1-Score: {metrics['f1']:.4f}")
            self.add_text(f"ROC-AUC: {metrics['roc_auc']:.4f}")

            self.add_text("\nМатрица ошибок:")
            self.add_text(str(metrics['confusion_matrix']))

            self.add_text("\nОтчет классификации:")
            self.add_text(metrics['classification_report'])

    def add_regression_results(self, results):
        """Добавить результаты регрессии"""
        self.add_header("4. РЕЗУЛЬТАТЫ РЕГРЕССИИ", level=1)

        # Создание таблицы с метриками
        metrics_data = []
        for model_name, metrics in results.items():
            metrics_data.append({
                'Модель': model_name,
                'R² Score': f"{metrics['r2']:.4f}",
                'RMSE': f"{metrics['rmse']:.4f}",
                'MAE': f"{metrics['mae']:.4f}",
                'MSE': f"{metrics['mse']:.4f}"
            })

        df = pd.DataFrame(metrics_data)
        self.add_text("\nТаблица 2. Метрики качества моделей регрессии\n")
        self.add_text(df.to_string(index=False))

        # Подробные результаты для каждой модели
        for model_name, metrics in results.items():
            self.add_header(f"4.{list(results.keys()).index(model_name) + 1}. {model_name}", level=2)
            self.add_text(f"\nR² Score: {metrics['r2']:.4f}")
            self.add_text(f"RMSE: {metrics['rmse']:.4f}")
            self.add_text(f"MAE: {metrics['mae']:.4f}")
            self.add_text(f"MSE: {metrics['mse']:.4f}")

    def add_explanations_section(self):
        """Добавить секцию с объяснениями"""
        self.add_header("5. ОБЪЯСНЕНИЯ МОДЕЛЕЙ", level=1)

        self.add_text("""
Для получения объяснений моделей были использованы следующие методы:

5.1. Feature Importance (Важность признаков)
   - Для моделей на основе деревьев (Gradient Boosting, Random Forest) используется
     критерий Information Gain (IG), который показывает насколько каждый признак
     важен для принятия решений в дереве.
   - Для логистической регрессии используются коэффициенты модели, которые показывают
     влияние каждого признака на логит вероятности положительного класса.

5.2. Permutation Importance (Важность на основе перестановок)
   - Измеряет насколько ухудшается качество модели при случайной перестановке
     значений каждого признака.
   - Работает для любых моделей и не зависит от внутренней структуры модели.

5.3. Partial Dependence Plots (Графики частичной зависимости)
   - Показывают зависимость предсказания модели от одного или двух признаков,
     усредняя влияние остальных признаков.
   - Позволяют понять, как изменение признака влияет на предсказание.

5.4. SHAP Values (Значения Шепли)
   - Основаны на теории кооперативных игр и показывают вклад каждого признака
     в предсказание для конкретного примера.
   - Summary Plot показывает общую картину важности признаков для всех примеров.
   - Waterfall Plot показывает пошаговое влияние признаков на конкретное предсказание.
   - Force Plot визуализирует как признаки "подталкивают" предсказание от базового
     значения к финальному предсказанию.

Все графики сохранены в директории 'plots/'.
""")

    def add_conclusions(self):
        """Добавить выводы"""
        self.add_header("6. ВЫВОДЫ", level=1)

        self.add_text("""
В ходе выполнения лабораторной работы были получены следующие результаты:

1. ОБУЧЕНИЕ МОДЕЛЕЙ
   - Обучены классические модели для бинарной классификации: Gradient Boosting,
     Random Forest и Logistic Regression.
   - Обучены модели для регрессии: Gradient Boosting и Random Forest.
   - Обучены нейронные сети для обеих задач.
   - Все модели показали приемлемое качество на тестовых данных.

2. ОБЪЯСНЕНИЯ МОДЕЛЕЙ
   - Получены значения важности признаков на основе критерия IG для моделей
     на основе деревьев.
   - Получены коэффициенты логистической регрессии, показывающие направление
     и силу влияния признаков.
   - Вычислены значения Permutation Importance для всех моделей, что позволило
     оценить реальное влияние признаков на качество предсказаний.
   - Построены графики частичной зависимости (PDP), которые показывают нелинейные
     зависимости между признаками и целевой переменной.
   - Вычислены значения Шепли (SHAP) для всех моделей, что дало возможность
     понять вклад каждого признака в индивидуальные предсказания.

3. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
   - Созданы Summary Plots для визуализации общей важности признаков.
   - Построены Waterfall Plots для демонстрации пошагового влияния признаков
     на конкретные предсказания.
   - Сгенерированы Force Plots для интерактивной визуализации объяснений.

4. ИНТЕРПРЕТАЦИЯ
   - Разные методы объяснений дали согласованные результаты относительно
     наиболее важных признаков.
   - Модели на основе деревьев показали хорошую интерпретируемость через
     Feature Importance.
   - SHAP values предоставили наиболее детальные объяснения, показав не только
     важность признаков, но и направление их влияния.
   - Графики PDP выявили нелинейные зависимости и пороговые эффекты в данных.

5. ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ
   - Полученные объяснения могут быть использованы для:
     * Валидации моделей и проверки их логики
     * Выявления потенциальных проблем и ошибок в данных
     * Принятия решений на основе понимания работы модели
     * Объяснения предсказаний заинтересованным сторонам

ОБЩИЙ ВЫВОД:
Использование различных методов объяснения моделей машинного обучения позволяет
получить глубокое понимание работы моделей, повысить доверие к их предсказаниям
и обеспечить прозрачность процесса принятия решений. Комбинация глобальных
объяснений (Feature Importance, Permutation Importance) и локальных объяснений
(SHAP values) дает наиболее полную картину работы модели.
""")

    def save_report(self, filename=None):
        """Сохранить отчет в файл"""
        if filename is None:
            filename = os.path.join(self.config.OUTPUT_DIR, self.config.REPORT_FILE)

        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))

        print(f"\n{'=' * 80}")
        print(f"Отчет сохранен в файл: {filename}")
        print(f"{'=' * 80}\n")

        return filename

    def save_metrics_csv(self, classification_results, regression_results):
        """Сохранить метрики в CSV файл"""
        filename = os.path.join(self.config.OUTPUT_DIR, self.config.METRICS_FILE)

        # Объединение всех метрик
        all_metrics = []

        for model_name, metrics in classification_results.items():
            all_metrics.append({
                'Задача': 'Классификация',
                'Модель': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc'],
                'R²': None,
                'RMSE': None,
                'MAE': None
            })

        for model_name, metrics in regression_results.items():
            all_metrics.append({
                'Задача': 'Регрессия',
                'Модель': model_name,
                'Accuracy': None,
                'Precision': None,
                'Recall': None,
                'F1-Score': None,
                'ROC-AUC': None,
                'R²': metrics['r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae']
            })

        df = pd.DataFrame(all_metrics)
        df.to_csv(filename, index=False, encoding='utf-8')

        print(f"Метрики сохранены в CSV файл: {filename}")

        return filename
