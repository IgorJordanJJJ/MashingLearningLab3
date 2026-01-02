"""
Модуль для получения объяснений моделей машинного обучения
Использует SHAP, Partial Dependence Plots и Permutation Importance
"""

import os
# Установка backend для matplotlib ДО всех импортов
os.environ['MPLBACKEND'] = 'Agg'  # Установка через переменную окружения
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Дополнительная установка для надежности
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from config import Config


class ModelExplainer:
    """
    Класс для получения и визуализации объяснений моделей
    """

    def __init__(self, model, model_name, X_train, X_test, y_test, feature_names, task_type='classification'):
        """
        Args:
            model: Обученная модель
            model_name: Название модели
            X_train: Тренировочные данные (для SHAP)
            X_test: Тестовые данные
            y_test: Тестовые метки
            feature_names: Список имен признаков
            task_type: 'classification' или 'regression'
        """
        self.config = Config()
        self.model = model
        self.model_name = model_name
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.task_type = task_type
        self.shap_values = None
        self.explainer = None

    # ==================== FEATURE IMPORTANCE ====================

    def get_feature_importance(self):
        """
        Получить важность признаков на основе критерия IG (для моделей на основе деревьев)
        или коэффициенты регрессии (для логистической регрессии)
        """
        print(f"\n{'=' * 80}")
        print(f"Получение важности признаков для модели {self.model_name}")
        print(f"{'=' * 80}")

        if hasattr(self.model, 'feature_importances_'):
            # Модели на основе деревьев (GradientBoosting, RandomForest)
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print("\nВажность признаков (Feature Importances):")
            print(feature_importance_df.to_string(index=False))

            # Визуализация
            self._plot_feature_importance(feature_importance_df)

            return feature_importance_df

        elif hasattr(self.model, 'coef_'):
            # Логистическая регрессия
            coefficients = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)

            print("\nКоэффициенты логистической регрессии:")
            print(feature_importance_df.to_string(index=False))

            # Визуализация
            self._plot_logistic_coefficients(feature_importance_df)

            return feature_importance_df

        else:
            print("Модель не поддерживает прямое получение важности признаков")
            return None

    def _plot_feature_importance(self, feature_importance_df):
        """Визуализация важности признаков для моделей на основе деревьев"""
        fig = plt.figure(figsize=self.config.FIGURE_SIZE)
        try:
            top_features = feature_importance_df.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Важность признака', fontsize=12)
            plt.ylabel('Признак', fontsize=12)
            plt.title(f'Feature Importance - {self.model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()

            filename = os.path.join(self.config.PLOTS_DIR, f'{self.task_type}_{self.model_name}_feature_importance.png')
            plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        finally:
            plt.close(fig)
            plt.close('all')
        print(f"График важности признаков сохранен: {filename}")

    def _plot_logistic_coefficients(self, feature_importance_df):
        """Визуализация коэффициентов логистической регрессии"""
        fig = plt.figure(figsize=self.config.FIGURE_SIZE)
        try:
            top_features = feature_importance_df.head(10)
            colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]

            plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Коэффициент', fontsize=12)
            plt.ylabel('Признак', fontsize=12)
            plt.title(f'Логистическая регрессия - Коэффициенты - {self.model_name}', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
            plt.gca().invert_yaxis()
            plt.tight_layout()

            filename = os.path.join(self.config.PLOTS_DIR, f'{self.task_type}_{self.model_name}_coefficients.png')
            plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        finally:
            plt.close(fig)
            plt.close('all')
        print(f"График коэффициентов сохранен: {filename}")

    # ==================== PERMUTATION IMPORTANCE ====================

    def get_permutation_importance(self):
        """Получить важность признаков на основе перестановок"""
        print(f"\n{'=' * 80}")
        print(f"Вычисление Permutation Importance для модели {self.model_name}")
        print(f"{'=' * 80}")

        # Сохраняем DataFrame для сохранения имен признаков
        # Если это DataFrame, используем его напрямую, иначе создаем DataFrame
        if hasattr(self.X_test, 'columns'):
            X_test_for_perm = self.X_test
        else:
            # Если это numpy array, создаем DataFrame с именами признаков
            X_test_for_perm = pd.DataFrame(self.X_test, columns=self.feature_names)
        
        y_test_array = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test

        # Для нейронных сетей нужно указать функцию оценки явно
        # Проверяем, является ли модель Keras Sequential
        is_keras_model = (self.model_name == 'NeuralNetwork' or 
                         (hasattr(self.model, 'predict') and 
                          hasattr(self.model, 'compile') and 
                          not hasattr(self.model, 'score')))
        
        # Определяем параметры для permutation_importance
        perm_params = self.config.PERMUTATION_PARAMS.copy()
        
        if is_keras_model:
            # Для нейронных сетей создаем обертку и функцию оценки
            from sklearn.metrics import accuracy_score, r2_score, make_scorer
            
            class KerasModelWrapper:
                """Обертка для Keras модели для работы с sklearn"""
                def __init__(self, model, task_type):
                    self.model = model
                    self.task_type = task_type
                
                def fit(self, X, y):
                    """Метод fit для совместимости со sklearn (модель уже обучена)"""
                    # Модель уже обучена, просто возвращаем self
                    return self
                
                def predict(self, X):
                    """Предсказания для Keras модели"""
                    # Преобразуем DataFrame в numpy array если нужно
                    if hasattr(X, 'values'):
                        X_array = X.values
                    else:
                        X_array = X
                    
                    predictions = self.model.predict(X_array, verbose=0)
                    
                    if self.task_type == 'classification':
                        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                            # Бинарная классификация
                            return (predictions.flatten() > 0.5).astype(int)
                        else:
                            # Многоклассовая классификация
                            return np.argmax(predictions, axis=1)
                    else:
                        # Регрессия
                        return predictions.flatten()
            
            # Создаем обертку
            model_for_perm = KerasModelWrapper(self.model, self.task_type)
            
            # Устанавливаем функцию оценки в зависимости от типа задачи
            if self.task_type == 'classification':
                perm_params['scoring'] = make_scorer(accuracy_score)
            else:
                perm_params['scoring'] = make_scorer(r2_score)
        else:
            # Для sklearn моделей используем модель напрямую
            model_for_perm = self.model

        perm_importance = permutation_importance(
            model_for_perm,
            X_test_for_perm,
            y_test_array,
            **perm_params
        )

        perm_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        print("\nPermutation Importance:")
        print(perm_importance_df.to_string(index=False))

        # Визуализация
        self._plot_permutation_importance(perm_importance_df)

        return perm_importance_df

    def _plot_permutation_importance(self, perm_importance_df):
        """Визуализация Permutation Importance"""
        fig = plt.figure(figsize=self.config.FIGURE_SIZE)
        try:
            top_features = perm_importance_df.head(10)

            plt.barh(range(len(top_features)), top_features['importance_mean'],
                    xerr=top_features['importance_std'], capsize=5)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Permutation Importance', fontsize=12)
            plt.ylabel('Признак', fontsize=12)
            plt.title(f'Permutation Importance - {self.model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()

            filename = os.path.join(self.config.PLOTS_DIR, f'{self.task_type}_{self.model_name}_permutation_importance.png')
            plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        finally:
            plt.close(fig)
            plt.close('all')
        print(f"График Permutation Importance сохранен: {filename}")

    # ==================== PARTIAL DEPENDENCE PLOTS ====================

    def get_partial_dependence(self, features=None):
        """Получить графики частичной зависимости (PDP)"""
        print(f"\n{'=' * 80}")
        print(f"Построение Partial Dependence Plots для модели {self.model_name}")
        print(f"{'=' * 80}")

        # Если признаки не указаны, выбираем топ-6
        if features is None:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                top_indices = np.argsort(importances)[-6:][::-1]
                features = top_indices.tolist()
            else:
                # Для логистической регрессии и нейросетей берем первые 6
                features = list(range(min(6, len(self.feature_names))))

        # Сохраняем DataFrame для сохранения имен признаков
        # Если это DataFrame, используем его напрямую, иначе создаем DataFrame
        if hasattr(self.X_test, 'columns'):
            X_test_for_pdp = self.X_test
        else:
            # Если это numpy array, создаем DataFrame с именами признаков
            X_test_for_pdp = pd.DataFrame(self.X_test, columns=self.feature_names)

        try:
            # Построение PDP
            fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE_LARGE)

            display = PartialDependenceDisplay.from_estimator(
                self.model,
                X_test_for_pdp,
                features,
                feature_names=self.feature_names,
                grid_resolution=self.config.PDP_PARAMS['grid_resolution'],
                n_jobs=self.config.PDP_PARAMS['n_jobs'],
                ax=ax
            )

            plt.suptitle(f'Partial Dependence Plots - {self.model_name}',
                        fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout()

            filename = os.path.join(self.config.PLOTS_DIR, f'{self.task_type}_{self.model_name}_pdp.png')
            plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
            plt.close(fig)  # Явное закрытие фигуры
            plt.close('all')  # Закрытие всех фигур для надежности
            print(f"Графики PDP сохранены: {filename}")

        except Exception as e:
            print(f"Ошибка при построении PDP: {str(e)}")
            plt.close('all')  # Закрытие всех фигур в случае ошибки

    # ==================== SHAP VALUES ====================

    def calculate_shap_values(self, max_samples=100):
        """Вычислить значения SHAP"""
        print(f"\n{'=' * 80}")
        print(f"Вычисление SHAP values для модели {self.model_name}")
        print(f"{'=' * 80}")

        # Подготовка данных
        X_train_sample = self.X_train[:max_samples] if len(self.X_train) > max_samples else self.X_train
        X_test_sample = self.X_test[:max_samples] if len(self.X_test) > max_samples else self.X_test

        # Для SHAP можно использовать numpy array, но для TreeExplainer лучше DataFrame
        # Преобразуем в numpy array только если это необходимо
        if hasattr(X_train_sample, 'values'):
            X_train_array = X_train_sample.values
            X_test_array = X_test_sample.values
        else:
            X_train_array = X_train_sample
            X_test_array = X_test_sample

        try:
            # Выбор подходящего explainer
            if self.model_name == 'NeuralNetwork':
                # Для нейронных сетей используем DeepExplainer или KernelExplainer
                print("Используется KernelExplainer для нейронной сети...")
                self.explainer = shap.KernelExplainer(self.model.predict, X_train_array)
                self.shap_values = self.explainer.shap_values(X_test_array)
            elif hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_'):
                # Для моделей на основе деревьев (RandomForest, GradientBoosting)
                print("Используется TreeExplainer...")
                # TreeExplainer может работать с DataFrame, попробуем сначала с DataFrame
                if hasattr(X_train_sample, 'columns'):
                    try:
                        self.explainer = shap.TreeExplainer(self.model)
                        self.shap_values = self.explainer.shap_values(X_test_sample)
                    except:
                        # Если не работает с DataFrame, используем numpy array
                        self.explainer = shap.TreeExplainer(self.model)
                        self.shap_values = self.explainer.shap_values(X_test_array)
                else:
                    self.explainer = shap.TreeExplainer(self.model)
                    self.shap_values = self.explainer.shap_values(X_test_array)
            else:
                # Для линейных моделей
                print("Используется LinearExplainer...")
                try:
                    self.explainer = shap.LinearExplainer(self.model, X_train_array)
                    self.shap_values = self.explainer.shap_values(X_test_array)
                except:
                    print("LinearExplainer не сработал, используется KernelExplainer...")
                    self.explainer = shap.KernelExplainer(self.model.predict, X_train_array)
                    self.shap_values = self.explainer.shap_values(X_test_array)

            # Для бинарной классификации может вернуться список из двух массивов
            # или массив с формой (n_samples, n_features, 2)
            if isinstance(self.shap_values, list):
                # Если это список, берем значения для положительного класса (индекс 1)
                self.shap_values = self.shap_values[1]
            elif isinstance(self.shap_values, np.ndarray) and len(self.shap_values.shape) == 3:
                # Если это массив формы (n_samples, n_features, 2), берем значения для положительного класса
                # Для бинарной классификации обычно используем класс 1 (положительный)
                self.shap_values = self.shap_values[:, :, 1]

            print(f"SHAP values вычислены. Размер: {self.shap_values.shape}")
            return self.shap_values

        except Exception as e:
            print(f"Ошибка при вычислении SHAP values: {str(e)}")
            return None

    def plot_shap_summary(self):
        """Построить Summary Plot для SHAP values"""
        if self.shap_values is None:
            print("SHAP values еще не вычислены. Вызовите calculate_shap_values() сначала.")
            return

        print(f"\nПостроение SHAP Summary Plot для модели {self.model_name}")

        X_test_sample = self.X_test[:len(self.shap_values)] if len(self.X_test) > len(self.shap_values) else self.X_test
        # Для SHAP summary plot можно использовать numpy array
        X_test_array = X_test_sample.values if hasattr(X_test_sample, 'values') else X_test_sample

        fig = plt.figure(figsize=self.config.FIGURE_SIZE)
        try:
            shap.summary_plot(
                self.shap_values,
                X_test_array,
                feature_names=self.feature_names,
                show=False,
                max_display=self.config.SHAP_PARAMS['max_display']
            )
            plt.title(f'SHAP Summary Plot - {self.model_name}', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()

            filename = os.path.join(self.config.PLOTS_DIR, f'{self.task_type}_{self.model_name}_shap_summary.png')
            plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        finally:
            plt.close(fig)  # Явное закрытие фигуры
            plt.close('all')  # Закрытие всех фигур для надежности
        print(f"SHAP Summary Plot сохранен: {filename}")

    def plot_shap_waterfall(self, instance_index=0):
        """Построить Waterfall Plot для индивидуального примера"""
        if self.shap_values is None:
            print("SHAP values еще не вычислены. Вызовите calculate_shap_values() сначала.")
            return

        print(f"\nПостроение SHAP Waterfall Plot для примера {instance_index}")

        X_test_sample = self.X_test[:len(self.shap_values)] if len(self.X_test) > len(self.shap_values) else self.X_test

        # Правильное получение строки из DataFrame или numpy array
        if hasattr(X_test_sample, 'iloc'):
            # Это pandas DataFrame
            instance_data = X_test_sample.iloc[instance_index].values
        elif hasattr(X_test_sample, 'values'):
            # Это pandas DataFrame (альтернативная проверка)
            instance_data = X_test_sample.values[instance_index]
        else:
            # Это numpy array
            instance_data = X_test_sample[instance_index]

        # Получаем SHAP values для конкретного примера
        shap_values_instance = self.shap_values[instance_index]
        
        # Проверяем, что это одномерный массив
        if len(shap_values_instance.shape) > 1:
            # Если это многомерный массив, берем первый срез
            shap_values_instance = shap_values_instance.flatten()[:len(self.feature_names)]
        
        # Получаем base value
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, np.ndarray):
                # Для бинарной классификации берем expected_value для положительного класса
                base_value = self.explainer.expected_value[1] if len(self.explainer.expected_value) > 1 else self.explainer.expected_value[0]
            else:
                base_value = self.explainer.expected_value
        else:
            base_value = 0

        fig = plt.figure(figsize=self.config.FIGURE_SIZE)
        try:
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_instance,
                    base_values=base_value,
                    data=instance_data,
                    feature_names=self.feature_names
                ),
                show=False,
                max_display=self.config.SHAP_PARAMS['max_display']
            )
            plt.title(f'SHAP Waterfall Plot - {self.model_name} (Пример {instance_index})',
                     fontsize=14, fontweight='bold')
            plt.tight_layout()

            filename = os.path.join(self.config.PLOTS_DIR,
                                   f'{self.task_type}_{self.model_name}_shap_waterfall_{instance_index}.png')
            plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        finally:
            plt.close(fig)  # Явное закрытие фигуры
            plt.close('all')  # Закрытие всех фигур для надежности
        print(f"SHAP Waterfall Plot сохранен: {filename}")

    def plot_shap_force(self, instance_index=0):
        """Построить Force Plot для индивидуального примера"""
        if self.shap_values is None:
            print("SHAP values еще не вычислены. Вызовите calculate_shap_values() сначала.")
            return

        print(f"\nПостроение SHAP Force Plot для примера {instance_index}")

        X_test_sample = self.X_test[:len(self.shap_values)] if len(self.X_test) > len(self.shap_values) else self.X_test

        # Правильное получение строки из DataFrame или numpy array
        if hasattr(X_test_sample, 'iloc'):
            # Это pandas DataFrame
            instance_data = X_test_sample.iloc[instance_index].values
        elif hasattr(X_test_sample, 'values'):
            # Это pandas DataFrame (альтернативная проверка)
            instance_data = X_test_sample.values[instance_index]
        else:
            # Это numpy array
            instance_data = X_test_sample[instance_index]

        # Получаем SHAP values для конкретного примера
        shap_values_instance = self.shap_values[instance_index]
        
        # Проверяем, что это одномерный массив
        if len(shap_values_instance.shape) > 1:
            # Если это многомерный массив, берем первый срез
            shap_values_instance = shap_values_instance.flatten()[:len(self.feature_names)]
        
        # Получаем base value
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, np.ndarray):
                # Для бинарной классификации берем expected_value для положительного класса
                base_value = self.explainer.expected_value[1] if len(self.explainer.expected_value) > 1 else self.explainer.expected_value[0]
            else:
                base_value = self.explainer.expected_value
        else:
            base_value = 0

        try:
            # Force plot требует JavaScript визуализации, поэтому сохраняем как HTML
            force_plot = shap.force_plot(
                base_value,
                shap_values_instance,
                instance_data,
                feature_names=self.feature_names
            )

            filename = os.path.join(self.config.PLOTS_DIR,
                                   f'{self.task_type}_{self.model_name}_shap_force_{instance_index}.html')
            shap.save_html(filename, force_plot)
            print(f"SHAP Force Plot сохранен: {filename}")

        except Exception as e:
            print(f"Ошибка при построении Force Plot: {str(e)}")

    def generate_all_explanations(self):
        """Генерация всех объяснений для модели"""
        print(f"\n{'=' * 80}")
        print(f"ГЕНЕРАЦИЯ ВСЕХ ОБЪЯСНЕНИЙ ДЛЯ МОДЕЛИ {self.model_name}")
        print(f"{'=' * 80}")

        # 1. Feature Importance
        self.get_feature_importance()

        # 2. Permutation Importance
        self.get_permutation_importance()

        # 3. Partial Dependence Plots
        self.get_partial_dependence()

        # 4. SHAP Values
        self.calculate_shap_values(max_samples=self.config.SHAP_PARAMS['sample_size'])

        if self.shap_values is not None:
            # 5. SHAP Summary Plot
            self.plot_shap_summary()

            # 6. SHAP Waterfall и Force для нескольких примеров
            n_examples = min(self.config.N_INDIVIDUAL_EXAMPLES, len(self.shap_values))
            for i in range(n_examples):
                self.plot_shap_waterfall(i)
                # Force plot пропускаем для нейросетей из-за возможных проблем
                if self.model_name != 'NeuralNetwork':
                    self.plot_shap_force(i)

        print(f"\n✓ Все объяснения для модели {self.model_name} сгенерированы!")
