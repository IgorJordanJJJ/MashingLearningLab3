"""
Модуль для обучения моделей машинного обучения
Включает классы для обучения классификаторов и регрессоров
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from config import Config


class ClassificationTrainer:
    """
    Класс для обучения моделей бинарной классификации
    """

    def __init__(self):
        self.config = Config()
        self.models = {}
        self.results = {}

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Обучить модель Gradient Boosting для классификации"""
        print("\n" + "=" * 80)
        print("Обучение Gradient Boosting Classifier")
        print("=" * 80)

        params = self.config.CLASSIFICATION_MODELS['GradientBoosting']
        print(f"Параметры модели: {params}")

        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Метрики
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        self.models['GradientBoosting'] = model
        self.results['GradientBoosting'] = metrics

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")

        return model, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Обучить модель Random Forest для классификации"""
        print("\n" + "=" * 80)
        print("Обучение Random Forest Classifier")
        print("=" * 80)

        params = self.config.CLASSIFICATION_MODELS['RandomForest']
        print(f"Параметры модели: {params}")

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Метрики
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        self.models['RandomForest'] = model
        self.results['RandomForest'] = metrics

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")

        return model, metrics

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Обучить модель Logistic Regression для классификации"""
        print("\n" + "=" * 80)
        print("Обучение Logistic Regression")
        print("=" * 80)

        params = self.config.CLASSIFICATION_MODELS['LogisticRegression']
        print(f"Параметры модели: {params}")

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Метрики
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        self.models['LogisticRegression'] = model
        self.results['LogisticRegression'] = metrics

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")

        return model, metrics

    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Обучить нейронную сеть для классификации"""
        print("\n" + "=" * 80)
        print("Обучение Neural Network Classifier")
        print("=" * 80)

        params = self.config.NEURAL_NETWORK_PARAMS['classifier']
        print(f"Параметры модели: {params}")

        # Создание модели
        model = Sequential()
        model.add(Dense(params['hidden_layers'][0], activation=params['activation'], input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.3))

        for units in params['hidden_layers'][1:]:
            model.add(Dense(units, activation=params['activation']))
            model.add(Dropout(0.2))

        model.add(Dense(1, activation='sigmoid'))

        # Компиляция
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        print("\nАрхитектура модели:")
        model.summary()

        # Обучение
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=params['early_stopping_patience'],
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_split=params['validation_split'],
            callbacks=[early_stopping],
            verbose=0
        )

        # Предсказания
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Метрики
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['history'] = history.history

        self.models['NeuralNetwork'] = model
        self.results['NeuralNetwork'] = metrics

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")

        return model, metrics

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Вычислить метрики качества классификации"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }

    def save_models(self):
        """Сохранить обученные модели"""
        if not self.config.SAVE_MODELS:
            return

        models_dir = self.config.MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)

        for model_name, model in self.models.items():
            if model_name == 'NeuralNetwork':
                # Сохранение нейронной сети
                model_path = os.path.join(models_dir, f'{model_name}_classifier.h5')
                model.save(model_path)
            else:
                # Сохранение классических моделей
                model_path = os.path.join(models_dir, f'{model_name}_classifier.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            print(f"Модель {model_name} сохранена в {model_path}")


class RegressionTrainer:
    """
    Класс для обучения моделей регрессии
    """

    def __init__(self):
        self.config = Config()
        self.models = {}
        self.results = {}

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Обучить модель Gradient Boosting для регрессии"""
        print("\n" + "=" * 80)
        print("Обучение Gradient Boosting Regressor")
        print("=" * 80)

        params = self.config.REGRESSION_MODELS['GradientBoosting']
        print(f"Параметры модели: {params}")

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)

        # Метрики
        metrics = self._calculate_metrics(y_test, y_pred)

        self.models['GradientBoosting'] = model
        self.results['GradientBoosting'] = metrics

        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")

        return model, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Обучить модель Random Forest для регрессии"""
        print("\n" + "=" * 80)
        print("Обучение Random Forest Regressor")
        print("=" * 80)

        params = self.config.REGRESSION_MODELS['RandomForest']
        print(f"Параметры модели: {params}")

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)

        # Метрики
        metrics = self._calculate_metrics(y_test, y_pred)

        self.models['RandomForest'] = model
        self.results['RandomForest'] = metrics

        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")

        return model, metrics

    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Обучить нейронную сеть для регрессии"""
        print("\n" + "=" * 80)
        print("Обучение Neural Network Regressor")
        print("=" * 80)

        params = self.config.NEURAL_NETWORK_PARAMS['regressor']
        print(f"Параметры модели: {params}")

        # Создание модели
        model = Sequential()
        model.add(Dense(params['hidden_layers'][0], activation=params['activation'], input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.3))

        for units in params['hidden_layers'][1:]:
            model.add(Dense(units, activation=params['activation']))
            model.add(Dropout(0.2))

        model.add(Dense(1))  # Выходной слой для регрессии

        # Компиляция
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        print("\nАрхитектура модели:")
        model.summary()

        # Обучение
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=params['early_stopping_patience'],
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_split=params['validation_split'],
            callbacks=[early_stopping],
            verbose=0
        )

        # Предсказания
        y_pred = model.predict(X_test).flatten()

        # Метрики
        metrics = self._calculate_metrics(y_test, y_pred)
        metrics['history'] = history.history

        self.models['NeuralNetwork'] = model
        self.results['NeuralNetwork'] = metrics

        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")

        return model, metrics

    def _calculate_metrics(self, y_true, y_pred):
        """Вычислить метрики качества регрессии"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def save_models(self):
        """Сохранить обученные модели"""
        if not self.config.SAVE_MODELS:
            return

        models_dir = self.config.MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)

        for model_name, model in self.models.items():
            if model_name == 'NeuralNetwork':
                # Сохранение нейронной сети
                model_path = os.path.join(models_dir, f'{model_name}_regressor.h5')
                model.save(model_path)
            else:
                # Сохранение классических моделей
                model_path = os.path.join(models_dir, f'{model_name}_regressor.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            print(f"Модель {model_name} сохранена в {model_path}")
