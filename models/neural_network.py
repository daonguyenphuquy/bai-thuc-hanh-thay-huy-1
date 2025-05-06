import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

# Thiết lập seed cho TensorFlow
tf.random.set_seed(42)


class KerasRegressor(BaseEstimator, RegressorMixin):
    """Wrapper cho mô hình Keras để tương thích với scikit-learn"""

    def __init__(self, build_fn=None, epochs=100, batch_size=32, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None

    def fit(self, X, y):
        if self.build_fn is None:
            self.model_ = self._default_model(X.shape[1])
        else:
            self.model_ = self.build_fn(X.shape[1])

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        return self

    def predict(self, X):
        return self.model_.predict(X, verbose=0).flatten()

    def _default_model(self, input_dim):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model


def create_nn_model(input_dim):
    """Tạo mô hình Neural Network"""
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_nn_model(X_train, y_train, preprocessor):
    """
    Huấn luyện mô hình Neural Network
    """
    # Tiền xử lý dữ liệu
    X_train_processed = preprocessor.fit_transform(X_train)

    # Tạo mô hình
    keras_model = KerasRegressor(
        build_fn=lambda: create_nn_model(X_train_processed.shape[1]),
        epochs=200,
        batch_size=32,
        verbose=0
    )

    # Chuẩn hóa biến mục tiêu
    target_transformer = TransformedTargetRegressor(
        regressor=keras_model,
        transformer=StandardScaler()
    )
    target_transformer.fit(X_train_processed, y_train)

    # Tạo pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', target_transformer)
    ])

    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)

    return keras_model, pipeline