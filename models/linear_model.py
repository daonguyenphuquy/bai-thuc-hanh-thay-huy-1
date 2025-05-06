from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np


def train_linear_model(X_train, y_train, preprocessor):
    """
    Huấn luyện mô hình Linear Regression
    """
    # Tạo mô hình
    lr = LinearRegression()

    # Tạo pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', lr)
    ])

    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)

    # Đánh giá bằng cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)

    print(f"Cross-validation RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")

    return lr, pipeline