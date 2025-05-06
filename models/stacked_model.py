from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np


def create_stacked_model(X_train, y_train, preprocessor, base_models):
    """
    Tạo và huấn luyện mô hình Stacking

    Parameters:
    -----------
    X_train : DataFrame
        Dữ liệu huấn luyện đầu vào
    y_train : Series
        Biến mục tiêu
    preprocessor : ColumnTransformer
        Bộ tiền xử lý dữ liệu
    base_models : list of tuples
        Danh sách các mô hình cơ sở dưới dạng (name, model)
    """
    # Tạo meta-learner
    meta_learner = LinearRegression()

    # Tạo mô hình Stacking
    stacked_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )

    # Tạo pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', stacked_model)
    ])

    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)

    # Đánh giá bằng cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)

    print(f"Cross-validation RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")

    return stacked_model, pipeline