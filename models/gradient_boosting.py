import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler


def objective(trial, X_train, y_train, preprocessor):
    """
    Hàm mục tiêu cho Bayesian Optimization với Optuna
    """
    # Định nghĩa không gian tìm kiếm siêu tham số
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
    }

    # Tạo mô hình XGBoost với các siêu tham số
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **params
    )

    # Tạo pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Đánh giá bằng cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-cv_scores.mean())

    return rmse


def optimize_and_train_gb(X_train, y_train, preprocessor, n_trials=50):
    """
    Tối ưu hóa siêu tham số và huấn luyện mô hình Gradient Boosting
    """
    print("Tối ưu hóa siêu tham số với Optuna...")

    # Tạo Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    # Tối ưu hóa
    study.optimize(lambda trial: objective(trial, X_train, y_train, preprocessor), n_trials=n_trials)

    # Lấy siêu tham số tốt nhất
    best_params = study.best_params
    print("Siêu tham số tốt nhất:", best_params)

    # Huấn luyện mô hình với siêu tham số tốt nhất
    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params
    )

    # Tạo pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', best_model)
    ])

    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)

    # Đánh giá bằng cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)

    print(f"Cross-validation RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")

    return best_model, pipeline