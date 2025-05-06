import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Tính Mean Absolute Percentage Error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_models(models_dict, X_train, y_train, X_test, y_test):
    """
    Đánh giá và so sánh các mô hình

    Parameters:
    -----------
    models_dict : dict
        Dictionary chứa các mô hình cần đánh giá
    X_train, y_train : array-like
        Dữ liệu huấn luyện
    X_test, y_test : array-like
        Dữ liệu kiểm tra

    Returns:
    --------
    best_model : object
        Mô hình có hiệu suất tốt nhất
    best_model_name : str
        Tên của mô hình tốt nhất
    """
    results = pd.DataFrame(columns=['Model', 'RMSE', 'R2', 'MAE', 'MAPE'])

    for model_name, model_pipeline in models_dict.items():
        # Dự đoán trên tập test
        y_pred = model_pipeline.predict(X_test)

        # Tính các chỉ số đánh giá
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except:
            mape = np.nan

        # Cross-validation
        cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())

        # Thêm kết quả vào DataFrame
        results = pd.concat([results, pd.DataFrame({
            'Model': [model_name],
            'RMSE': [rmse],
            'CV RMSE': [cv_rmse],
            'R2': [r2],
            'MAE': [mae],
            'MAPE': [mape]
        })], ignore_index=True)

        print(f"\n{model_name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R^2: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.4f}%")
        print(f"  5-fold CV RMSE: {cv_rmse:.4f}")

        # Vẽ residual plot
        if model_name == 'Gradient Boosting':
            plot_residuals(y_test, y_pred, model_name)

    # Sắp xếp kết quả theo RMSE tăng dần
    results = results.sort_values('RMSE')

    # Lấy mô hình tốt nhất
    best_model_name = results.iloc[0]['Model']
    best_model = models_dict[best_model_name]

    # In kết quả
    print("\n=== So sánh các mô hình ===")
    print(results)

    return best_model, best_model_name


def plot_residuals(y_true, y_pred, model_name):
    """
    Vẽ biểu đồ phân tích sai số
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='r', linestyles='--')
    plt.title(f'Residual Plot - {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)

    # Thêm đường xu hướng
    z = np.polyfit(y_pred, residuals, 1)
    p = np.poly1d(z)
    plt.plot(y_pred, p(y_pred), "b--", alpha=0.8)

    plt.tight_layout()
    plt.savefig(f'residual_plot_{model_name.replace(" ", "_")}.png')
    plt.close()