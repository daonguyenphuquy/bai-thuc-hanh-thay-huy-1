import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing  # Thay đổi ở đây
import warnings

warnings.filterwarnings('ignore')

from preprocessing.data_preparation import load_and_prepare_data
from preprocessing.feature_engineering import engineer_features
from preprocessing.outlier_detection import remove_outliers
from models.linear_model import train_linear_model
from models.gradient_boosting import optimize_and_train_gb
from models.neural_network import train_nn_model
from models.stacked_model import create_stacked_model
from utils.evaluation import evaluate_models
from utils.visualization import create_visualizations
from utils.shap_analysis import analyze_shap_values

def main():
    print("=== Dự đoán giá nhà California Housing ===")

    # Bước 1: Tải dữ liệu và chuẩn bị
    try:
        data = fetch_california_housing()  # Tải California Housing
        X, y = data.data, data.target
        feature_names = data.feature_names
        df = pd.DataFrame(X, columns=feature_names)
        df['PRICE'] = y  # Giữ tên cột PRICE để tương thích với mã hiện tại
    except Exception as e:
        print(f"Không thể tải dữ liệu: {e}")
        return

    print(f"Dữ liệu đã tải: {df.shape[0]} mẫu với {df.shape[1]} đặc trưng")

    # Bước 2: Phân tích và trực quan hóa dữ liệu
    create_visualizations(df, 'PRICE')

    # Bước 3: Xử lý outlier
    df_cleaned = remove_outliers(df)

    # Bước 4: Feature Engineering
    df_engineered = engineer_features(df_cleaned)

    # Bước 5: Chuẩn bị dữ liệu cho mô hình
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data(df_engineered, target_col='PRICE')

    # Bước 6: Huấn luyện và đánh giá các mô hình
    print("\n--- Huấn luyện các mô hình ---")

    # Linear Regression
    print("\nMô hình Linear Regression:")
    lr_model, lr_pipeline = train_linear_model(X_train, y_train, preprocessor)

    # Gradient Boosting
    print("\nMô hình Gradient Boosting:")
    gb_model, gb_pipeline = optimize_and_train_gb(X_train, y_train, preprocessor)

    # Neural Network
    print("\nMô hình Neural Network:")
    nn_model, nn_pipeline = train_nn_model(X_train, y_train, preprocessor)

    # Stacking
    print("\nMô hình Stacking:")
    stacked_model, stacked_pipeline = create_stacked_model(
        X_train, y_train,
        preprocessor=preprocessor,
        base_models=[('lr', lr_model), ('gb', gb_model), ('nn', nn_model)]
    )

    # Bước 7: Đánh giá và so sánh các mô hình
    models_dict = {
        'Linear Regression': lr_pipeline,
        'Gradient Boosting': gb_pipeline,
        'Neural Network': nn_pipeline,
        'Stacked Model': stacked_pipeline
    }

    best_model, best_model_name = evaluate_models(models_dict, X_train, y_train, X_test, y_test)

    # Bước 8: SHAP analysis trên mô hình Gradient Boosting
    analyze_shap_values(gb_pipeline, X_test, feature_names)

    print(f"\nKết luận: Mô hình tốt nhất là {best_model_name}")

if __name__ == "__main__":
    main()