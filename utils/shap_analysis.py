import  shap
import numpy as np
import matplotlib.pyplot as plt


def analyze_shap_values(gb_pipeline, X_test, feature_names):
    """
    Phân tích SHAP values cho mô hình Gradient Boosting

    Parameters:
    -----------
    gb_pipeline : Pipeline
        Pipeline của mô hình Gradient Boosting
    X_test : DataFrame
        Dữ liệu test
    feature_names : list
        Danh sách tên các đặc trưng
    """
    # Tiền xử lý dữ liệu test
    X_test_processed = gb_pipeline.named_steps['preprocessor'].transform(X_test)

    # Lấy mô hình Gradient Boosting từ pipeline
    gb_model = gb_pipeline.named_steps['regressor']

    # Tạo explainer
    explainer = shap.TreeExplainer(gb_model)

    # Tính SHAP values
    shap_values = explainer.shap_values(X_test_processed)

    # Lấy tên các đặc trưng sau khi xử lý
    processed_feature_names = []

    # Lấy tên các đặc trưng
    if hasattr(gb_pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
        processed_feature_names = gb_pipeline.named_steps['preprocessor'].get_feature_names_out()
    else:
        # Fallback cho phiên bản sklearn cũ hơn
        processed_feature_names = [f"feature_{i}" for i in range(X_test_processed.shape[1])]

    # Vẽ SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_processed, feature_names=processed_feature_names, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()

    # Vẽ SHAP bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_processed, feature_names=processed_feature_names, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('shap_importance.png')
    plt.close()

    # Vẽ biểu đồ phụ thuộc SHAP cho 3 đặc trưng quan trọng nhất
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[-3:]

    for idx in top_indices:
        plt.figure(figsize=(10, 7))
        shap.dependence_plot(idx, shap_values, X_test_processed, feature_names=processed_feature_names, show=False)
        plt.title(f'SHAP Dependence Plot: {processed_feature_names[idx]}')
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{processed_feature_names[idx]}.png')
        plt.close()

    print("Đã hoàn thành phân tích SHAP.")