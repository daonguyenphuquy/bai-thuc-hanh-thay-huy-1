import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def remove_outliers(df, contamination=0.05):
    """
    Phát hiện và xử lý outliers bằng phương pháp Isolation Forest
    """
    # Sao chép DataFrame để tránh thay đổi dữ liệu gốc
    df_copy = df.copy()

    # Lấy các cột số để phát hiện outliers
    numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns

    # Áp dụng Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_pred = iso_forest.fit_predict(df_copy[numeric_cols])

    # Nhãn -1 đại diện cho outliers, 1 đại diện cho inliers
    outliers = np.where(outlier_pred == -1)[0]

    print(f"Đã phát hiện {len(outliers)} outliers ({len(outliers) / len(df_copy) * 100:.2f}% dữ liệu)")

    # Lọc outliers
    df_cleaned = df_copy.drop(outliers)

    return df_cleaned