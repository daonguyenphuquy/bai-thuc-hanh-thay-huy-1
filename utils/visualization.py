import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def create_visualizations(df, target_col):
    """
    Tạo các biểu đồ trực quan hóa

    Parameters:
    -----------
    df : DataFrame
        DataFrame chứa dữ liệu
    target_col : str
        Tên cột mục tiêu
    """
    # Thiết lập style
    sns.set(style="whitegrid")

    # 1. Phân phối của biến mục tiêu
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col], kde=True)
    plt.title(f'Phân phối của {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Tần số')
    plt.savefig('target_distribution.png')
    plt.close()

    # 2. Pairplot
    # Chọn các cột số
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Nếu quá nhiều cột, chỉ lấy một số cột quan trọng
    if len(numeric_cols) > 6:
        # Tính tương quan với biến mục tiêu
        correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        # Lấy 5 biến có tương quan cao nhất với biến mục tiêu
        top_features = correlations.index[:6]  # Bao gồm cả biến mục tiêu
        sns.pairplot(df[top_features], height=2.5)
    else:
        sns.pairplot(df[numeric_cols], height=2.5)

    plt.suptitle('Pairplot của các đặc trưng và biến mục tiêu', y=1.02)
    plt.savefig('pairplot.png')
    plt.close()

    # 3. Heatmap tương quan
    plt.figure(figsize=(12, 10))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Ma trận tương quan')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # 4. Tính và hiển thị Pearson correlation với biến mục tiêu
    correlations = df.corr()[target_col].sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlations.values, y=correlations.index)
    plt.title(f'Tương quan Pearson với {target_col}')
    plt.xlabel('Hệ số tương quan')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    plt.close()

    print("Đã tạo các biểu đồ trực quan hóa.")

    # In ra các đặc trưng có tương quan mạnh với biến mục tiêu
    strong_correlations = correlations[abs(correlations) > 0.5].drop(target_col, errors='ignore')
    print("\nCác đặc trưng có tương quan mạnh với biến mục tiêu:")
    print(strong_correlations)