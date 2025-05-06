import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(df, features):
    """Tính hệ số VIF cho các đặc trưng"""
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i)
                       for i in range(len(features))]
    return vif_data


def handle_multicollinearity(df, threshold=5):
    """Xử lý đa cộng tuyến dựa trên VIF"""
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'PRICE' in numeric_features:
        numeric_features.remove('PRICE')

    vif = calculate_vif(df, numeric_features)
    print("Hệ số VIF ban đầu:")
    print(vif)

    features_to_keep = numeric_features.copy()
    while True:
        vif = calculate_vif(df, features_to_keep)
        max_vif = vif['VIF'].max()
        if max_vif < threshold:
            break

        # Loại bỏ đặc trưng có VIF cao nhất
        exclude_feature = vif.loc[vif['VIF'] == max_vif, 'feature'].values[0]
        features_to_keep.remove(exclude_feature)
        print(f"Loại bỏ đặc trưng {exclude_feature} vì có VIF = {max_vif}")

    features_to_drop = [f for f in numeric_features if f not in features_to_keep]
    return features_to_drop


def engineer_features(df):
    """Thực hiện kỹ thuật feature engineering"""
    df_new = df.copy()

    # Tạo đặc trưng mới
    if 'RM' in df_new.columns and 'CRIM' in df_new.columns:
        df_new['ROOM_PER_CRIME'] = df_new['RM'] / (df_new['CRIM'] + 0.1)  # Tránh chia cho 0

    if 'TAX' in df_new.columns:
        avg_tax = df_new['TAX'].mean()
        df_new['HIGH_TAX'] = (df_new['TAX'] > avg_tax).astype(int)

    # Tạo các đặc trưng tương tác
    if 'RM' in df_new.columns and 'LSTAT' in df_new.columns:
        df_new['RM_LSTAT'] = df_new['RM'] * df_new['LSTAT']

    if 'DIS' in df_new.columns and 'AGE' in df_new.columns:
        df_new['DIS_AGE'] = df_new['DIS'] * df_new['AGE']

    # Tạo các đặc trưng phi tuyến bậc 2
    numeric_cols = df_new.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'PRICE' in numeric_cols:
        numeric_cols.remove('PRICE')

    if 'HIGH_TAX' in numeric_cols:
        numeric_cols.remove('HIGH_TAX')

    # Lựa chọn một số đặc trưng quan trọng để tránh bùng nổ số lượng đặc trưng
    important_features = ['LSTAT', 'RM', 'PTRATIO', 'DIS']
    important_features = [f for f in important_features if f in numeric_cols]

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df_new[important_features])

    # Tạo tên cho các đặc trưng đa thức
    poly_features_names = poly.get_feature_names_out(important_features)

    # Chỉ thêm các đặc trưng đa thức thuần túy (không bao gồm các đặc trưng ban đầu)
    for i in range(len(important_features), len(poly_features_names)):
        feature_name = f"POLY_{poly_features_names[i].replace(' ', '_')}"
        df_new[feature_name] = poly_features[:, i]

    # Xử lý đa cộng tuyến
    features_to_drop = handle_multicollinearity(df_new)
    df_new = df_new.drop(columns=features_to_drop)

    return df_new