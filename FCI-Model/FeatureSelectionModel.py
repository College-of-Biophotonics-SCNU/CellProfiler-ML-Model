import pandas as pd
import numpy as np
from scipy.stats import shapiro, boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# 假设 df 是包含特征和标签的 DataFrame
df = pd.read_csv('your_data.csv')

# 分离特征和标签
X = df.drop('label_column', axis=1)  # 假设 label_column 是你的标签列
y = df['label_column']


# Shapiro-Wilk 正态性检验
def shapiro_test(X):
    p_values = []
    for col in X.columns:
        _, p_value = shapiro(X[col])
        p_values.append(p_value)
    return p_values


# Box-Cox 变换
def apply_boxcox(X, p_values, alpha=0.05):
    transformed_X = X.copy()
    lambdas = {}
    for i, col in enumerate(X.columns):
        if p_values[i] < alpha:  # 如果未通过正态性检验
            transformed_X[col], lambda_val = boxcox(X[col] + 1)  # 避免负数和零
            lambdas[col] = lambda_val
    return transformed_X, lambdas


# 去除冗余特征
def remove_redundant_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return X.drop(X[to_drop], axis=1)


# Logistic Lasso 回归
def lasso_feature_selection(X, y, alpha=0.01):
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0 / alpha, random_state=42)
    sfm = SelectFromModel(lasso, threshold='median')
    sfm.fit(X, y)
    selected_features = X.columns[sfm.get_support()]
    return selected_features


# 多项式 Logistic Lasso 回归
def poly_lasso_feature_selection(X, y, degree=2, alpha=0.01):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0 / alpha, random_state=42)
    sfm = SelectFromModel(lasso, threshold='median')
    sfm.fit(X_poly_df, y)
    selected_features = poly.get_feature_names_out(X.columns)[sfm.get_support()]
    return selected_features


# 主流程
def main_flow(df):
    X = df.drop('label_column', axis=1)
    y = df['label_column']

    # Shapiro-Wilk 正态性检验
    p_values = shapiro_test(X)

    # Box-Cox 变换
    X_transformed, lambdas = apply_boxcox(X, p_values)

    # 去除冗余特征
    X_no_redundant = remove_redundant_features(X_transformed)

    # Logistic Lasso 回归
    selected_features = lasso_feature_selection(X_no_redundant, y)

    # 多项式 Logistic Lasso 回归
    selected_features_poly = poly_lasso_feature_selection(X_no_redundant[selected_features], y)

    return selected_features_poly

# 假设 df 是你的 DataFrame
selected_features = main_flow(df)

print(selected_features)