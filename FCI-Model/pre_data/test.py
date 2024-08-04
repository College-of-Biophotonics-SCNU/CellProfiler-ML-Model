from scipy import stats
import pandas as pd
import numpy as np

# 创建一个示例 DataFrame
data = {
    'feature1': [10, 12, 15, 18, 20, 22, 25, 28, 30, 100],
    'feature2': [20, 22, 25, 28, 30, 32, 35, 38, 40, 200],
    'feature3': [30, 32, 35, 38, 40, 42, 45, 48, 50, 300]
}
df = pd.DataFrame(data)

# 打印原始数据
print("Original Data:")
print(df)

# 计算每列的Z分数
z_scores = np.abs(stats.zscore(df))

print(z_scores)

# 创建一个布尔掩码来标识哪些行应该保留
mask = (z_scores < 3).all(axis=1)

# 应用掩码以保留满足条件的行
df_clean = df[mask]

# 打印清洗后的数据
print("\nCleaned Data:")
print(df_clean)