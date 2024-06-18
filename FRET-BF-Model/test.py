import pandas as pd

# 创建两个简单的DataFrame
df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
df2 = pd.DataFrame({'A': [3, 4]}, index=[1, 2])

# 使用concat拼接，但不重置索引
result1 = pd.concat([df1, df2])
print("不重置索引:")
print(result1)

# 使用concat拼接，并重置索引
result2 = pd.concat([df1, df2], ignore_index=True)
print("\n重置索引:")
print(result2)