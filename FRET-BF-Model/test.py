import pandas as pd

# 假设你有一个名为df的DataFrame
# df = pd.read_csv('your_file.csv')  # 例如，从CSV文件中读取数据

# 为了演示，我们创建一个简单的DataFrame
data = {
    'Meat_Chicken': [1, 2, 3],
    'Meat_Beef': [4, 5, 6],
    'AreaShape_Carrot': [7, 8, 9],
    'Fruit_Apple': [10, 11, 12]
}
df = pd.DataFrame(data)

# 使用正则表达式过滤出所有以"Meat_"开头的列
meat_df = df.filter(regex='^AreaShape_')

print(meat_df)