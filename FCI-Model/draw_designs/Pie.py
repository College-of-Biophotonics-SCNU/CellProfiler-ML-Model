import matplotlib.pyplot as plt

# 数据
labels = ['AreaShape', 'Intensity', 'Texture']
sizes = [49, 15, 52]

# 创建一个环形图，设置中心空白区域的大小
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, labels=['', '', ''], autopct='', startangle=90, wedgeprops=dict(width=0.4))

# 绘制一个圆形，用于创建环形图的效果
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# 设置标题
ax.set_title('Extracted Hoechst phenotype features')

# 添加图例
ax.legend(wedges, labels,
          title="Categories",
          loc="upper left",
          bbox_to_anchor=(0.9, 0, 0.5, 1))

# 调整图例文字和百分比的字体大小
for text in texts:
    text.set_fontsize(14)
for autotext in autotexts:
    autotext.set_fontsize(14)


plt.savefig('../../data/result/Hoechst/Pie.jpg', dpi=300)
# 显示图表
plt.show()
