import matplotlib.pyplot as plt

# 1. 准备示例数据
# 假设 X 轴是系统 A 的状态 (1-3)，Y 轴是系统 B 的状态 (1-3)
# 轨迹按时间顺序记录为 (x, y) 坐标点
trajectory = [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (2, 2), (1, 1)]

# 每个状态节点的停留时间（用于决定散点的大小）
durations = [300, 100, 500, 150, 400, 200, 600] 

# 提取 X 和 Y 坐标
x_coords = [point[0] for point in trajectory]
y_coords = [point[1] for point in trajectory]

# 2. 初始化画布
fig, ax = plt.subplots(figsize=(8, 8))

# 3. 绘制节点
# 中间节点：半透明实心圆形
ax.scatter(x_coords[1:], y_coords[1:], s=durations[1:], c='skyblue', edgecolors='none', alpha=0.5, zorder=3)
# 起点：空心圆形
ax.scatter(x_coords[0], y_coords[0], s=durations[0], facecolors='none', edgecolors='black', linewidths=2, zorder=4)

# 4. 绘制轨迹箭头
for i in range(len(trajectory) - 1):
    start_point = trajectory[i]
    end_point = trajectory[i + 1]
    
    # 使用 annotate 绘制带箭头的线
    ax.annotate('', xy=end_point, xytext=start_point,
                arrowprops=dict(arrowstyle="->", color="gray", 
                                shrinkA=15, shrinkB=15, # 缩进线条以免穿透圆点
                                connectionstyle="arc3,rad=0")) # 直线

# 5. 设置网格和坐标轴
# 设置主刻度标签（分类名称）
ax.set_xticks([1, 2, 3])
ax.set_yticks([1, 2, 3])
ax.set_xticklabels(['State A1', 'State A2', 'State A3'])
ax.set_yticklabels(['State B1', 'State B2', 'State B3'])

# 使用次要刻度来绘制完美的网格线 (画在标签中间)
ax.set_xticks([0.5, 1.5, 2.5, 3.5], minor=True)
ax.set_yticks([0.5, 1.5, 2.5, 3.5], minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1, alpha=0.5)

# 限制坐标轴范围
ax.set_xlim(0.5, 3.5)
ax.set_ylim(0.5, 3.5)

# 隐藏主刻度的网格线
ax.grid(which='major', visible=False)

# 添加标题和标签
ax.set_title("State Space Grid Example", fontsize=14, pad=20)
ax.set_xlabel("System A States")
ax.set_ylabel("System B States")

plt.show()