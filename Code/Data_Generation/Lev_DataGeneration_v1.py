import pandas as pd
import numpy as np

# 设定随机种子，保证结果可复现
np.random.seed(42)

# 生成 1000 行数据，每列自变量范围为 0~10
num_samples = 1000
X = np.random.uniform(0, 10, size=(num_samples, 5))

# 自定义因变量映射关系
def target_function(x1, x2, x3, x4, x5):
    return 2*x1 + 3*x2 - 0.5*x3 + 0.8*x4 + np.sin(x5) + np.random.normal(0, 0.5)

# 计算因变量
Y = np.array([target_function(*row) for row in X])

# 组装成 DataFrame
df = pd.DataFrame(X, columns=['X1', 'X2', 'X3', 'X4', 'X5'])
df['Y'] = Y

# 查看前几行数据
print(df.head())

# 保存为 CSV 文件
df.to_csv("Project/Repo/Machine_Learning/Data/Raw_Data/Lev_Dataset/Normal/training_data.csv", index=False)
