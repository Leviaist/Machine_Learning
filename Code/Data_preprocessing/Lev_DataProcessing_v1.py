import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 路径相关的处理
current_path = os.path.dirname(os.path.abspath(__file__))                                       # 当前目录
parent_path = os.path.dirname(os.path.dirname(current_path))                                    # 根级目录
data_path =  os.path.join(parent_path, "Data/Raw_Data/Lev_Dataset/Normal/training_data.csv")    # 数据路径

# 读取 CSV 数据
df = pd.read_csv(data_path)

# 查看数据基本信息
print(df.info())   # 看数据类型
print(df.head())   # 预览前 5 行数据

# 检查缺失值
print(df.isnull().sum())

# 填充缺失值（常用方法）
df.fillna(df.mean(), inplace=True)   # 用均值填充
# df.fillna(0, inplace=True)         # 用 0 填充
# df.dropna(inplace=True)            # 直接删除缺失值行

# 归一化（将值缩放到 [0,1]）
scaler = MinMaxScaler()
df[['X1', 'X2', 'X3', 'X4', 'X5']] = scaler.fit_transform(df[['X1', 'X2', 'X3', 'X4', 'X5']])

# 标准化（均值 0，方差 1）
# std_scaler = StandardScaler()
# df[['X1', 'X2', 'X3', 'X4', 'X5']] = std_scaler.fit_transform(df[['X1', 'X2', 'X3', 'X4', 'X5']])

# 设定自变量 (X) 和 目标变量 (Y)
X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
Y = df['Y']

# 按 80% 训练集，20% 测试集划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 保存处理后的数据
X_train.to_csv(os.path.join(parent_path, "Data/Processed_Data/Lev_Dataset/Normal/X_train.csv"), index=False)
X_test.to_csv(os.path.join(parent_path, "Data/Processed_Data/Lev_Dataset/Normal/X_test.csv"), index=False)
Y_train.to_csv(os.path.join(parent_path, "Data/Processed_Data/Lev_Dataset/Normal/Y_train.csv"), index=False)
Y_test.to_csv(os.path.join(parent_path, "Data/Processed_Data/Lev_Dataset/Normal/Y_test.csv"), index=False)

print("数据预处理完成，已保存训练 & 测试数据集！")
