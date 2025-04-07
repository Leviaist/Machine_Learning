import os 

# 路径相关的处理
current_path = os.path.dirname(os.path.abspath(__file__))                                       # 当前目录
parent_path = os.path.dirname(os.path.dirname(current_path))                                    # 根级目录
data_path =  os.path.join(parent_path, "Data/Raw_Data/Lev_Dataset/Normal/training_data.csv")    # 数据路径