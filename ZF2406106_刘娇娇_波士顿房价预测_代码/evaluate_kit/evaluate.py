import pandas as pd
import numpy as np

# 定义提交文件的路径
# 提交文件应包含模型预测的结果，格式需与 ground truth 文件一致
submission_path = "submission.csv"

# 加载真实值（ground truth）和提交的预测数据
# ground truth 文件 "answer.csv" 包含真实的 MEDV（房价中位数）值
# 提交文件 "submission.csv" 包含模型预测的 MEDV 值
gt_df = pd.read_csv("answer.csv")  # 真实值数据框
submission_df = pd.read_csv(submission_path)  # 提交的预测数据框

# 使用均方误差（Mean Squared Error, MSE）作为评估指标
# MSE 计算公式：MSE = mean((y_true - y_pred)^2)
# 其中 y_true 是真实值，y_pred 是预测值
mse = np.mean((gt_df["MEDV"] - submission_df["MEDV"]) ** 2)

# 输出均方误差结果，用于评估模型性能
# MSE 越小，表示预测值与真实值之间的偏差越小，模型性能越好
print(f"Mean Squared Error: {mse}")
