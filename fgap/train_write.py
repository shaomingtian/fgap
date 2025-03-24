import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, HistGradientBoostingRegressor)
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time

import joblib  # 用于保存模型

import argparse


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="train write_models using a merged excel.")
    parser.add_argument('merged_excel', type=str, help='Path to the merged Excel file')
    return parser

parser = create_parser()
args = parser.parse_args()

# 读取Excel文件
file_path = args.merged_excel
data = pd.read_excel(file_path)

# 数据预处理
# 将非数字特征编码
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 定义特征和标签
X = data.drop(columns=['WriteBw', 'ReadBw', 'FileName', 'basename', 'darshan_log', 'time_start_access', 'time_end_access', 'total_write_size', 'total_read_size'])  # 特征
y = data['WriteBw']  # 标签

# 降低最后256个特征的权重
if X.shape[1] >= 256:
    X.iloc[:, -256:] *= 0.2  # 将最后256个特征的权重减小到50%

# 增加特定特征的权重
features_to_increase_weight = ['ExeNumFiles', 'FileSystem', 'FileNumHosts']
for feature in features_to_increase_weight:
    if feature in X.columns:
        X[feature] *= 5  # 将这些特征的权重增加50%.
X['ExeNumFiles'] *= 20
X['FileNumHosts'] *= 10
X['TopWriteLength'] *= 5

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.29, random_state=42)

# 定义模型
models = {
    'Linear_Regression': LinearRegression(),
    'Decision_Tree': DecisionTreeRegressor(random_state=42),
    'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Histogram-based_Gradient_Boosting': HistGradientBoostingRegressor(),
    'Extra_Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'Extreme_Gradient_Boosting': XGBRegressor(n_estimators=100, random_state=42)
}

# 评估模型性能
results = {}

for model_name, model in models.items():
    start_time = time.time()
    # 训练模型
    model.fit(X_train, y_train)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{model_name} train time: {execution_time:.2f} 秒")

    # 预测
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{model_name} pred time: {execution_time:.2f} 秒")

    y_pred_train = model.predict(X_train)

    # 创建一个包含训练集和测试集真实值与预测值的数据框
    data_train = pd.DataFrame({'True': y_train, 'Predicted': y_pred_train, 'Data Set': 'Train'})
    data_test = pd.DataFrame({'True': y_test, 'Predicted': y_pred, 'Data Set': 'Test'})
    data_combined = pd.concat([data_train, data_test])

    joblib.dump(model, f'models/write_model_{model_name}.joblib')  # 使用joblib保存模型


# 自定义调色板
    palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}

    # 创建 JointGrid 对象
    plt.figure(figsize=(4, 6), dpi=200)
    g = sns.JointGrid(data=data_test, x="True", y="Predicted", hue="Data Set", height=4, palette=palette)
    g.set_axis_labels(fontsize=12)
    g.ax_joint.tick_params(labelsize=12)  # 设置主图的刻度值大小
    g.ax_marg_x.tick_params(labelsize=12)  # 设置 x 边缘的刻度值大小
    g.ax_marg_y.tick_params(labelsize=12)  # 设置 y 边缘的刻度值大小

    # 绘制中心的散点图
    g.plot_joint(sns.scatterplot, alpha=0.5)

    # 添加测试集的回归线
    sns.regplot(data=data_test, x="True", y="Predicted", scatter=False, ax=g.ax_joint,
                color='#f4ba8a', label='Test Regression Line')

    # 添加边缘的柱状图
    g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)
    g.ax_marg_y.set_visible(False)
    g.ax_marg_x.set_visible(False)

    # 添加拟合优度文本在右下角
    ax = g.ax_joint
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred)

    # 计算 Adjusted R-Squared
    n = len(y_test)  # 样本大小
    p = X_test.shape[1]  # 特征数量
    adjusted_r2_train = 1 - (1 - r2_train) * (n - 1) / (n - p - 1)
    adjusted_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    ax.text(0.85, 0.13, f'$MAPE$ = {mape:.2f}%', transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax.text(0.85, 0.03, f'$Adjust\_R^2$ = {adjusted_r2_test:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # 在右侧和上方添加边框
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('black')

    g.ax_joint.legend_.remove()

    # 保存图像
    plt.savefig(f'models/write-{model_name}.png', bbox_inches='tight')
    plt.close()  # 关闭当前图像以释放内存

    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # 计算 Adjusted R-Squared
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # 计算 Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # 存储结果
    results[model_name] = {
        'Mean Squared Error': mse,
        'R² Score': r2,
        'Adjusted R²': adjusted_r2,
        'Mean Absolute Error': mae,
        'Mean Absolute Percentage Error': mape
    }

# 绘制调整后模型的比较图
model_names = list(results.keys())
a_r2s = [results[model]['Adjusted R²'] for model in model_names]
mapes = [results[model]['Mean Absolute Percentage Error'] for model in model_names]

# 创建一个新的图形和双轴对象
fig, ax1 = plt.subplots()
# 设置柱宽
bar_width = 0.35
# 设置横坐标的位置
x = np.arange(len(model_names))

# 绘制左边的柱子
bars_left = ax1.bar(x - bar_width / 2, a_r2s, bar_width, hatch="////", color='#193E8F', label='Adjusted R²')
ax1.set_ylabel('Adjusted R²', color='#193E8F')
ax1.tick_params(axis='y', labelcolor='#193E8F')

# 创建右边的坐标轴
ax2 = ax1.twinx()
# 绘制右边的柱子
bars_right = ax2.bar(x + bar_width / 2, mapes, bar_width, hatch="\\\\\\\\", color='#F09739', label='MAPE')
ax2.set_ylabel('MAPE (%)', color='#F09739')
ax2.tick_params(axis='y', labelcolor='#F09739')

# 设置横坐标标签和标题
ax1.set_xticks(x)
ax1.set_xticklabels(model_names)

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.savefig('models/wirte_total.png', dpi=300, bbox_inches='tight')

# 输出结果
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Mean Squared Error: {metrics['Mean Squared Error']:.4f}")
    print(f"  R² Score: {metrics['R² Score']:.4f}")
    print(f"  Adjusted R²: {metrics['Adjusted R²']:.4f}")
    print(f"  Mean Absolute Percentage Error: {metrics['Mean Absolute Percentage Error']:.4f}%")
    print(f"  Mean Absolute Error: {metrics['Mean Absolute Error']:.4f}")
    print()