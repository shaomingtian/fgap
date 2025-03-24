import pandas as pd
import joblib

# 1. 读取 Excel 文件，获取 features 信息
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import fgap_config

def load_features_from_excel(file_path):
    data = pd.read_excel(file_path)

    # 数据预处理
    # 将非数字特征编码
    # label_encoders = {}
    # for column in data.select_dtypes(include=['object']).columns:
    #     le = LabelEncoder()
    #     data[column] = le.fit_transform(data[column])
    #     label_encoders[column] = le

    # 定义特征和标签
    # data.drop(columns=['WriteBw', 'ReadBw', 'FileName', 'basename', 'darshan_log'])  # 特征
    data.drop(columns=['WriteBw', 'ReadBw', 'FileName', 'basename', 'darshan_log', 'time_start_access', 'time_end_access', 'total_write_size', 'total_read_size'])  # 特征
    return data

    #
    # df = pd.read_excel(file_path, sheet_name=None)  # 读取所有表
    # features_data = {}
    #
    # for sheet_name, sheet_data in df.items():
    #     # 删除列名为 "basename" 和 "dirname" 的列
    #     columns_to_drop = ["darshan_log", "FileName", "basename",  "ReadBw", "WriteBw"]
    #     sheet_data = sheet_data.drop(columns=columns_to_drop, errors='ignore')  # errors='ignore' 会在未找到列时不报错
    #
    #     features_data[sheet_name] = sheet_data
    #
    # return features_data

# 2. 读取 fs.config 配置文件
def load_fs_config(config_path):
    fs_config = {}
    with open(config_path, 'r') as f:
        lines = f.readlines()

    # 跳过以#开头的注释行
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue

        # 切分行中的信息
        parts = line.strip().split()
        if len(parts) < 6:
            continue  # 确保行中至少包含足够的信息

        fs_type = parts[0]
        dir_path = parts[1]
        stripe_size = parts[2]
        stripe_count = int(parts[3])
        bf = parts[4]
        sf = parts[5]

        # 将信息存储到字典中
        if fs_type not in fs_config:
            fs_config[fs_type] = {
                'dir': dir_path,
                'stripe_size': stripe_size,
                'stripe_count': stripe_count,
                'bf': bf,
                'sf': sf
            }

    return fs_config

# 3. 加载模型
def load_model(model_path):
    return joblib.load(model_path)

# 4. 进行预测
def predict_with_models(features_data, fs_config):
    predictions = []

    write_models = [("write_model_Decision_Tree.joblib", "model1"), ("write_model_Extra_Trees.joblib", "model2")]
    read_models = [("read_model_Decision_Tree.joblib", "model1"), ("read_model_Extra_Trees.joblib", "model2")]

    # 写入性能模型预测
    # for (write_model_path, write_model_name), (read_model_path, read_model_name) in zip(write_models, read_models):
    write_model_path = fgap_config.FGAP_WRITE_MODEL[1]
    read_model_path = fgap_config.FGAP_READ_MODEL[1]
    if write_model_path != "" and read_model_path != "":
    # for write_model_path, read_model_path in zip(fgap_config.FGAP_WRITE_MODEL, fgap_config.FGAP_READ_MODEL):
        write_model = load_model(write_model_path)
        read_model = load_model(read_model_path)
        expected_feature_names = write_model.feature_names_in_
        for fs_type, config in fs_config.items():
            stripe_size = config['stripe_size']
            stripe_count = config['stripe_count']
            fs_dir = config['dir']


            for index, feature_df in features_data.iterrows():
                feature_combination = feature_df.to_frame().T
                feature_combination["FileSystem"] = fs_type
                feature_combination["StripeSize"] = stripe_size
                feature_combination["StripeCount"] = stripe_count
                feature_combination = feature_combination[expected_feature_names]
                # print(feature_combination)
                label_encoders = {}
                for column in feature_combination.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    feature_combination[column] = le.fit_transform(feature_combination[column])
                    label_encoders[column] = le

                # 性能预测
                if fs_type == "tmpfs":
                    if feature_df["FileNumHosts"] == 1:
                        read_bw_predicted = write_bw_predicted = [20000]
                    else:
                        read_bw_predicted = write_bw_predicted = [0]
                else:
                    write_bw_predicted = write_model.predict(feature_combination)
                    read_bw_predicted = read_model.predict(feature_combination)
                if write_bw_predicted[0] < 0:
                    write_bw_predicted[0] = 0
                if read_bw_predicted[0] < 0:
                    read_bw_predicted[0] = 0
                # print("read", feature_df["basename"], read_model_name, fs_type, stripe_size, stripe_count, f"{read_bw_predicted[0]:.2f}")
                # print("write", feature_df["basename"], write_model_name, fs_type, stripe_size, stripe_count, f"{write_bw_predicted[0]:.2f}")

                record = {
                    'basename': feature_df["basename"],
                    'write_model': write_model_path,
                    'read_model': read_model_path,
                    'filesystem': fs_type,
                    'stripe_size': stripe_size,
                    'stripe_count': stripe_count,
                    'fs_dir': fs_dir,
                    'WriteBw': f"{write_bw_predicted[0]:.2f}",
                    'ReadBw': f"{read_bw_predicted[0]:.2f}",
                    'Size': feature_df["FileSize"],
                    'ReadSize': feature_df["total_read_size"],
                    'WriteSize': feature_df["total_write_size"],
                    'TS': feature_df["time_start_access"],
                    'TE': feature_df["time_end_access"]
                }
                predictions.append(record)
    return predictions

def write_records_to_txt(predictions, filename="pred.txt"):
    with open(filename, 'w') as f:
        # 写入文件头
        f.write("# pred.txt\n")
        f.write("# filename fs_type fs_dir fs_stripe_size fs_stripe_count pred_bw pred_rw Size ReadSize WriteSize TS TE\n")

        for record in predictions:
            # 创建一行字符串
            line = f"{record['basename']} {record['filesystem']} {record['fs_dir']} {record['stripe_size']} " \
                   f"{record['stripe_count']} {record['WriteBw']} {record['ReadBw']} {record['Size']} " \
                   f"{record['ReadSize']} {record['WriteSize']} {record['TS']} {record['TE']}\n"
            f.write(line)  # 写入文件


def pred_save_perf(excel_file_path, record_performance_path):
    fs_config_path = fgap_config.FS_CONFIG_PATH
    features_data = load_features_from_excel(excel_file_path)
    fs_config = load_fs_config(fs_config_path)
    predictions = predict_with_models(features_data, fs_config)
    write_records_to_txt(predictions, record_performance_path)

# 主程序
if __name__ == "__main__":
    # 文件路径和配置
    excel_file_path = 'darshan-log/4nodes-ppn1-100m-5m-F-C-g-v1.darshan.xlsx'  # 替换为实际文件路径
    fs_config_path = 'fs.config'   # 替换为实际配置文件路径

    pred_save_perf(excel_file_path, "pred_test.txt")
