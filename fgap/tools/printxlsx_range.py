import pandas as pd
import sys

def read_and_display_xlsx(file_name, start_col, end_col):
    # 尝试读取 Excel 文件
    try:
        # 读取 Excel 文件到 DataFrame
        df = pd.read_excel(file_name)

        # 检查列的范围是否有效
        if start_col < 1 or end_col < 1 or start_col > df.shape[1] or end_col > df.shape[1] or start_col > end_col:
            print(f"无效的列范围: 可用列范围为 1 到 {df.shape[1]}，请确认输入的开始列和结束列。")
            return
        
        # 输出指定范围的列的数据
        print(df.iloc[:, start_col-1:end_col])  # 由于 iloc 是基于 0 的索引

    except FileNotFoundError:
        print(f"文件 '{file_name}' 未找到，请检查文件路径。")
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    # 检查是否提供了文件名和列范围作为参数
    if len(sys.argv) != 4:
        print("用法: python read_xlsx.py <xlsx文件名> <开始列序号> <结束列序号>")
        sys.exit(1)

    file_name = sys.argv[1]
    try:
        start_col = int(sys.argv[2])
        end_col = int(sys.argv[3])
    except ValueError:
        print("开始列序号和结束列序号必须是整数。")
        sys.exit(1)

    # 调用函数读取和显示文件内容
    read_and_display_xlsx(file_name, start_col, end_col)
