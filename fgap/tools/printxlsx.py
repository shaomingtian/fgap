import pandas as pd
import sys

def read_and_display_xlsx(file_name, output_columns=22):
    # 尝试读取 Excel 文件
    try:
        # 读取 Excel 文件到 DataFrame
        df = pd.read_excel(file_name)
        
        # 检查是否有足够的列，然后输出前 output_columns 列
        if df.shape[1] < output_columns:
            print(f"文件中列的总数只有 {df.shape[1]} 列，实际输出所有列内容:")
            output_columns = df.shape[1]
        
        # 输出前 output_columns 列的数据
        print(df.iloc[:, :output_columns])
    
    except FileNotFoundError:
        print(f"文件 '{file_name}' 未找到，请检查文件路径。")
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    # 检查是否提供了文件名作为参数
    if len(sys.argv) != 2:
        print("用法: python read_xlsx.py <xlsx文件名>")
        sys.exit(1)

    file_name = sys.argv[1]
    
    # 调用函数读取和显示文件内容
    read_and_display_xlsx(file_name)
