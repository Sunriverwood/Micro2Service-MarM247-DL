import pandas as pd
import os


def sort_csv_file(input_path, output_path):
    """
    读取一个CSV文件，根据温度和时间进行排序，并保存到新文件。

    :param input_path: 输入的CSV文件路径。
    :param output_path: 排序后要保存的CSV文件路径。
    """
    print(f"开始处理文件: '{input_path}'")

    # 1. 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件未找到，请检查路径是否正确。")
        return

    # 2. 使用 try-except 块来处理可能的读取或列名错误
    try:
        # 读取CSV文件到DataFrame
        df = pd.read_csv(input_path)

        # 定义排序依据的列
        sort_columns = ['True Time', 'True Temperature']

        # 检查所需列是否存在于DataFrame中
        if not all(col in df.columns for col in sort_columns):
            print(f"错误: CSV文件中缺少必要的排序列。需要 '{sort_columns[0]}' 和 '{sort_columns[1]}'。")
            return

        # 3. 执行排序
        #   - by: 指定排序的列名列表
        #   - ignore_index=True: 重置排序后结果的索引，使其从0开始连续递增
        print(f"正在根据 {sort_columns} 进行排序...")
        df_sorted = df.sort_values(by=sort_columns, ignore_index=True)

        # 4. 将排序后的DataFrame保存到新的CSV文件
        #   - index=False: 不将DataFrame的索引写入到CSV文件中
        df_sorted.to_csv(output_path, index=False)
        print(f"操作成功！排序后的文件已保存到: '{output_path}'")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == '__main__':
    # --- 配置 ---
    # 定义原始文件和输出文件的路径
    input_csv_path = 'reg/predictions_results.csv'
    sorted_csv_path = 'reg-0C/predictions_results_time.csv'

    # 执行排序函数
    sort_csv_file(input_csv_path, sorted_csv_path)