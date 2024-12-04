from sklearn.metrics import confusion_matrix
import pandas as pd

num_common = 9
num_all = 54
rare_class = 99

def filtered(common_path,result_path):
    # 读取两个CSV文件
    df_common = pd.read_csv(common_path, usecols=['Actual', 'Predicted'])
    # 创建一个掩码，指示哪些行的Predicted值不为NaN
    mask = df_common['Predicted'].notna()
    # 使用掩码过滤掉两个DataFrame中Predicted为NaN的行
    df_common_filtered = df_common[mask]
    # 保存新的DataFrame到CSV文件
    df_common_filtered.to_csv(result_path, index=False)

def metrics(data_path,metrics_path):
    # 读取集成结果并计算指标
    df = pd.read_csv(data_path)
    # 获取所有独特的物种
    species = df['Actual_process'].unique()
    # 存储结果的列表
    results = []
    # 添加计算总体准确率
    correct_predictions = (df['Actual_process'] == df['Predicted']).sum()
    total_samples = df.shape[0]
    total_accuracy = correct_predictions / total_samples
    # 遍历每个物种，计算指标
    for specie in species:
        # 计算TP, FP, FN, TN
        tp = len(df[(df['Actual_process'] == specie) & (df['Predicted'] == specie)])
        fp = len(df[(df['Actual_process'] != specie) & (df['Predicted'] == specie)])
        fn = len(df[(df['Actual_process'] == specie) & (df['Predicted'] != specie) & (df['Predicted'] != rare_class)])
        tn = len(df[(df['Actual_process'] != specie) & (df['Predicted'] != specie)])
        
        # 计算Precision, Recall, F1-Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (fn + tp) if (fn + tp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # 将结果添加到列表
        results.append({
            'Species': specie,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Recall': round(recall, 4)*100,
            'Precision': round(precision, 4)*100,
            'F1-Score': round(f1, 4)*100,
        })
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    # 按Species的值从小到大排序
    results_df = results_df.sort_values(by='Species')
    # 指定列顺序
    column_order = ['Species', 'TP', 'FP', 'FN', 'TN','Recall', 'Precision',  'F1-Score']
    # 重新排列列顺序
    results_df = results_df[column_order]
    # 保存结果到CSV文件
    results_df.to_csv(metrics_path, index=False)
    print(f"Total Accuracy: {total_accuracy:.4f}")

def matrix(ensemble_path, matrix_path):
    # 读取数据，确保 Actual 和 Predicted 列是整数类型
    df = pd.read_csv(ensemble_path, usecols=['Actual', 'Predicted'], dtype={'Actual': int, 'Predicted': int})
    # 确保处理特殊标签rare_class
    df['Predicted'] = df['Predicted'].apply(lambda x: rare_class if x > num_common-1 else x)
    # 获取所有独特的类别标签
    unique_labels = sorted(set(df['Actual'].unique()).union({rare_class}))
    # 生成混淆矩阵
    cm = confusion_matrix(df['Actual'], df['Predicted'], labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    # 确保所有的列名都是字符串类型，以便可以正确删除
    cm_df.columns = [str(col) for col in cm_df.columns]
    # 删除列名为num_common到num_all的列
    cols_to_remove = [str(i) for i in range(num_common, num_all) if str(i) in cm_df.columns]
    cm_df.drop(columns=cols_to_remove, inplace=True)
    # 将结果保存到CSV文件
    cm_df.to_csv(matrix_path, index=True)

def calculate_missrate(matrix_path):
    # 加载CSV文件
    data = pd.read_csv(matrix_path, index_col=0)
    # 选择分子区域（行num_common-num_all，列0-num_common）
    numerator_data = data.iloc[num_common:num_all, :num_common]
    # 选择分母区域（行num_common-num_all，列所有）
    denominator_data = data.iloc[num_common:num_all,:] 
    # 计算总分子和总分母
    total_numerator = numerator_data.values.sum()
    total_denominator = denominator_data.values.sum()
    # 计算MissRate
    miss_rate = (total_numerator / total_denominator) * 100  # 转换为百分比形式
    print(f'MissRate: {miss_rate:.2f}')
    
def main():
    data_path = "95.csv"
    matrix_path = 'matrix.csv'
    metrics_path = 'metrics.csv'

    # 计算矩阵
    matrix(data_path,matrix_path)
    # 计算性能指标
    metrics(data_path,metrics_path)
    # 计算漏判
    calculate_missrate(matrix_path)

if __name__ == '__main__':
    main()

