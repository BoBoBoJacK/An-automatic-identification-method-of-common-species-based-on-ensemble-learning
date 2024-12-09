import os
import shutil
from sklearn.metrics import confusion_matrix
import pandas as pd
import subprocess

num_common = 9
num_all = 54
rare_class = 99

def config(result_csv_path, result_path):
    csv_data = pd.read_csv(result_csv_path)
    # Actual列保持不变
    # 创建新列 Actual_process， 对于>7 的值替换为 rare_class
    csv_data['Actual_process'] = csv_data['Actual'].apply(lambda x: rare_class if x > num_common-1 else x)
    # 调整列的顺序，确保 Actual 和 Actual_process 排在一起
    columns_order = ['Actual', 'Actual_process'] + [col for col in csv_data.columns if col not in ['Actual', 'Actual_process']]
    csv_data = csv_data[columns_order]
    
    # 选取除了Actual和Actual_process以外的所有列作为预测数据
    pre_data = csv_data.iloc[:, 3:]  
 
    # 计算预测值的最大索引和最大值
    csv_data['max_idx'] = pre_data.idxmax(axis=1).astype(int)
    csv_data['max_val'] = pre_data.max(axis=1)

    # 确保目标文件夹存在
    os.makedirs(result_path, exist_ok=True)

    # 置信阈值为 95%
    confidence_threshold = 95 / 100
    with open(os.path.join(result_path, "95.csv"), 'a+') as f:
        # 写入表头
        f.write("Actual,Actual_process,Predicted\n")
        # 根据置信阈值写入数据
        for _, row in csv_data.iterrows():
            # 如果最大值大于等于阈值，则使用最大索引，否则使用 rare_class
            confi_pre = row['max_idx'] if row['max_val'] >= confidence_threshold else rare_class
            # 写入实际值，处理后的实际值，和预测结果
            f.write(f"{row['Actual']},{row['Actual_process']},{confi_pre}\n")

def process_all(result_csv_path, result_path):
    csv_data = pd.read_csv(result_csv_path)
    # Actual列保持不变
    # 创建新列 Actual_process， 对于>num_common 的值替换为 rare_class
    csv_data['Actual_process'] = csv_data['Actual'].apply(lambda x: rare_class if x > num_common-1 else x)
    csv_data['Predicted'] = csv_data['Predicted'].apply(lambda x: rare_class if x > num_common-1 else x)
    csv_data = csv_data[['Actual', 'Actual_process', 'Predicted']]
    csv_data.to_csv(result_path, index=False)

def ensemble(all_path, common_path, result_path):
    df_all = pd.read_csv(all_path, usecols=['Actual', 'Predicted'])
    df_common = pd.read_csv(common_path, usecols=['Predicted'])

    df_all.loc[:, 'Predicted'] = df_all['Predicted'].where(
        df_all['Predicted'] == df_common['Predicted'], rare_class
    )

    # 使用 .loc[] 避免 SettingWithCopyWarning 警告
    df_all.loc[:, 'Actual_process'] = df_all['Actual'].apply(lambda x: rare_class if x > num_common-1 else x)
    
    df_all = df_all[['Actual', 'Actual_process', 'Predicted']]
    df_all.to_csv(result_path, index=False)

def metrics(ensemble_path,metrics_path):
    df = pd.read_csv(ensemble_path)
    species = df['Actual_process'].unique()
    results = []
    correct_predictions = (df['Actual_process'] == df['Predicted']).sum()
    total_samples = df.shape[0]
    total_accuracy = correct_predictions / total_samples

    for specie in species:
        tp = len(df[(df['Actual_process'] == specie) & (df['Predicted'] == specie)])
        fp = len(df[(df['Actual_process'] != specie) & (df['Predicted'] == specie)])
        fn = len(df[(df['Actual_process'] == specie) & (df['Predicted'] != specie) & (df['Predicted'] != rare_class)])
        tn = len(df[(df['Actual_process'] != specie) & (df['Predicted'] != specie)])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (fn + tp) if (fn + tp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
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
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Species')
    column_order = ['Species', 'TP', 'FP', 'FN', 'TN','Recall', 'Precision',  'F1-Score']
    results_df = results_df[column_order]
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

def calculate_AutomationRate(matrix_path):
    # 加载CSV文件
    data = pd.read_csv(matrix_path, index_col=0)
    # 选择分子区域（所有行，列0-num_common）
    numerator_data = data.iloc[:, :num_common]
    # 分母 = 所有
    denominator_data = data.iloc[:,:]  
    # 计算总分子和总分母
    total_numerator = numerator_data.values.sum()
    total_denominator = denominator_data.values.sum()
    
    # 计算自动识别率 =（常见物种个数 / 总数）* 100%
    AutomationRate = (total_numerator / total_denominator) * 100
    print(f"AutomationRate: {AutomationRate:.2f}")

def run_ensemble(directory):
    # 指定 Python 解释器和脚本路径
    command = ['python', 'calculate.py']
    # 修改当前工作目录到指定目录
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    print(f'Running in {directory}:')
    # 输出标准输出
    print(result.stdout)  

def main():
    ensemble_path = "ensemble.csv"
    matrix_path = 'matrix.csv'
    metrics_path = 'metrics.csv'
    all_path = 'all.csv'
    common_path = './config_com/95.csv'
    result_com_path = 'com_results.csv'
    result_all_path = 'all_results.csv'
    confi_com_log = './config_com'

    print('正在计算阈值')
    # 计算常见物种模型置信阈值
    config(result_com_path, confi_com_log)
    # 处理全物种结果
    process_all(result_all_path, all_path)
    # 集成
    ensemble(all_path,common_path,ensemble_path)
    # 计算指标
    metrics(ensemble_path,metrics_path)
    # 计算矩阵
    matrix(ensemble_path,matrix_path)
    # 计算漏判
    calculate_missrate(matrix_path)

    # 移动文件到指定目录
    shutil.copy(all_path, '../all/all.csv')
    shutil.copy(common_path, '../com/95.csv')

    # 通过矩阵计算漏判和覆盖率
    calculate_AutomationRate(matrix_path)

    # 在指定目录下运行 
    run_ensemble('../com')
    run_ensemble('../all')

if __name__ == '__main__':
    main()
