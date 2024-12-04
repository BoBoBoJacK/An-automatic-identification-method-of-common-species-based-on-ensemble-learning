import os
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import shutil


def load_data(csv_file_path, img_root):
    data = pd.read_csv(csv_file_path)
    data['filename'] = data['filename'] .apply(lambda x: os.path.join(img_root, x))
    return data

def filter_unknown_labels(train_data, val_data, output_csv_path):
    # 获取训练集中的所有标签
    valid_labels = set(train_data['label'].unique())

    # 找到验证集中未在训练集标签集合中的样本
    unknown_labels_data = val_data[~val_data['label'].isin(valid_labels)]

    # 保存这些未知标签的样本信息到一个新的 CSV 文件中
    unknown_labels_data.to_csv(output_csv_path, index=False)

    # 过滤掉验证集中的未知标签，只保留训练集中存在的标签
    val_data_filtered = val_data[val_data['label'].isin(valid_labels)]

    # 打印一下过滤后的验证集标签
    print("Filtered validation labels:", val_data_filtered['label'].unique())

    return val_data_filtered, unknown_labels_data

def main():
    # 指定CSV文件路径和图像的根目录
    sevent_root = "/data/users/lhx/csv/CC_9/"
    img_root = '/data/datasets/'
    model_path = './model'

    # 拼接路径
    train_csv = os.path.join(sevent_root, "train_all.csv")
    val_csv = os.path.join(sevent_root, "val_all.csv")

    # 加载并预处理数据
    train_data = load_data(train_csv, img_root)
    val_data = load_data(val_csv, img_root)

    # 指定输出文件路径
    output_csv_path = "deleted_unknown_labels.csv"

    # 过滤掉验证集中的未知标签，并记录删除的样本
    val_data_filtered, unknown_labels_data = filter_unknown_labels(train_data, val_data, output_csv_path)

    # 如果你想查看被删除的未知标签数据
    print(f"Number of deleted unknown labels: {len(unknown_labels_data)}")
    print(unknown_labels_data.head())  # 可以打印前几行查看

    # 创建一个模型保存路径
    model_path = "./model/"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # 创建一个MultiModalPredictor对象
    predictor = MultiModalPredictor(label='label', path=model_path, pretrained=False)

    # 自动训练模型
    predictor.fit(
        train_data=train_data, 
        tuning_data=val_data_filtered, 
        save_path = model_path,
        hyperparameters={
            "model.timm_image.checkpoint_name": "resnext50_32x4d",
            # "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop", "trivial_augment"], # 数据增强
            # "optimization.learning_rate": 1.0e-4,
            # "optimization.optim_type": "adamw",
            # "optimization.weight_decay": 1.0e-3, # 权重衰减
            # "optimization.lr_decay": 0.9, # 学习率衰减
            # "optimization.lr_schedule": "cosine_decay", # 可选 polynomial_decay ，linear_decay
            # "env.batch_size": 128, # 训练使用的批大小
            # "env.eval_batch_size_ratio": 4, # 预测或评估使用的批量大小
            # "env.num_workers": 2, # 训练中使用的工作进程数
            # "env.num_workers_evaluation": 2, # 预测或评估中使用的工作进程数量
            # "env.num_gpus": 1,  # GPU数量为1
            # "optimization.warmup_steps": 0.1, # 前10%的step预热
            # "optimization.val_check_interval": 0.5, # 训练周期内检查验证集的频率
            # "optimization.max_epochs": 50,  # 将训练轮数调整为50轮
            # "optimization.patience": 5,  # 5轮没提升就停止
            # "optimization.skip_final_val": True #  跳过验证
    })

    # 保存模型
    predictor.save(model_path)

if __name__ == '__main__':
    main()