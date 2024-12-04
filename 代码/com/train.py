import os
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import shutil

def load_data(csv_file_path, img_root):
    data = pd.read_csv(csv_file_path)
    data['filename'] = data['filename'] .apply(lambda x: os.path.join(img_root, x))
    return data

def main():
    # 指定CSV文件路径和图像的根目录
    sevent_root = "/data/users/lhx/csv/CC_9/"
    img_root = '/data/datasets/'
    model_path = './model'

    # 拼接路径
    train_csv = os.path.join(sevent_root, "train.csv")
    val_csv = os.path.join(sevent_root, "val.csv")

    # 加载并预处理数据
    train_data = load_data(train_csv, img_root)
    val_data = load_data(val_csv, img_root)

    # 创建一个模型保存路径
    model_path = "./model/"
    if os.path.exists(model_path):
        shutil.rmtree(model_path )

    # 创建一个MultiModalPredictor对象
    predictor = MultiModalPredictor(label='label', path=model_path)

    # 自动训练模型
    predictor.fit(
        train_data=train_data, 
        tuning_data=val_data, 
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


