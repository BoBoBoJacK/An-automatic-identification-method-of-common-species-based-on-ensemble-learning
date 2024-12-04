本实验使用了 AutoGluon 进行物种分类任务。若您尚未安装 AutoGluon，请前往官方下载并按照安装步骤进行安装：
[AutoGluon官方文档](https://auto.gluon.ai/stable/index.html)

数据集说明
本实验以 CC 数据集 为例，任务为物种分类。数据集中的物种被分为常见物种和稀有物种：
常见物种：前9个物种
稀有物种：剩余的物种

文件夹结构
本项目包含以下几个文件夹和文件：
1. Com/
此文件夹包含常见物种模型的相关代码：

train.py：用于训练常见物种模型的代码。
test.py：用于测试常见物种模型的代码。
calculate.py：生成常见物种模型的评估指标和混淆矩阵。
2. All/
此文件夹包含全物种模型的相关代码：

train.py：用于训练全物种模型的代码。
test.py：用于测试全物种模型的代码。
calculate.py：生成全物种模型的评估指标和混淆矩阵。
3. test/
此文件夹包含一个文件：ensemble.py。此文件实现集成学习代码，负责执行两个阶段的模型评估并生成相关结果。它将会调用 Com/calculate.py 和 ALL/calculate.py，并生成混淆矩阵、评估指标以及结果文件。


使用说明
1. 安装依赖
请确保已安装 AutoGluon 和相关的 Python 依赖。如果尚未安装，请参考AutoGluon官方文档
2. 训练模型
进入相应的文件夹（Com/ 或 All/），执行以下命令训练模型：
python train.py
3. 测试模型
在训练完成后，使用以下命令进行模型测试并保存结果：
python test.py
4. 生成评估结果
在test目录下运行 ensemble.py 文件：
python ensemble.py
该操作将生成每个模型的混淆矩阵、metrics 文件和其他结果，并保存在 Com/ 和 ALL/ 文件夹中。
注意:calculate.py 和 ensemble.py中需要根据实际情况更改参数，如物种总数和常见物种个数等

参考
[AutoGluon官方文档](https://auto.gluon.ai/stable/index.html)
