# coding:utf-8
import csv
import os
import os.path
import random
from collections import Counter

save_path = "./"
csv_file = "/data/users/lhx/csv/SS/S1-6/all_addhuman/S1_6_original.csv"
train_all_file = os.path.join(save_path, "train_all.csv")
val_all_file = os.path.join(save_path, "val_all.csv")
train_file = os.path.join(save_path, "train.csv")
val_file = os.path.join(save_path, "val.csv")
test_file = os.path.join(save_path, "test.csv")
info_common_file = os.path.join(save_path, "ss_common_info.csv")
info_all_file = os.path.join(save_path, "ss_all_info.csv")

def count_allspecies_from_csv(rows):
    """从原始的csv中统计每个有效物种的数量，仅统计指定的标签列且排除'human'"""
    species_counts = Counter()
    try:
        for row in rows:
            labels = [row[i].strip() for i in [8, 10, 12] if i < len(row) and 
                      row[i].strip() and 
                      row[i].strip().lower() != 'human']
            for label in labels:
                species_counts[label] += 1
    except Exception as e:
        print(f"处理数据时出现错误：{e}")
    return species_counts

def count_species_from_csv(csv_file_path, species_rank):
    """从生成的CSV文件中读取数据并统计物种的数量，输入为文件路径和排名到物种名的映射字典"""
    species_count = Counter()
    rank_to_species = {v: k for k, v in species_rank.items()}  # 从排名数字映射到物种名

    try:
        with open(csv_file_path, "r", newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过表头
            for row in reader:
                rank = int(row[1])  # 从数据行中获取物种排名
                species = rank_to_species.get(rank, None)  # 获取对应的物种名
                if species:
                    species_count[species] += 1
                else:
                    # 处理不存在于字典中的排名
                    print(f"Rank {rank} not found in species_rank dictionary.")
    except Exception as e:
        print(f"Error reading or processing the file: {e}")
    return species_count

def create_species_rank(species_counts):
    """根据物种的出现频率生成一个物种到其排名和数量的映射"""
    sorted_species = sorted(species_counts.items(), key=lambda item: item[1], reverse=True)
    species_rank = {species: i for i, (species, count) in enumerate(sorted_species)}
    return species_rank

def split_dataset(dataset, train_ratio, val_ratio):
    """将数据集分割为训练集、验证集和测试集"""
    random.shuffle(dataset)
    total = len(dataset)
    train_num = int(train_ratio * total)
    val_num = int(val_ratio * total)
    test_num = total - train_num - val_num
    train_set = dataset[:train_num]
    val_set = dataset[train_num:train_num + val_num]
    test_set = dataset[-test_num:]
    return train_set, val_set, test_set

def write_csv(file_path, rows, header):
    """train和val中只保留前11个常见物种"""
    with open(file_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            if int(row[1]) > 10:
                continue  # 跳过此行，不写入CSV
            writer.writerow(row)

def write_csv_all(file_path, rows, header):
    """将数据写入CSV文件，同时修改filename格式和处理排名"""
    with open(file_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

def process_data(rows, species_counts, species_rank):
    """处理并根据物种数量选择数据，同时计数选择后的数据中每个物种的数量"""
    species_data = {species: [] for species in species_counts}
    selected_species_counts = Counter()

    for r in rows:
        for i in range(7, len(r), 2):  # 从第8列开始，每一对文件名和标签
            label = r[i + 1].strip()
            if label == "human" or label == "":
            # if label == "":
                continue
            rank = species_rank.get(label, None)
            if rank is not None:
                species_data[label].append([r[i], rank])

    selected_data = []
    for species, data in species_data.items():
        selected_data.extend(data)
        selected_species_counts[species] += len(data)

    return selected_data, selected_species_counts

def main():
    print("开始处理数据...")

    with open(csv_file, "r") as fr:
        rows = list(csv.reader(fr))[1:]  # 移除表头，一次性读取数据存储于内存中

    # 根据读取的数据统计物种
    original_species_counts = count_allspecies_from_csv(rows)
    species_rank = create_species_rank(original_species_counts)

    print(species_rank)  # 查看species_rank是否为None或是否正确初始化

    # 处理数据并根据物种数量选择数据
    filtered_data, combined_counts = process_data(rows, original_species_counts, species_rank)

    # 分割数据集
    train_set, val_set, test_set = split_dataset(filtered_data, 0.8, 0.1)
    write_csv(train_file, train_set, ["filename", "label"])
    write_csv(val_file, val_set, ["filename", "label"])
    write_csv_all(train_all_file, train_set, ["filename", "label"])
    write_csv_all(val_all_file, val_set, ["filename", "label"])
    write_csv_all(test_file, test_set, ["filename", "label"])

    # 统计各个数据集中的物种数量
    train_counts = count_species_from_csv(train_file, species_rank)
    val_counts = count_species_from_csv(val_file, species_rank)
    train_all_counts = count_species_from_csv(train_all_file, species_rank)
    val_all_counts = count_species_from_csv(val_all_file, species_rank)
    test_counts = count_species_from_csv(test_file, species_rank)

    common_distributions = [
        [
            species, 
            train_counts.get(species, 0), 
            val_counts.get(species, 0), 
            test_counts.get(species, 0), 
            combined_counts[species]
        ] 
        for species in combined_counts
    ]

    all_distributions = [
        [
            species, 
            train_all_counts.get(species, 0), 
            val_all_counts.get(species, 0), 
            test_counts.get(species, 0), 
            combined_counts[species]
        ] 
        for species in combined_counts
    ]   

    # 按总数从大到小排序
    all_distributions.sort(key=lambda x: x[4], reverse=True)
    common_distributions.sort(key=lambda x: x[4], reverse=True)
    # 写入info文件
    write_csv_all(info_common_file, common_distributions, ["label", "train", "val", "test", "total"])
    write_csv_all(info_all_file, all_distributions, ["label", "train", "val", "test", "total"])

    if all(original_species_counts[species] == combined_counts[species] for species in original_species_counts):
        print("successful")
    else:
        # 打印前十个不匹配的物种
        mismatches = [(species, original_species_counts[species], combined_counts[species]) for species in original_species_counts if original_species_counts[species] != combined_counts[species]]
        # 按原始数量排序
        mismatches.sort(key=lambda x: x[1], reverse=True)  
        print("前十个不匹配的物种信息:")
        for mis in mismatches[:10]:
            print(f"物种: {mis[0]}, 原始数量: {mis[1]}, 合计数量: {mis[2]}")


if __name__ == "__main__":
    main()
