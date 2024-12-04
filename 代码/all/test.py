import os
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

def main():
    sevent_root = "/data/users/lhx/csv/CC_9/"
    img_root = '/data/datasets/'
    test_csv = "test.csv"
    test_csv_path = os.path.join(sevent_root, test_csv)
    test_df = pd.read_csv(test_csv_path)
    test_df['filename'] = test_df['filename'].apply(lambda x: img_root + x)

    # 加载标签
    labels = pd.read_csv(test_csv_path, usecols=[1]).iloc[:, 0]

    predictor = MultiModalPredictor.load("./model/")
    predictions = predictor.predict_proba(test_df)
    predictions = predictions.round(4)
    labels = pd.read_csv(test_csv_path, usecols=[1]).iloc[:, 0]
    predicted_labels = predictions.idxmax(axis=1)
    
    # 合并实际标签、预测标签和每个类的概率
    result = pd.DataFrame({
        'Actual': labels,
        'Predicted': predicted_labels
    }).join(predictions)

    print(result)
    
    result.to_csv('../test/all_results.csv', index=False)



if __name__ == '__main__':
    main()
