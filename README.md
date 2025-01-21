# An Automatic Identification Method of Common Species Based on Ensemble Learning

This experiment uses **AutoGluon** for the task of species classification. If you haven't installed **AutoGluon** yet, please go to the official website to download it and install it according to the installation steps: [AutoGluon Official Documentation](https://auto.gluon.ai/stable/index.html).

## Dataset Description

This experiment takes the **CC Dataset** as an example, and the task is species classification. The species in the dataset are divided into common species and rare species:

- **Common Species**: The first 9 species.
- **Rare Species**: The remaining species.

## Folder Structure

This project contains the following folders and files:

### 1. `Com/`
This folder contains the relevant code for the common species model:

- `train.py`: The code used to train the common species model.
- `test.py`: The code used to test the common species model.
- `calculate.py`: Generates the evaluation metrics and confusion matrix for the common species model.

### 2. `All/`
This folder contains the relevant code for the all-species model:

- `train.py`: The code used to train the all-species model.
- `test.py`: The code used to test the all-species model.
- `calculate.py`: Generates the evaluation metrics and confusion matrix for the all-species model.

### 3. `test/`
This folder contains one file: `ensemble.py`. This file implements the ensemble learning code and is responsible for performing the model evaluation in two stages and generating the relevant results. It will call `Com/calculate.py` and `All/calculate.py`, and generate the confusion matrix, evaluation metrics, and result files.

## Usage Instructions

### 1. Install Dependencies

Please ensure that **AutoGluon** and the relevant Python dependencies have been installed. If not, please refer to the [AutoGluon Official Documentation](https://auto.gluon.ai/stable/index.html) for installation.

### 2. Train the Model

Enter the corresponding folder (`Com/` or `All/`), and execute the following command to train the model:

```
python train.py
```

### 3. Test the Model

After the training is completed, use the following command to test the model and generate the evaluation results:

```
python test.py
```

### 4. Generate Evaluation Results

After the testing is completed, run the `ensemble.py` file in the `test` directory:

```
python ensemble.py
```

This operation will generate the confusion matrix, metrics file, and other results for each model, and save them in the `Com/` and `ALL/` folders.

## Note: In `calculate.py` and `ensemble.py`, parameters need to be changed according to the actual situation, such as the total number of species and the number of common species, etc.

Reference
[AutoGluon Official Documentation](https://auto.gluon.ai/stable/index.html) 
