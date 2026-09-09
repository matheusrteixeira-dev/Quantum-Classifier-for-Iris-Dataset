# Quantum Classifier with QSVM Method for Iris Dataset Classification

## Introduction

This repository contains a Python implementation of a quantum classifier using the QSVM method for classifying the Iris dataset. The code demonstrates how to train and evaluate a quantum classifier alongside a classical SVM classifier for comparison.

## Required Libraries

The following libraries are imported to implement the quantum classifier:

- `timeit`: For measuring the execution time of training the classical and quantum classifiers.
- `load_iris`: To load the Iris dataset.
- `MinMaxScaler`: For data normalization.
- `pd`: For data manipulation using pandas.
- `sns`: For data visualization using seaborn.
- `train_test_split`: To split the dataset into training and testing sets.
- `classification_report`, `accuracy_score`, `recall_score`, `f1_score`, `confusion_matrix`, `roc_curve`, `auc`, `roc_auc_score`, `multilabel_confusion_matrix`: To calculate evaluation metrics for the classifier.
- `SVC`: To implement the classical SVM classifier.
- `OneVsRestClassifier`: To create a multiclass classifier from the classical SVM classifier.
- `ZFeatureMap`, `RealAmplitudes`: For quantum data encoding.
- `COBYLA`: To choose the optimizer for training the quantum classifier.
- `Aer`: To configure the quantum simulator.
- `VQC`: To implement the quantum VQC classifier.
- `plt`: For visualization of graphs.
- `numpy`: For numerical operations.
- `cycle`: To create a cyclic iterator for colors in graphs.
- `PrettyTable`: To display formatted tables.

## Data Import

The Iris dataset is loaded using the `load_iris()` function from the `sklearn.datasets` library. The features are stored in a variable called `features`, and the labels are stored in a variable called `labels`.

## Data Normalization

The features of the dataset are normalized using the `MinMaxScaler` class from the `sklearn.preprocessing` library.

## Data Visualization

The code creates a DataFrame with the Iris dataset and plots a pair plot using the `seaborn` library to visualize the distribution of the data with different colors for each class.

## Data Splitting

The dataset is split into training and testing sets using the `train_test_split` function from the `sklearn.model_selection` library.

## Function to Train the Classical Classifier

A function called `train_classic_classifier()` is defined to train the classical SVM classifier. The classifier is created using the `OneVsRestClassifier` class and trained using the `fit()` method.

## Data Encoding

The code performs quantum data encoding using the `ZFeatureMap`, which is a first-order Pauli evolution quantum circuit. There is a commented example of using `ZZFeatureMap`, which is a second-order Pauli evolution quantum circuit with entanglement.

## Ansatz: The Variational Quantum Circuit

A variational quantum circuit named `var_circuit` is created with different numbers of repetitions (`reps`) of parametric gates (`RealAmplitudes`) depending on the number of features (qubits).

## Optimizer Choice

A `COBYLA` optimizer is created with a maximum number of iterations set to 100.

## Defining the Callback

A function named `callback_func` is created to store the values of iterations and costs during the training of the quantum classifier. It is used as an argument in the training of the VQC to track the training progress.

## Function to Train the Quantum Classifier

A function called `train_quantum_classifier()` is defined to train the quantum VQC classifier. The classifier is created using the `VQC` class, and the training is performed using the `fit()` method. The `train_quantum_classifier()` function also returns the trained classifier.

## Measuring the Execution Time

The execution time of training the classical and quantum classifiers is measured using the `timeit` library. The times are stored in the variables `time_classic` and `time_quantum`, respectively.

## Evaluating the Classifiers

Both the classical and quantum classifiers are evaluated on the training and testing sets using evaluation metrics such as accuracy, specificity, and classification report. Predictions are calculated using the test and training sets, and the metrics are printed in the output.

## Calculating ROC Curves for the Classifiers

Dictionaries are created to store the false positive rates (`fpr`), true positive rates (`tpr`), and areas under the ROC curve (`roc_auc`) for each class and for the micro-average. Metrics are calculated for both the classical and quantum classifiers on both the training and testing sets. The ROC curves are plotted for each class, as well as the micro-average, for the training and testing sets.

## Function to Calculate Specificity

A function called `specificity_score()` is defined to calculate the specificity of the classifier's predictions. The function uses the confusion matrix for each label and calculates the specificity for each class and returns the average.

## Metrics for the Classical and Quantum Classifiers

The evaluation metrics are printed for both the classical and quantum classifiers, including training time, classification report, accuracy, and specificity for both the training and testing sets.

## Plotting the ROC Curves

The ROC curves are plotted for both the classical and quantum classifiers on both the training and testing sets, along with the micro-average.

## Confusion Matrices for the Classical and Quantum Classifiers

The confusion matrices are plotted for both the classical and quantum classifiers on both the training and testing sets. Each confusion matrix is a table that shows the counts of true positives, true negatives, false positives, and false negatives for each class.

## Evaluating the Classifier's Behavior Based on Training Set Size

This section of the code conducts a study to evaluate the behavior of the classifier, both classical and quantum, based on the size of the training set. The objective is to understand how the classifiers' performance is affected by the size of the training dataset.

## Initialization of the Results Table

A table named `ptable` is initialized to store the evaluation results. It has columns for the training set size, model name (classical or quantum), training and testing accuracy, training and testing specificity, training and testing recall, training and testing F1-score, and training and testing area under the ROC curve (AUROC).

## Training Sizes

The variable `train_sizes` is defined as a list containing different proportions of the training dataset: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]. These values represent the fractions of the original dataset that will be used for training.

## Initialization of Dictionaries to Store Results

Four dictionaries are initialized to store the results of calculating the ROC curves (false positive rates, true positive rates, and areas under the ROC curve) for both models (classical and quantum) and for each training size. The dictionaries are:

- `fpr_dict_classic`: Stores the false positive rates for the classical model.
- `tpr_dict_classic`: Stores the true positive rates for the classical model.
- `roc_auc_dict_classic`: Stores the areas under the ROC curve for the classical model.
- `fpr_dict_quantum`: Stores the false positive rates for the quantum model.
- `tpr_dict_quantum`: Stores the true positive rates for the quantum model.
- `roc_auc_dict_quantum`: Stores the areas under the ROC curve for the quantum model.

## Loop for Each Training Size

A loop is executed for each training size defined in the `train_sizes` list. Inside the loop, the training dataset is split into training and testing sets using the `train_test_split()` function from the `sklearn.model_selection` library. The size of the training set is controlled by the specified proportion in the loop.

## Training and Predictions

For each training size, the classical SVM classifier (`svc_classico`) and the quantum VQC classifier (`vqc`) are trained with the current training set. Predictions are then made using the test and training sets.

## Label and Prediction Binarization

The classifier's predictions are binarized using the `label_binarize()` function from the `sklearn.preprocessing` library. This is necessary to calculate the ROC curves and the areas under the ROC curve for each class.

## Calculation of ROC Curves and Performance Metrics

For each training size and each model (classical and quantum), the ROC curves are calculated for each class using the `roc_curve()` and `auc()` functions from the `sklearn.metrics` library. Additionally, performance metrics such as accuracy, specificity, recall, and F1-score are calculated for both the training and testing sets.

## Updating the Results Table

The results of accuracy, specificity, recall, F1-score, and areas under the ROC curve for both models (classical and quantum) and for each training size are added to the `ptable`.

## Visualization of ROC Curves

The ROC curves are plotted for each class, as well as the micro-average, for both models (classical and quantum) and for each training size. The cycle of colors is used to distinguish the ROC curves of different training sizes.

## Printing the Results Table

The `ptable` is printed, showing the results of accuracy, specificity, recall, F1-score, and areas under the ROC curve for both models (classical and quantum) and for each training size. The table provides an overview of the classifiers' performance at different proportions of the training dataset.

