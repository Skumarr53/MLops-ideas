import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file into a pandas DataFrame
file_path = 'path_to_your_file.csv'
data = pd.read_csv(file_path)

# Generate the overall confusion matrix
overall_conf_matrix = confusion_matrix(data['label_GT'], data['predictions'])

# Generate confusion matrices for each individual topic (unique sentence2)
topic_conf_matrices = {}
for topic in data['sentence2'].unique():
    topic_data = data[data['sentence2'] == topic]
    topic_conf_matrix = confusion_matrix(topic_data['label_GT'], topic_data['predictions'])
    topic_conf_matrices[topic] = topic_conf_matrix

# Helper function to display confusion matrices
def plot_conf_matrix(matrix, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Entailment', 'Entailment'], yticklabels=['Not Entailment', 'Entailment'])
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Display the overall confusion matrix
plot_conf_matrix(overall_conf_matrix, 'Overall Confusion Matrix')

# Display confusion matrices for each topic
for topic, matrix in topic_conf_matrices.items():
    plot_conf_matrix(matrix, f'Confusion Matrix for Topic: {topic}')



# A confusion matrix is a performance measurement tool for classification models. It compares the actual outcomes (ground truth) with the predicted outcomes. It provides the following values:

# True Positives (TP): Correctly predicted positive cases (Entailment).
# True Negatives (TN): Correctly predicted negative cases (Not Entailment).
# False Positives (FP): Incorrectly predicted positive cases (predicted Entailment when it should be Not Entailment).
# False Negatives (FN): Incorrectly predicted negative cases (predicted Not Entailment when it should be Entailment).

# ### Interpretation:

High TP and high TN indicate good model performance.
Low FP and low FN indicate fewer mistakes.
The matrix helps to identify areas where the model needs improvement, like incorrectly predicting Entailment or Not Entailment.

# This summary allows stakeholders to quickly grasp the accuracy and reliability of the model's predictions.
# Based on the confusion matrices, the model generally performs well with high accuracy in predicting entailment and non-entailment labels across topics. 