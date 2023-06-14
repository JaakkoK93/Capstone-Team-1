import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sys

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler



# Read the kernel memory data from the input file

input_file = sys.argv[1]

kernel_memory_data = pd.read_csv(input_file, sep=';')



# Iterate through all columns and multiply column value 2 with column value 1

total_columns = kernel_memory_data.shape[1]

column1_selector = 1

column2_selector = 2

dynamic_column_number = 0

df_total_memory = pd.DataFrame()

rating_list = []

total_memory = 0

column_name = 8192



for i in range(total_columns // 2):

    column1 = kernel_memory_data.iloc[:, column1_selector]

    column2 = kernel_memory_data.iloc[:, column2_selector]

    column_timestamp = kernel_memory_data.iloc[:, 0]

    index_number = 0

    row_value = 0

    while index_number < len(kernel_memory_data.index):

        total_memory = column2.loc[row_value] * column1.loc[row_value]

        rating_list.append(total_memory)

        row_value += 1

        index_number += 1



    df_total_memory["Column_Timestamp"] = column_timestamp

    df_total_memory["Column_" + str(column1_selector + 1) + "_" + str(column2_selector + 1)] = rating_list

    column1_selector += 2

    column2_selector += 2

    dynamic_column_number += 1

    rating_list.clear()



df_extracted_sample_data = df_total_memory[['Column_Timestamp', 'Column_22_23']]

df_extracted_sample_data_total_mem = df_total_memory[['Column_22_23']]

df_total_mem = df_extracted_sample_data_total_mem.iloc[::5, :]



label_list = []

df_labels = pd.DataFrame()



# Define the gradual increase percentage and the threshold value

gradual_increase = 0.5  # 0.5% gradual increase

threshold_value = 1922912



for i in range(df_total_mem.shape[0]):

    first_row_selector = 0

    second_row_selector = 1

    index_number = 0

    while index_number < len(df_total_mem.index) - 1:

        if df_total_mem['Column_22_23'].iloc[second_row_selector] >= threshold_value:

            label_list.append(0)  # Assign label 0 (bad)

        else:

            percentage_calculation = df_total_mem['Column_22_23'].iloc[first_row_selector] * (1 + gradual_increase / 100)

            danger_zone = threshold_value - (threshold_value * 20) / 100

            if (

                df_total_mem['Column_22_23'].iloc[second_row_selector] > danger_zone

                and df_total_mem['Column_22_23'].iloc[second_row_selector] >= percentage_calculation

            ):

                label_list.append(0)  # Assign label 0 (bad)

            else:

                label_list.append(1)  # Assign label 1 (good)



        index_number += 1

        first_row_selector += 1

        second_row_selector += 1



    df_labels["Column_labels"] = label_list

    label_list.clear()



df_extracted_sample_data_timestamp = df_total_memory[['Column_Timestamp']]

df_timestamp_sorted = df_extracted_sample_data_timestamp.iloc[::5, :]

df_timestamp_sorted = df_timestamp_sorted.drop(df_timestamp_sorted.index[0])

df_total_mem = df_total_mem.drop(df_total_mem.index[0])



Col1292 = kernel_memory_data[['Column1292']]

Col1293 = kernel_memory_data[['Column1293']]

df_col_1292_1293 = pd.DataFrame()



for i in range(Col1292.shape[0]):

    col1292_sorted_values_by_time = Col1292.iloc[::5, :]

    col1293_sorted_values_by_time = Col1293.iloc[::5, :]



    df_col_1292_1293["Col1292"] = col1292_sorted_values_by_time

    df_col_1292_1293["Col1293"] = col1293_sorted_values_by_time



df_col_1292_1293 = df_col_1292_1293.drop(df_col_1292_1293.index[0])



if (

    len(df_timestamp_sorted) == len(df_total_mem) == len(df_col_1292_1293) == len(df_labels)

):

    print("")

    final_df = pd.concat(

        [

            df_timestamp_sorted.reset_index(drop=True),

            df_total_mem.reset_index(drop=True),

            df_col_1292_1293.reset_index(drop=True),

            df_labels.reset_index(drop=True),

        ],

        axis=1,

        join="inner",

    )

else:

    print("")



final_df.to_csv("clean_dataset.csv", index=False)



# Split the dataset into features (X) and labels (y)

X = final_df.drop("Column_labels", axis=1)

y = final_df["Column_labels"]







# Split the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Standardize the features

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)



# Train and evaluate the Decision Tree Classifier model

dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", accuracy_dt)



# Train and evaluate the Logistic Regression model

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

print("Logistic Regression Accuracy:", accuracy_lr)



# Train and evaluate the Support Vector Machine (SVM) model

svm_model = SVC(C=1.0, kernel='linear')

svm_model.fit(X_train_std, y_train)

y_pred_svm = svm_model.predict(X_test_std)

accuracy_svm = accuracy_score(y_test, y_pred_svm)

print("Support Vector Machine Accuracy:", accuracy_svm)



# Train and evaluate the Naive Bayes model

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)

print("Naive Bayes Accuracy:", accuracy_nb)



# Find the ML algorithm with the highest accuracy

accuracies = [accuracy_dt, accuracy_lr, accuracy_svm, accuracy_nb]

best_accuracy = max(accuracies)

best_model = None



if best_accuracy == accuracy_dt:

    best_model = dt_model

elif best_accuracy == accuracy_lr:

    best_model = lr_model

elif best_accuracy == accuracy_svm:

    best_model = svm_model

else:

    best_model = nb_model



print("")    

print("Best Model is: ", best_model)

print("Accuray is: " , best_accuracy * 100, "%")



# Visualize the anomalies detected by the best performing model

anomaly_indices = [i for i, label in enumerate(best_model.predict(X_test)) if label == 0]

anomaly_timestamps = X_test.iloc[anomaly_indices]['Column_Timestamp']

anomaly_values = X_test.iloc[anomaly_indices]['Column_22_23']



plt.figure(figsize=(10, 6))

plt.scatter(X_test['Column_Timestamp'], X_test['Column_22_23'], label='Normal')

plt.scatter(anomaly_timestamps, anomaly_values, color='r', label='Anomaly')

plt.xlabel('Timestamp')

plt.ylabel('Total Allocated Memory')

plt.title('Anomaly Detection - Best Performing Model: ' + str(best_model))

plt.legend()

plt.savefig('memroy_anomaly.png')  # Save the figure as an image

plt.show()