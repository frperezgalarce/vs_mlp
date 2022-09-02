import pandas as pd
from sklearn import metrics
import utilities as ut
from sklearn.ensemble import RandomForestClassifier
import numpy as np
results = []
num_classes = 2

learning_rate = 0.005
samples = 2000

for n in [5000, 10000, 50000]:
    for t in range(30):
        train_dataset, test_dataset = ut.load_files(dataset=1)
        train_dataset = train_dataset.sample(n)  
        train_dataset = ut.sort_columns(train_dataset)
        test_dataset = ut.sort_columns(test_dataset)
        try:
            print(n)
            train_labels = train_dataset['label']
            test_labels = test_dataset['label']
            del train_dataset['label']
            del test_dataset['label']
            rf = RandomForestClassifier(n_estimators = 100)
            rf.fit(train_dataset, train_labels)
            print(train_dataset.columns)
            print(train_dataset.shape)

            predicted_labels = rf.predict_proba(test_dataset)
            predicted_train_labels = rf.predict_proba(train_dataset)
            train_labels = train_labels.replace('ClassA','0.0').replace('ClassB',"1.0")
            test_labels = test_labels.replace('ClassA','0.0').replace('ClassB',"1.0")
            y_train = train_labels.astype(float)
            y_test = test_labels.astype(float)
            y_pred_test = np.argmax(predicted_labels, axis=1)
            y_pred_train = np.argmax(predicted_train_labels, axis=1)

            acc_train = metrics.accuracy_score(y_train, y_pred_train)
            print(acc_train)
            acc_test = metrics.accuracy_score(y_test, y_pred_test)
            print(acc_test)
            recall_test = metrics.recall_score(y_test, y_pred_test)
            print(recall_test)
            recall_train = metrics.recall_score(y_train, y_pred_train)
            print(recall_train)
            f1_train = metrics.f1_score(y_train, y_pred_train)
            print(f1_train)
            f1_test = metrics.f1_score(y_test, y_pred_test)
            print(f1_test)
            roc_train = metrics.roc_auc_score(train_labels, predicted_train_labels[:,1])
            print(roc_train)
            roc_test = metrics.roc_auc_score(test_labels, predicted_labels[:,1])
            print(roc_test)
            results.append([acc_train, acc_test,recall_train, recall_test, f1_train, f1_test, roc_train, roc_test, n])
            pd.DataFrame(results, columns=['acc_train', 'acc_test','recall_train', 'recall_test','f1_train', 'f1_test', 
                                            'roc_train', 'roc_test', 'n',]).to_csv('test.csv')
        except Exception as error:
            print(error) 
