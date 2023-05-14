from sklearn import svm
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Load dataset
label = ['moving1', 'leftclick', 'rolldown', 'zoombig', 'zoomsmall', 'rollup', 'rightclick']
X_train = []
y_train = []
X_test = []
y_test = []
for i in range(7):
    for data in glob.glob(label[i]+'.csv'):
        print(data)
        with open(data) as f:
            reader = csv.reader(f)
            for row in reader:
                t = []
                for xyz in row:
                    s = xyz.strip('[]')
                    arr = [float(x) for x in s.split()]
                    t.append(arr)
                t = np.array(t)
                flattened_data = t.flatten()
                X_train.append(flattened_data)
                y_train.append(i)
    for data in glob.glob(label[i]+'test.csv'):
        print(data)
        with open(data) as f:
            reader = csv.reader(f)
            for row in reader:
                t = []
                for xyz in row:
                    s = xyz.strip('[]')
                    arr = [float(x) for x in s.split()]
                    t.append(arr)
                t = np.array(t)
                flattened_data = t.flatten()
                X_test.append(flattened_data)
                y_test.append(i)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_true = np.zeros((3500, 7))
for i, label in enumerate(y_test):
    y_true[i, label] = 1
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(type(X_train[0]))
print(X_train[0].shape)
print(X_train[0])

x_train = X_train.reshape(-1, 3, 21, 1)
x_test = X_test.reshape(-1, 3, 21, 1)
print(y_train)
print(y_test)
# Xây dựng mô hình SVMs

# Chọn hàm kernel tùy chọn, ví dụ: linear, rbf, sigmoid, ...
def train_model( X_train, y_train, c):
    model = svm.SVC(kernel='linear', C=c)
    model.fit(X_train, y_train)
    return model

# Xác định các siêu tham số
c_list = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

best_accuracy = 0
best_model = None
# Lưu trữ thông số đánh giá
accuracy_list = []
precision_list = []
recall_list = []
model_save = []
# Đánh giá hiệu suất
for cc in c_list:
    trained_model = train_model(X_train, y_train, cc)
    model_save.append(trained_model)
    y_pred = trained_model.predict(X_test)
    y_pred = np.round(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = trained_model

    print(f"Model with c={cc}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("------------------------")

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(range(len(accuracy_list)), accuracy_list, label='Accuracy')
plt.plot(range(len(precision_list)), precision_list, label='Precision')
plt.plot(range(len(recall_list)), recall_list, label='Recall')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Evaluation Metrics')
plt.legend()
plt.xticks(range(len(accuracy_list)), rotation=90)
plt.show()
# Lưu mô hình
joblib.dump(best_model, 'SVMs_model.pkl')