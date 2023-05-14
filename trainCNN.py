import glob
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Load MNIST dataset
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
                X_train.append(t)
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
                X_test.append(t)
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
# Build the model
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(21, 3)))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model
# Xác định các siêu tham số
epochs_list = [100,200]
batch_size_list = [32, 64]

best_accuracy = 0
best_model = None
# Lưu trữ thông số đánh giá
accuracy_list = []
precision_list = []
recall_list = []
model_save = []
# Đánh giá hiệu suất
for epochs in epochs_list:
    for batch_size in batch_size_list:
        trained_model = train_model(model, X_train, y_train, epochs, batch_size)
        model_save.append(trained_model)
        y_pred = trained_model.predict(X_test)
        y_pred = np.round(y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model

        print(f"Model with epochs={epochs}, batch_size={batch_size}:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("------------------------")

# In kết quả của mô hình tốt nhất
print("Best Model:")
print(best_model.summary())
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
# Lưu model
best_model.save('CNN_model.h5')