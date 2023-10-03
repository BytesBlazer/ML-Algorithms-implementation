import numpy as np
import pandas as pd
from KNN import KNN

data = pd.read_csv("iris.csv")
data = data.sample(frac=1)

split_size = int(len(data) * 0.8)
x_train, y_train = np.array(data.iloc[:split_size, :-1]), np.array(data.iloc[:split_size, -1])
x_test, y_test = np.array(data.iloc[split_size:, :-1]), np.array(data.iloc[split_size:, -1])

clf = KNN(k=5)
clf.fit(x_train, y_train)
predictions = clf.predictions(x_test)

acc = np.sum(predictions == y_test) / len(y_test)
print("Accuracy: ", acc)