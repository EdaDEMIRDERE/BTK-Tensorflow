import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

dataframe = pd.read_excel("maliciousornot.xlsx")
print(dataframe.info())
print(dataframe.describe())
print(dataframe.corr()["Type"].sort_values())

plt.figure(figsize=(12, 8))
dataframe.corr()["Type"].sort_values().plot(kind="bar")
plt.show()
sns.countplot(x="Type", data=dataframe)
plt.show()

y = dataframe["Type"].values
x = dataframe.drop("Type", axis=1).values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# model build

#  print(X_train.shape)  (438, 30)

model = Sequential()
model.add(Dense(units=30, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(units=20, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=10, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=20)

model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=700, verbose=True, callbacks=[early_stopping])

model_loss = pd.DataFrame(model.history.history)
plt.title("Loss")
model_loss.plot()
plt.show()

preds = model.predict_classes(X_test)
print("classification report\n", classification_report(y_test, preds))

print("confusion matrix\n", confusion_matrix(y_test, preds))
