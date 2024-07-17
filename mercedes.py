import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error


dataframe = pd.read_excel("mercedes.xlsx")
print(dataframe.head())  # .describe()

print(dataframe.isnull().sum())

sbn.histplot(dataframe["price"])
plt.show()

#  sbn.countplot(dataframe["year"])  # yanlış çıktı bu ya
#  plt.show()


numeric_df = dataframe.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print(corr_matrix, "\n")

plt.figure(figsize=(10, 8))
sbn.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

print(numeric_df.corr()["price"].sort_values())

sbn.scatterplot(x="mileage", y="price", data=dataframe)
plt.show()

# True : most cheap
most_expensive_cars = dataframe.sort_values("price", ascending=False).head(30)
print(most_expensive_cars)

removed = len(dataframe) * 0.01  # 131

dataset = dataframe.sort_values("price", ascending=False).iloc[131:]
print(dataset)

plt.figure(figsize=(7, 5))
sbn.histplot(dataset["price"])
plt.show()

print("\n", dataset.groupby("year")["price"].mean())

cars_1970 = dataframe[dataframe.year == 1970]
print("\nCars at 1970:\n", cars_1970)

dataset = dataset[dataset.year != 1970]
print("\n", dataset.groupby("year")["price"].mean())

dataset = dataset.drop("transmission", axis=1)

# dataset labels - features
y = dataset["price"].values
x = dataset.drop("price", axis=1).values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
"böyle de yapılabiliyor"
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

model = Sequential()
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=250, epochs=300)

loss_data = pd.DataFrame(model.history.history)

loss_data.plot(figsize=(7, 5))

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahminler')
plt.title('Gerçek Değerler vs Tahminler')
plt.show()

new_car = dataset.drop("price", axis=1).iloc[2].values
new_car = scaler.transform(new_car.reshape(-1, 5))
print(model.predict(new_car))