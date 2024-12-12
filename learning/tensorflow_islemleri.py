import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

dataframe = pd.read_excel("bisiklet_fiyatlari.xlsx")
dataframe.describe()
dataframe.head()

sbn.pairplot(dataframe)
plt.show()

label_y = dataframe["Fiyat"].values  # values demezsek series olur dersek array olur
feature_x = dataframe[["BisikletOzellik1", "BisikletOzellik2"]].values

## split data as test/train ##
X_train, X_test, y_train, y_test = train_test_split(feature_x, label_y, test_size=0.33, random_state=15)

## scaling ##
"0 ile 1 arasına alınır değerler"
scaler = MinMaxScaler()
scaler.fit(X_train)  # X_train because the model must not know test data
#  If you scale the training and test data according to different rules,
#  you cannot accurately evaluate the performance of the model.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

## Model ##
model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))

model.add(Dense(1))  # output layer

model.compile(optimizer="rmsprop", loss="mse")
model.fit(X_train, y_train, epochs=250)

loss = model.history.history["loss"]
sbn.lineplot(x=range(len(loss)), y=loss)
plt.show()

train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print("train loss: ", train_loss)
print("test loss: ", test_loss)

test_pred = model.predict(X_test)
test_pred = pd.Series(test_pred.reshape(330, ))

y_df = pd.DataFrame(y_test, columns=["Actual Y "])
pred_df = pd.concat([y_df, test_pred], axis=1)
pred_df.columns = ["Actual Y", "Predicted Y"]
sbn.scatterplot(x="Actual Y", y="Predicted Y", data=pred_df)
plt.show()

mae = mean_absolute_error(pred_df["Actual Y"], pred_df["Predicted Y"])
print("MAE: ", mae)
mse = mean_squared_error(pred_df["Actual Y"], pred_df["Predicted Y"])
print("MSE: ", mse)

model.save("bisiklet_fiyatlari.keras")

##

new_b = [[1751, 1750]]
new_b = scaler.transform(new_b)
print("\n\nmodel1 : ", model.predict(new_b))

model_2 = load_model("bisiklet_fiyatlari.keras")
print("\nmodel2 : ", model_2.predict(new_b))
