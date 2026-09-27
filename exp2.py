import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('House Price Prediction Dataset.csv')

le = LabelEncoder()
df["Location"] = le.fit_transform(df["Location"])
df["Condition"] = le.fit_transform(df["Condition"])
df["Garage"] = le.fit_transform(df["Garage"])

x = df.drop(['Id', 'Price'], axis=1) 
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R2 Score:", score)
print("Mean Squared Error (MSE):", mse)
#print(df.head())


#correlation -- the relationship between the varriables in table
#corelation matrix 
#corelation is between -1 to +1
#+1 means strong positive correlation
#-1 means strong negative correlation
#0 means no correlation


df.corr(numeric_only=True)
print(df.corr())
