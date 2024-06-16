import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv("./Advertising.csv")
print(df.head())
print(df.info())
df = df.drop(["Unnamed: 0"],axis = 1)

df.dropna()
y = df["Sales"]
X = df.drop(["Sales"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)

#The trained model is stored in a pickle file 
pickle.dump(lr, open('model.pkl','wb') )

#The trained model is loaded into the variable model
model = pickle.load(open('model.pkl','rb'))
print(model.score(X_test, y_test))

#A random testing data point picked from the "Advertising.csv file in the folder"
new_data_point = pd.DataFrame([{
    'TV': 281.4,
    'Radio': 39.6,
    'Newspaper': 55.8
}])

#The outcome is printed.
print(model.predict(new_data_point))