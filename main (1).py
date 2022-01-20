#relevant libraries 
import pandas as pd 
from sklearn import linear_model #ML
import matplotlib.pyplot as plt

#reads data
data_frame = pd.read_fwf('brain_body.txt')
x_values = data_frame[['Brain']]
y_values = data_frame[["Body"]]

#Trains model w/sklearn
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#Visualizes
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()