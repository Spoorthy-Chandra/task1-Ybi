#code for creating data set

import pandas as pd
import numpy as np
np.random.seed(0)
temperature = np.random.uniform(10, 35, 100)  
marketing_rupees = np.random.uniform(5000, 50000, 100)  
revenue = 1000 + 25 * temperature + 0.01 * marketing_rupees + np.random.normal(0, 100, 100)
data = pd.DataFrame({'Temperature_C': temperature, 'Marketing_Rupees': marketing_rupees, 'Revenue': revenue})
data.to_csv('icecream_sales_data.csv', index=False)

#original code 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
np.random.seed(0)
temperature = np.random.uniform(10, 35, 100)  
marketing_rupees = np.random.uniform(5000, 50000, 100) 
revenue = 1000 + 25 * temperature + 0.01 * marketing_rupees + np.random.normal(0, 100, 100)  
data = pd.DataFrame({'Temperature_C': temperature, 'Marketing_Rupees': marketing_rupees, 'Revenue': revenue})
print(data.head()) 
print(data.info())
print(data.describe())

X = data[['Temperature_C', 'Marketing_Rupees']]  
y = data['Revenue']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.show()

