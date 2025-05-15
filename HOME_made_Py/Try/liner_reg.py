import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the dataset
data = pd.read_csv(r"D:\MCA Sem-1\DBMS Lab\BattlegroundsMobileIndia_Setup_exe\Salary_dataset.csv")

print("First 5 rows of data:")
print(data.head())

# Prepare features (X) and target (Y)
X = data[['YearsExperience']]  # Must be 2D
Y = data['Salary']             # 1D target variable

print("\nFeature (YearsExperience):")
print(X.head())

print("\nTarget (Salary):")
print(Y.head())

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, Y)

# Predict the salary for existing data to plot regression line
Y_pred = model.predict(X)

# Print the slope and intercept
print("\nSlope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot the data points and regression line
plt.scatter(X, Y, color='blue', label='Actual Salary')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Salary vs Experience")
plt.legend()
plt.show()

# Take new input from user for prediction
# Take new input from user for prediction
new_exp = float(input("Enter years of experience to predict salary: "))
new_data = pd.DataFrame({'YearsExperience': [new_exp]})  # Fix warning by keeping column names
predicted_salary = model.predict(new_data)

print(f"Predicted Salary for {new_exp} years of experience: {predicted_salary[0]:.2f}")

