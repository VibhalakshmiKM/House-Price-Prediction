import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Function to get user input for dataset
def get_user_data():
    num_houses = int(input("Enter number of houses: "))
    data = []
    prices = []
    for i in range(num_houses):
        sqft = float(input(f"Enter square footage for house {i+1}: "))
        bedrooms = int(input(f"Enter number of bedrooms for house {i+1}: "))
        age = int(input(f"Enter age of house {i+1}: "))
        bathrooms = float(input(f"Enter number of bathrooms for house {i+1}: "))
        garage = int(input(f"Enter number of garage spaces for house {i+1}: "))
        lot_size = float(input(f"Enter lot size (in square feet) for house {i+1}: "))
        price = float(input(f"Enter price of house {i+1}: "))
        data.append([sqft, bedrooms, age, bathrooms, garage, lot_size])
        prices.append(price)
    return np.array(data), np.array(prices)
# Get user data
X, y = get_user_data()
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict on test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Create subplots for visualizing all attributes
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
attributes = ['Square Footage', 'Bedrooms', 'Age', 'Bathrooms', 'Garage Spaces', 'Lot Size']
colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink']
for i in range(6):
    row, col = divmod(i, 3)
    axes[row, col].scatter(X[:, i], y, color=colors[i], label='Actual Data')
    axes[row, col].plot(X[:, i], model.predict(X), color='red', linestyle='--', label='RegressionLine')
    axes[row, col].set_xlabel(attributes[i])
    axes[row, col].set_ylabel('House Price')
    axes[row, col].set_title(f'{attributes[i]} vs. Price')
    axes[row, col].legend()
# Display the plots
plt.tight_layout()
plt.show()