import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Function to generate noisy linear data
def generate_data(a, b, points, noise_level):
    x = np.linspace(-10, 10, points)
    y = a * x + b + np.random.randn(points) * noise_level
    return x, y

print("Linear Line Fitter")

# Taking input from the user
a = float(input("Write the coefficient for the linear equation: "))
b = float(input("Write the constant for the linear equation: "))
points = int(input("Write the number of points to be generated: "))
noise_level = float(input("Write the noise level: "))

# Generating data
x_data, y_data = generate_data(a, b, points, noise_level)

# Creating the neural network model
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(64, activation='linear'))
model.add(Dense(1))

# Compiling and fitting the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_data, y_data, epochs=500, verbose=0)

# Predicting values for the plot
x_values = np.linspace(-10, 10, 400)
y_predicted = model.predict(x_values)

# Plotting the data points and the predicted line
plt.scatter(x_data, y_data, label='Data points', color='orange')
plt.plot(x_values, y_predicted, label='AI predicted line', color='blue')

plt.title("Linear Line Fitting with Neural Networks")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.show()
