import numpy as np
import matplotlib.pyplot as plt

# Sample data: X and Y coordinates of the data points
x = np.array([1, 2, 3, 4, 5])  # Independent variable (input)
y = np.array([2, 4, 5, 4, 5])  # Dependent variable (output)

# Calculate the means of x and y
x_mean = np.mean(x)  # Mean of the x values
y_mean = np.mean(y)  # Mean of the y values

# Calculate coefficients for the regression line
# The slope (m) is calculated using the formula: m = Σ((x - x_mean) * (y - y_mean)) / Σ((x - x_mean)²)
numerator = np.sum((x - x_mean) * (y - y_mean))  # Covariance of x and y
denominator = np.sum((x - x_mean) ** 2)  # Variance of x
slope = numerator / denominator  # Slope of the regression line

# Calculate the y-intercept (b) using the formula: b = y_mean - m * x_mean
intercept = y_mean - slope * x_mean  # Intercept of the regression line

# Generate regression line values
# Create an array of x values for plotting the regression line
x_reg = np.linspace(min(x), max(x), 100)  # 100 points between the min and max of x
y_reg = slope * x_reg + intercept  # Calculate the corresponding y values for the regression line

# Plotting the data points and the regression line
plt.scatter(x, y, color='blue', label='Data Points')  # Plot the original data points
plt.plot(x_reg, y_reg, color='red', label='Regression Line')  # Plot the regression line
plt.title('Manual Linear Regression')  # Title of the plot
plt.xlabel('X')  # Label for the x-axis
plt.ylabel('Y')  # Label for the y-axis
plt.legend()  # Show legend
plt.grid()  # Show grid for better readability
plt.show()  # Display the plot