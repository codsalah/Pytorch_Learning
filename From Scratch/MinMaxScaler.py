from sklearn.preprocessing import MinMaxScaler
import numpy as np

######################################  Using SKLearn  #######################################

# Sample data
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
data2 = np.array([[10, 25, 37],
                 [42, 55, 68],
                 [75, 600, 1000]])

# Print original data
print("Original data:")
print(data2)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to the data
scaler.fit(data2)

# Transform the data
scaled_data2 = scaler.transform(data2)

print("Original data:")
print(data2)
print("\nScaled data:")
print(scaled_data2)


######################################  From Scratch #######################################

def min_max_scaler(data, feature_range=(0, 1)):
    # Calculate the minimum and maximum values for each feature
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    # Perform Min-Max scaling
    scaled_data = (data - min_vals) / (max_vals - min_vals)

    # Scale to the specified feature range
    scaled_data = scaled_data * (feature_range[1] - feature_range[0]) + feature_range[0]

    return scaled_data

# Sample data
data3 = np.array([[10, 25, 37],
                 [42, 55, 68],
                 [75, 600, 1000]])

# Perform Min-Max scaling
scaled_data3 = min_max_scaler(data3)

# Print scaled data
print("\nOriginal data:")
print(data3)
print("\nScratch Version => Scaled data:")
print(scaled_data3)