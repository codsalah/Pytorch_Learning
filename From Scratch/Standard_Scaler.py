from sklearn.preprocessing import StandardScaler
import numpy as np

######################################  Using SKLearn  #######################################

# Sample data
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data
scaler.fit(data)

# Transform the data
scaled_data = scaler.transform(data)

print("Original data:")
print(data)
print("\nScaled data:")
print(scaled_data)

######################################  From Scratch #######################################

class StandardScaler:
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

# Sample data
data2 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Initialize StandardScaler
scaler2 = StandardScaler()

# Fit the scaler to the data and transform it
scaled_data2 = scaler2.fit_transform(data2)

print("Original data:")
print(data2)
print("\nScratch Version => Scaled data:")
print(scaled_data2)
