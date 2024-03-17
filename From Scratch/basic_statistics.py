import numpy as np
import math

def mean(data):
    if len(data) ==0:
        return None
    return sum(data) / len(data) 

def mode(data):
    if len(data) ==0:
        return None
    freq = {}
    for value in data:
        freq[value] = freq.get(value, 0) + 1
    max_freq = max(freq.values())
    modes = [key for key, val in freq.items() if val == max_freq]

    return modes

def std(data):
    if len(data) == 0:
        return None
    mean_value = mean(data)
    variance = sum((x - mean_value) ** 2 for x in data) / len(data)
    return math.sqrt(variance)


data = np.array([1, 2, 3, 4, 5, 5, 6, 6, 6, 7])
print("Mean:", mean(data))
print("Mode:", mode(data))
print("Standard Deviation:", std(data))

############################# Verify Using Numpy ############################# 
mode = np.argmax(np.bincount(data))

print("NumPy Mean:", np.mean(data))
print("NumPy Mode:", mode)  
print("NumPy Standard Deviation:", np.std(data))