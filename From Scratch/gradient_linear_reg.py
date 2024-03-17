import numpy as np

# Initialize parameters
x = np.random.randn(10,1)
true_w = 2  # True weight
true_b = 3  # True bias
noise = np.random.rand(10, 1)  # Add noise
y = true_w * x + true_b + noise  # Calculate y with noise
# parameters
w = 0.0
b = 0.0
# Hyperparameter
learning_rate = 0.01

# Create gradient descent function
def gradient_desc(x, y, w, b, learning_rate):
    # 1. init parameters
    dldw = 0.0 # partial derivative of loss(l) with respect to weight(w)
    dldb = 0.0 # partial derivative of loss(l) with respect to bias(b)
    # 2. calculate the shape
    N = x.shape[0]  #shape of x

    # loop through each one of our samples inside data
    # loss = (y-(wx + b))**2
    for xi, yi in zip(x, y):
        dldw += -2 * xi*(yi- (w*xi+b))  #iteratively update w
        dldb += -2 * (yi- (w*xi+b))     #iteratively update b
    
    # 3. calculate total gradient then taking the average
    # Make an update to the w ,b Parameter
    w = w - learning_rate*(1/N) * dldw
    b = b - learning_rate*(1/N) * dldb
    return w, b

# Iterative updates
for epoch in range(500):
    w, b = gradient_desc(x, y, w, b, learning_rate)
    yhat = w * x + b
    loss = np.mean((y - yhat) ** 2)  # Calculate mean squared error (MSE)
    print(f"{epoch} loss is {loss}, ")