import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f_deriv, init_x, 
                     step_size=0.0001, 
                     precision=0.000001):
    current_x = init_x
    last_x = float('inf')  # initialize last_x with infinity
    x_list = [current_x]

    for _ in range(1000):
        print(current_x)
        while abs(current_x - last_x) > precision:
            last_x = current_x
            gradient = f_deriv(current_x)
            current_x -= gradient * step_size
            x_list.append(current_x)
    
    print(f'Minimum y exists at x {current_x}')
    return x_list


def trail1():
    def f(x):
        return 3 * x ** 2 + 4 * x + 7 
    
    def f_derivative(x):
        return 6 * x + 4
    
    func_name = 'Gradient descent on 3x^2 + 4x + 7'

    for initial_x in [-7.5, 5, -2/3]:  # Corrected typo here
        title = f'{func_name} starting from {initial_x}'
        x_list = gradient_descent(f_derivative, initial_x, step_size=0.01)
        visualize(f, -10, 10, x_list, title)


def visualize(f_func, range_start, range_end , x_list , plt_title):
    x = np.linspace(range_start, range_end, 50)
    y = f_func(x)

    plt.plot(x,y)
    plt.title(plt_title)
    plt.xlabel('x')
    plt.ylabel('y')

    for idx, xp in enumerate(x_list[::3]):
        yp = f_func(xp)
        color = 'ro' if idx%2 else 'bo'
        plt.plot(xp, yp, color)

    plt.show()  # Corrected: adding plt.show() to display the plot

# Call the main function
trail1()


