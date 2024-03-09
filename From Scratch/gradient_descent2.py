import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as LA

def gradient_descent(f_deriv_x, f_deriv_y, init_x, init_y, 
                     step_size=0.001, precision=0.000001):
    current_x = init_x
    current_y = init_y
    last_x = float('inf')  # initialize last_x with infinity
    last_y = float('inf')  # initialize last_y with infinity
    x_list = [current_x]
    y_list = [current_y]

    for _ in range(1000):
        gradient_x = f_deriv_x(current_x, current_y)
        gradient_y = f_deriv_y(current_x, current_y)
        next_x = current_x - gradient_x * step_size
        next_y = current_y - gradient_y * step_size
        if abs(next_x - current_x) < precision and abs(next_y - current_y) < precision:
            break
        current_x = next_x
        current_y = next_y
        x_list.append(current_x)
        y_list.append(current_y)
    
    print(f'Minimum z exists at x={current_x}, y={current_y}')
    return x_list, y_list



def visualize(f_func, range_start, range_end, x_list, y_list, plt_title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(range_start, range_end, 50)
    y = np.linspace(range_start, range_end, 50)
    X, Y = np.meshgrid(x, y)
    Z = f_func(X, Y)

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.plot(x_list, y_list, f_func(np.array(x_list), np.array(y_list)), color='r', marker='o')
    ax.set_title(plt_title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    try:
        inv_M = LA.inv(ax.get_proj())
    except LA.LinAlgError:
        # Singular matrix, handle appropriately (e.g., print an error message)
        print("Singular matrix encountered during visualization.")
        inv_M = np.eye(4)  # Identity matrix

    ax.format_coord = lambda x, y: format_coord(x, y, ax, inv_M)

    plt.show()

def format_coord(x, y, ax, inv_M):
    """
    Function to format coordinates on mouse hover.
    """
    xd, yd = ax.transData.transform((x, y))  # Data coordinates
    x, y, _ = ax.format_coord(x, y).split(", ")
    x, y, _ = float(x[2:]), float(y[2:]), 0  # Convert to float
    x, y, _ = ax.transData.inverted().transform((x, y, 0))  # Display coordinates
    x, y, _ = np.dot(inv_M, (x, y, 0, 1))[:-1]  # World coordinates
    return f'x={x}, y={y}'

def f1(x, y):
    return x ** 2 + y ** 2

def f1_derivative_x(x, y):
    return 2 * x

def f1_derivative_y(x, y):
    return 2 * y

def f2(x, y):
    return x ** 2 - y ** 2

def f2_derivative_x(x, y):
    return 2 * x

def f2_derivative_y(x, y):
    return -2 * y

def main():
    initial_x_values = [3, -4, 1]
    initial_y_values = [-5, 2, 0]
    
    for i in range(len(initial_x_values)):
        x_initial = initial_x_values[i]
        y_initial = initial_y_values[i]
        title = f'Gradient descent on x^2 + y^2, starting from x={x_initial}, y={y_initial}'
        x_list, y_list = gradient_descent(f1_derivative_x, f1_derivative_y, x_initial, y_initial)
        visualize(f1, -5, 5, x_list, y_list, title)

    for i in range(len(initial_x_values)):
        x_initial = initial_x_values[i]
        y_initial = initial_y_values[i]
        title = f'Gradient descent on x^2 - y^2, starting from x={x_initial}, y={y_initial}'
        x_list, y_list = gradient_descent(f2_derivative_x, f2_derivative_y, x_initial, y_initial)
        visualize(f2, -5, 5, x_list, y_list, title)

if __name__ == "__main__":
    main()
