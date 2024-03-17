def compute_cost(m, c, X, Y):
    total_error = 0
    for x, y_gt in zip(X, Y):
        y_pd = m * x + c
        err = y_gt - y_pd
        squared_err = err ** 2
        total_error += squared_err
    cost = total_error / len(Y)
    return cost

# Example usage:
X = [2, 3.5, 4, 10]
Y = [1, 6, 31, 51]
M = [0.5, 1, 1.5]  # Example list of slopes
C = [0, 5, 10]     # Example list of intercepts

best_cost = float("inf")
best_m = None
best_c = None

for m in M:
    for c in C:
        this_cost = compute_cost(m, c, X, Y)
        if this_cost < best_cost:
            best_cost = this_cost
            best_m = m
            best_c = c

print("Best Cost:", best_cost)
print("Best Slope (m):", best_m)
print("Best Intercept (c):", best_c)








