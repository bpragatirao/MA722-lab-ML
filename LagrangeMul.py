def lagrange():
    # analytic solution
    x = 0.5
    y = 0.5
    f_val = x**2 + y**2
    return x, y, f_val

x, y, fval = lagrange()
print("Solution: x =", x, "y =", y)
print("Minimum value f(x,y) =", fval)
