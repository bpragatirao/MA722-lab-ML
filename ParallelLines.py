import math

def derive_a_b(reg_str: str):
    if not reg_str.isdigit() or len(reg_str) < 2:
        print("Register number must be a numeric string with at least 2 digits.")
    a = int(reg_str[-1])
    b = int(reg_str[-2])
    if a == 0:
        a = 1
    if b == 0:
        b = 1
    return a, b

def dist_btw_lines(a: float, b: float, c1: float, c2: float):
    """Compute distance between ax+by+c1=0 and ax+by+c2=0"""
    denom = math.hypot(a, b)  # sqrt(a^2 + b^2)
    if denom == 0:
        print("Coefficients a and b cannot both be zero.")
    return abs(c2 - c1) / denom


print("Distance between two parallel lines ax+by+c1=0 and ax+by+c2=0")
reg = input("Enter your Register Number ").strip()
a, b = derive_a_b(reg)
print(f"Derived coefficients: a = {a}, b = {b} ")

use_defaults = input("Use these a,b? (Y/n): ").strip().lower()
if use_defaults == 'n':
    # optional manual override; but we'll validate parallelism
    print("Enter coefficients for first line (a1, b1) and second line (a2, b2).")
    a1 = float(input("a1: ").strip())
    b1 = float(input("b1: ").strip())
    a2 = float(input("a2: ").strip())
    b2 = float(input("b2: ").strip())
    if not (math.isclose(a1, a2) and math.isclose(b1, b2)):
        raise ValueError("The two lines are not parallel: coefficients (a,b) must be the same for both lines.")
    a, b = a1, b1
# read c1 and c2
c1 = float(input("Enter c1 (for first line ax+by+c1=0): ").strip())
c2 = float(input("Enter c2 (for second line ax+by+c2=0): ").strip())

dist = dist_btw_lines(a, b, c1, c2)
print(f"\nDistance between the lines: {dist:.6f}")
# show a quick worked example check
print(f"(Computed as |{c2} - {c1}| / sqrt({a}^2 + {b}^2) = {abs(c2-c1):.6f} / {math.hypot(a,b):.6f})")