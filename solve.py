import numpy as np

# =========================
# MXFP4 configuration
# =========================

# quantization points q_i
# q = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
q = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

# # midpoints m_i (last endpoint is 6)
# m = np.array([
#     0,
#     0.25,   # (0 + 0.5)/2
#     0.75,   # (0.5 + 1)/2
#     1.25,   # (1 + 1.5)/2
#     1.75,   # (1.5 + 2)/2
#     2.5,    # (2 + 3)/2
#     3.5,    # (3 + 4)/2
#     5.0,    # (4 + 6)/2
#     6.0     # upper bound
# ])
m = np.array([0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.0])

# =========================
# Numerical integration
# =========================

def integrate_simpson(f, a, b, n=200):
    """
    Simpson integration of f over [a, b].
    n must be even.
    """
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (h / 3) * (
        y[0]
        + y[-1]
        + 4 * np.sum(y[1:-1:2])
        + 2 * np.sum(y[2:-2:2])
    )


# =========================
# Phi(alpha)
# =========================

def Phi(alpha):
    """
    Compute Phi(alpha) = sum_i ∫ (z - q_i * S)^2 e^{-z} dz
    """
    S = alpha / 6.0
    total = 0.0

    for i in range(len(q)):
        a = m[i] * S
        b = m[i + 1] * S if i + 1 < len(m) else m[i] * S

        def integrand(z):
            return (z - q[i] * S) ** 2 * np.exp(-z)

        total += integrate_simpson(integrand, a, b)

    return total


# =========================
# Numerical derivative
# =========================

def Phi_prime(alpha, h=1e-4):
    """
    Central finite difference for Phi'(alpha)
    """
    return (Phi(alpha + h) - Phi(alpha - h)) / (2 * h)


# =========================
# Root function g(alpha)
# =========================

def g(alpha):
    return Phi_prime(alpha) - 2.0 * np.exp(-alpha)


# =========================
# Bisection solver
# =========================

def solve_alpha_bisection(a=2.0, b=10.0, tol=1e-4, max_iter=100):
    ga = g(a)
    gb = g(b)

    if ga * gb > 0:
        raise ValueError("Root not bracketed. Try a wider interval.")

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        gc = g(c)

        if abs(b - a) < tol:
            return c

        if ga * gc <= 0:
            b = c
            gb = gc
        else:
            a = c
            ga = gc

    return 0.5 * (a + b)


# =========================
# Main
# =========================

if __name__ == "__main__":
    alpha_star = solve_alpha_bisection()
    print(f"Optimal alpha ≈ {alpha_star:.5f}")
