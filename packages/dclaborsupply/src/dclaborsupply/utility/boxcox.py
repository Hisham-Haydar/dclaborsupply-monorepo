"""Box-Cox transform + derivatives (lifted, migration matrix Wave 1.3 / §F utility home).

Lifted byte-faithfully from MNL/scripts/enhanced/estimation_utils.py (the
Wave-0.2 R1-fixed version: box_cox_derivative_theta Taylor branch correct to
O(theta^4), `... + theta**3 * log_x3 / 15.0`). Copy + import adaptation only; no
math change. Zero MNL/old-repo imports.
"""
import numpy as np

# Optional Numba acceleration (NOT a base dependency). Pure-NumPy fallback when
# numba is absent, so `import dclaborsupply` stays light. Lifted verbatim from
# estimation_utils.py.
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    numba = None
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ==============================================================================
# Box-Cox Transformations
# ==============================================================================

if HAS_NUMBA:
    @numba.jit(nopython=True, fastmath=True)
    def box_cox_transform(x: np.ndarray, theta: float) -> np.ndarray:
        """
        Box-Cox transformation: BC(x; θ) = (x^θ - 1)/θ for θ≠0, log(x) for θ→0

        Uses limit approximation for |θ| < 1e-8:
            BC(x; 0) = log(x)

        Parameters
        ----------
        x : np.ndarray
            Input array (must be positive)
        theta : float
            Box-Cox exponent

        Returns
        -------
        np.ndarray
            Transformed values
        """
        if abs(theta) < 1e-8:
            # Limit: θ → 0
            return np.log(x)
        else:
            return (np.power(x, theta) - 1.0) / theta


    @numba.jit(nopython=True, fastmath=True)
    def box_cox_derivative_x(x: np.ndarray, theta: float) -> np.ndarray:
        """
        Derivative of Box-Cox w.r.t. x: ∂BC/∂x = x^(θ-1)

        Parameters
        ----------
        x : np.ndarray
            Input array (must be positive)
        theta : float
            Box-Cox exponent

        Returns
        -------
        np.ndarray
            Derivative values
        """
        return np.power(x, theta - 1.0)


    @numba.jit(nopython=True, fastmath=True)
    def box_cox_derivative_theta(x: np.ndarray, theta: float) -> np.ndarray:
        """
        Derivative of Box-Cox w.r.t. θ: ∂BC/∂θ with improved numerical stability

        For θ≠0:
            ∂BC/∂θ = (x^θ * log(x) * θ - (x^θ - 1)) / θ²

        For θ→0:
            ∂BC/∂θ|_{θ=0} = 0.5 * (log(x))²

        Uses Taylor expansion for |theta| < 0.05 to avoid catastrophic cancellation

        Parameters
        ----------
        x : np.ndarray
            Input array (must be positive)
        theta : float
            Box-Cox exponent

        Returns
        -------
        np.ndarray
            Derivative values
        """
        log_x = np.log(x)

        # Use Taylor expansion for |theta| < 0.05 to avoid numerical issues
        # Taylor: ∂BC/∂θ ≈ (log x)²/2 + θ(log x)³/6 + θ²(log x)⁴/24
        if abs(theta) < 0.05:
            log_x2 = log_x * log_x
            log_x3 = log_x2 * log_x
            # Third-order Taylor expansion for better accuracy
            return 0.5 * log_x2 * (1.0 + 2.0 * theta * log_x / 3.0 + theta * theta * log_x2 / 4.0 + theta ** 3 * log_x3 / 15.0)
        else:
            x_theta = np.power(x, theta)
            numerator = x_theta * (theta * log_x - 1.0) + 1.0
            return numerator / (theta * theta)

else:
    # NumPy fallback implementations (slower but functional)
    def box_cox_transform(x: np.ndarray, theta: float) -> np.ndarray:
        """Box-Cox transformation (NumPy implementation)"""
        if abs(theta) < 1e-8:
            return np.log(x)
        else:
            return (np.power(x, theta) - 1.0) / theta


    def box_cox_derivative_x(x: np.ndarray, theta: float) -> np.ndarray:
        """Derivative of Box-Cox w.r.t. x (NumPy implementation)"""
        return np.power(x, theta - 1.0)


    def box_cox_derivative_theta(x: np.ndarray, theta: float) -> np.ndarray:
        """Derivative of Box-Cox w.r.t. θ (NumPy implementation) with improved stability"""
        log_x = np.log(x)

        # Use Taylor expansion for |theta| < 0.05 to avoid numerical issues
        if abs(theta) < 0.05:
            log_x2 = log_x * log_x
            log_x3 = log_x2 * log_x
            return 0.5 * log_x2 * (1.0 + 2.0 * theta * log_x / 3.0 + theta * theta * log_x2 / 4.0 + theta ** 3 * log_x3 / 15.0)
        else:
            x_theta = np.power(x, theta)
            numerator = x_theta * (theta * log_x - 1.0) + 1.0
            return numerator / (theta * theta)
