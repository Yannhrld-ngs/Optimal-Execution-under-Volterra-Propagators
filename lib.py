import numpy as np
from scipy.linalg import cho_factor, cho_solve

class Portfolio:
    """
    Discrete-time optimal portfolio optimiser.

    Represents a portfolio of N assets over a horizon T discretised
    into n time steps. Trading parameters are passed at solve time,
    allowing the same object to be reused across scenarios.

    Parameters
    ----------
    N : int
        Number of assets.
    T : float
        Investment horizon.
    n : int
        Number of time steps.

    Usage
    -----
    >>> ptf = Portfolio(N=2, T=10.0, n=500)
    >>> result = ptf.build_optimal(
    ...     X0=[10.0, 0.0], mu=mu,
    ...     Lambda=0.03*np.eye(2),
    ...     Sigma=0.2*np.eye(2),
    ...     gamma=0.0,
    ...     C=np.array([[0.06, 0.05], [0.05, 0.06]]),
    ... )
    >>> u_star: trading speed # shape (n+1, N)
    >>> X_star: inventory       # shape (n+1, N)
    """

    _KERNELS = ('zero', 'exponential', 'power-law')

    def __init__(self, N: int, T: float, n: int):
        if N < 1:
            raise ValueError("N must be >= 1.")
        if T <= 0:
            raise ValueError("T must be > 0.")
        if n < 1:
            raise ValueError("n must be >= 1.")

        self.N = int(N)
        self.T = float(T)
        self.n = int(n)
        self.dt = T / n
        self.times = np.linspace(0.0, T, n + 1)

        # Internal cache
        self._Dn_factor = None
        self._params = {}

    # Protected methods
    def _phi(self, lag: np.ndarray, scale: float, kernel: str) -> np.ndarray:
        """
        Scalar decay kernel evaluated at lag = tk - tj.

        Parameters
        ----------
        lag    : float or ndarray — tk - tj, can be negative or zero
        scale  : float            — kernel scale parameter
        kernel : str              — 'zero', 'exponential', or 'power-law'

        Returns
        -------
        float or ndarray, same shape as lag
        """
        if kernel == 'zero':
            return np.zeros_like(np.asarray(lag, dtype=float))
        elif kernel == 'exponential':
            return np.where(lag > 0, np.exp(-scale * lag), 0.0)
        elif kernel == 'power-law':
            safe_lag = np.maximum(lag, 1e-300)
            return np.where(lag > 0, safe_lag ** (-scale), 0.0)

    def _G(self, C: np.ndarray, tk: float, tj: float,
           scale: float, kernel: str) -> np.ndarray:
        """
        Matrix-valued propagator G(tk, tj) = C * phi(tk - tj).

        Returns
        -------
        ndarray, shape (N, N)
        """
        return C * self._phi(tk - tj, scale, kernel)

    def _build_Dn(self) -> np.ndarray:
        """
        Build D^n of size N*n x N*n (decision variable is u_0..u_{n-1})

        D^n = (dt/2)          * I_{n+1}  x Lambda_bar          [temporary]
            + (dt^2/2)        * (LG+LG^T) x C                  [transient]
            + (gamma*dt^2/2)  * (L1+L1^T) x Sigma              [risk]
            + (varrho*dt^2/2) * J_n        x Pi                 [terminal]
        """
        N, n, dt = self.N, self.n, self.dt
        p      = self._params
        kernel = p['kernel']
        scale  = p['kernel_scale']
        t = self.times[:n]    
        tk = t[:, None]    
        tj = t[None, :]
        lag = tk - tj

        #  Block 1: temporary impact 
        Lambda_bar = 0.5 * (p['Lambda'] + p['Lambda'].T)
        D1 = np.kron(np.eye(n), (dt / 2.0) * Lambda_bar)

        #  Block 2: transient impact 
        Gn = self._phi(lag, scale, kernel)
        D2 = np.kron((Gn + Gn.T) * (dt ** 2 / 2.0), p['C'])

        #  Block 3: risk 
        L1 = np.where(tk > tj, self.T - tk, 0.0)
        D3 = np.kron((L1 + L1.T) * (p['gamma'] * dt ** 2 / 2.0), p['Sigma'])

        #  Block 4: terminal penalty
        L2 = np.ones((n, n))
        D4 = np.kron(L2 * (p['varrho'] * dt ** 2 / 2.0), p['Pi'])

        # Summing all
        D = D1 + D2 + D3 + D4
        self._Dn_factor = cho_factor(D, check_finite=False) # save cholesky decomposition
        return D

    def _build_gn(self, mu: np.ndarray) -> np.ndarray:
        """
        Build the linear signal vector g^n of size N(n+1).

        g_j = dt * sum_{k=j+1}^{n-1} mu_k * dt
             - gamma * (T - t_j) * Sigma @ X0
             - varrho * dt * Pi @ X0
        """
        N, n, dt = self.N, self.n, self.dt
        p = self._params
        g = np.zeros((n, N))

        for j in range(n):
            tj = self.times[j]
            future = (dt**2 * np.sum(mu[j + 1:n], axis=0) if j + 1 < n 
            else np.zeros(N))
            risk_cost = p['gamma']  * (self.T - tj) * (p['Sigma'] @ p['X0'])
            terminal_cost = p['varrho'] * (p['Pi'] @ p['X0'])
            g[j] = future - risk_cost - terminal_cost

        gn = (dt * g).flatten()
        return gn

    def _set_params(self, X0, Lambda, Sigma, gamma, C,
                    varrho, Pi, kernel, kernel_scale):
        """
        Validate and store structural parameters.

        Fix 5: single underscore (protected), not double underscore (private).
        """
        N = self.N
        if kernel not in self._KERNELS:
            raise ValueError(
                f"kernel must be one of {self._KERNELS}, got '{kernel}'"
            )
        if kernel == 'power-law' and not 0 < kernel_scale < 0.5:
            raise ValueError(
                "kernel_scale must be in (0, 0.5) for the power-law kernel."
            )
        if kernel == 'exponential' and kernel_scale <= 0:
            raise ValueError(
                "kernel_scale must be > 0 for the exponential kernel."
            )

        Pi_arr = (np.eye(N) if Pi is None
                  else np.asarray(Pi, float).reshape(N, N))

        self._params = {
            'X0':           np.asarray(X0,     float).reshape(N),
            'Lambda':       np.asarray(Lambda, float).reshape(N, N),
            'Sigma':        np.asarray(Sigma,  float).reshape(N, N),
            'gamma':        float(gamma),
            'C':            np.asarray(C,      float).reshape(N, N),
            'varrho':       float(varrho),
            'Pi':           Pi_arr,
            'kernel':       kernel,
            'kernel_scale': float(kernel_scale),
        }

    # Public method

    def build_optimal(
        self,
        X0:          np.ndarray,
        mu:          np.ndarray,
        Lambda:      np.ndarray,
        Sigma:       np.ndarray,
        gamma:       float,
        C:           np.ndarray,
        varrho:      float = 0.0,
        Pi                 = None,
        kernel:      str   = 'power-law',
        kernel_scale: float = 0.25,
    ) -> tuple :
        """
        Compute optimal trading speed, inventory path, and portfolio weights.

        Solves: u* = (1/2) * (D^n)^{-1} * g^n

        D^n is rebuilt only when structural parameters change. Calling
        this method again with the same model parameters but a different
        alpha signal reuses the cached Cholesky factorisation.

        Parameters
        ----------
        X0 : array-like, shape (N,)
            Initial portfolio (number of shares per asset).
        mu : array-like, shape (n+1, N)
            Deterministic alpha signal at each time step.
            Pass np.zeros((n+1, N)) for no-signal liquidation.
        Lambda : array-like, shape (N, N)
            Temporary impact matrix (positive definite, not necessarily
            symmetric). Diagonal: self-impact. Off-diagonal: cross-impact.
        Sigma : array-like, shape (N, N)
            Return covariance matrix (symmetric positive semidefinite).
        gamma : float
            Risk aversion coefficient (>= 0).
        C : array-like, shape (N, N)
            Cross-impact matrix (symmetric positive semidefinite).
        varrho : float, optional
            Terminal inventory penalty intensity (>= 0). Default 0.0.
        Pi : array-like, shape (N, N), optional
            Terminal penalty matrix. Defaults to identity.
        kernel : str, optional
            Decay kernel. One of 'zero', 'exponential', 'power-law'.
            Default: 'power-law'.
        kernel_scale : float, optional
            Scale parameter of the decay kernel.
            - Exponential: mean-reversion speed (> 0).
            - Power-law:   exponent in (0, 0.5).
            Default: 0.25.

        Returns
        -------
        Tuple
            trading_speed : ndarray, shape (n+1, N)
            inventory     : ndarray, shape (n+1, N)
        """
        n = self.n
        mu     = np.asarray(mu,float).reshape(n + 1, self.N)

        # Params saving for caching
        _structural = ('X0', 'Lambda', 'Sigma', 'gamma', 'C',
                       'varrho', 'Pi', 'kernel', 'kernel_scale')
        old = {k: self._params.get(k) for k in _structural}

        self._set_params(X0, Lambda, Sigma, gamma, C,
                         varrho, Pi, kernel, kernel_scale)

        new = {k: self._params[k] for k in _structural}

        needs_rebuild = self._Dn_factor is None or any(
            not np.array_equal(old[k], new[k]) for k in _structural
        )
        if needs_rebuild:
            self._build_Dn()

        # Solve u* = (1/2)(D^n)^{-1} g^n
        gn     = self._build_gn(mu)
        u_flat = 0.5 * cho_solve(self._Dn_factor, gn, check_finite=False) # Inversing using cholesky decomposition
        u_core = u_flat.reshape(n, self.N)    # u_0..u_{n-1}        
        u_star = np.vstack([u_core, np.zeros((1, self.N))]) # Append u_n = 0 (no trading at maturity)

        # Getting Inventory
        X_star      = np.zeros((n + 1, self.N))
        X_star[0]   = self._params['X0']
        X_star[1:]  = self._params['X0'] + np.cumsum(
            u_star[:-1], axis=0
        ) * self.dt

        return u_star, X_star