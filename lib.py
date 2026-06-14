import numpy as np
from scipy.linalg import cho_factor, cho_solve

class Portfolio:
    """
    Discrete-time optimal portfolio optimiser.
 
    Solves u* = (1/2) * (D^n)^{-1} * g^n
    where D^n is N*n x N*n and g^n is N*n.
    Decision variable: u = (u_0,...,u_{n-1}), u_n = 0 by convention.
    """
 
    _KERNELS = ('zero', 'exponential', 'power-law')
 
    def __init__(self, N: int, T: float, n: int):
        if N < 1: raise ValueError("N must be >= 1.")
        if T <= 0: raise ValueError("T must be > 0.")
        if n < 1: raise ValueError("n must be >= 1.")
        self.N = int(N)
        self.T = float(T)
        self.n = int(n)
        self.dt = T / n
        self.times = np.linspace(0.0, T, n + 1)
        self._Dn_factor = None
        self._params = {}
 
    def _phi(self, lag, scale, kernel):
        if kernel == 'zero':
            return np.zeros_like(np.asarray(lag, dtype=float))
        elif kernel == 'exponential':
            return np.where(lag > 0, np.exp(-scale * lag), 0.0)
        elif kernel == 'power-law':
            return np.where(lag > 0, np.maximum(lag, 1e-300) ** (-scale), 0.0)
 
    def _G(self, C, tk, tj, scale, kernel):
        return C * self._phi(tk - tj, scale, kernel)
 
    def _build_gn(self, E_dA):
        """
        Build g^n of size N*n.
 
        From the derivation (corrected formula):
            g_j = dt * sum_{k=j}^{n-1} E[Delta A_k]
                - gamma * dt * (T - t_j) * Sigma @ X0
                - varrho * dt * Pi @ X0
 
        The sum starts at k=j (not k=j+1) because X_{k+1} = X_0 + dt*sum_{l=0}^{k} u_l
        so u_j contributes to X_{k+1} for all k >= j.
 
        E_dA[k] = E[Delta A_k], shape (n, N). No outer dt.
        """
        N, n, dt = self.N, self.n, self.dt
        p = self._params
        g = np.zeros((n, N))
 
        for j in range(n):
            tj            = self.times[j]
            future        = dt * np.sum(E_dA[j:n], axis=0)   # sum_{k=j}^{n-1}
            risk_cost     = p['gamma']  * dt * (self.T - tj) * (p['Sigma'] @ p['X0'])
            terminal_cost = p['varrho'] * dt                 * (p['Pi']    @ p['X0'])
            g[j]          = future - risk_cost - terminal_cost
 
        return g.flatten() 
        
    
    def _build_Dn(self):
        """
        Build D^n of size N*n x N*n.
 
        Decision variable is u = (u_0,...,u_{n-1}).
        Grid: t_0,...,t_{n-1}  (n points, NOT including t_n=T).
 
        D^n = dt/2          * I_n x Lambda_bar          [temporary]
            + dt^2/2        * (Gn + Gn^T) x C           [transient]
            + gamma*dt^2/2  * W x Sigma                 [risk]  W_kj = min(T-tk, T-tj)
            + varrho*dt^2/2 * J_n x Pi                  [terminal]
        """
        N, n, dt = self.N, self.n, self.dt
        p  = self._params
        t  = self.times[:n]
        tk = t[:, None]
        tj = t[None, :]
 
        # Block 1: temporary
        Lbar = 0.5 * (p['Lambda'] + p['Lambda'].T)
        C = np.kron(np.eye(n), (dt / 2.0) * Lbar)
 
        # Block 2: transient
        Gn = self._phi(tk - tj, p['kernel_scale'], p['kernel'])
        B = np.kron((Gn + Gn.T) * (dt ** 2 / 2.0), p['C'])
 
        # Block 3: risk
        LL_T = np.minimum(self.T - tk, self.T - tj)/dt
        D1 = np.kron(LL_T * (p['gamma'] * dt**3 / 2.0), p['Sigma'])
 
        # Block 4: terminal  (J_n = ones(n,n))
        D2 = np.kron(np.ones((n, n)) * (p['varrho'] * dt ** 2 / 2.0), p['Pi'])
 
        Dn = B+C+D1+D2
        self._Dn_factor = cho_factor(Dn, check_finite=False)
        return Dn
 
 
    def _set_params(self, X0, Lambda, Sigma, gamma, C, varrho, Pi, kernel, kernel_scale):
        if kernel not in self._KERNELS:
            raise ValueError(f"kernel must be one of {self._KERNELS}, got '{kernel}'")
        if kernel == 'power-law' and not 0 < kernel_scale < 0.5:
            raise ValueError("kernel_scale must be in (0, 0.5) for power-law.")
        if kernel == 'exponential' and kernel_scale <= 0:
            raise ValueError("kernel_scale must be > 0 for exponential.")
        Pi_arr = np.eye(self.N) if Pi is None else np.asarray(Pi, float).reshape(self.N, self.N)
        self._params = {
            'X0':           np.asarray(X0,     float).reshape(self.N),
            'Lambda':       np.asarray(Lambda, float).reshape(self.N, self.N),
            'Sigma':        np.asarray(Sigma,  float).reshape(self.N, self.N),
            'gamma':        float(gamma),
            'C':            np.asarray(C,      float).reshape(self.N, self.N),
            'varrho':       float(varrho),
            'Pi':           Pi_arr,
            'kernel':       kernel,
            'kernel_scale': float(kernel_scale),
        }
 
    def build_optimal(self, X0, E_dA, Lambda, Sigma, gamma, C,varrho=0.0, Pi=None, kernel='power-law', kernel_scale=0.25):
        """
        Solve u* = (1/2)(D^n)^{-1} g^n.
 
        Parameters
        ----------
        E_dA : array-like, shape (n, N)
            E[Delta A_k] for k=0..n-1.
            For OU signal: E_dA[k] = exp(-beta*t_k)*I0*dt.
            For no signal: np.zeros((n, N)).
        """
        n    = self.n
        E_dA = np.asarray(E_dA, float).reshape(n, self.N)
 
        _structural = ('X0','Lambda','Sigma','gamma','C','varrho','Pi','kernel','kernel_scale')
        old = {k: self._params.get(k) for k in _structural}
        self._set_params(X0, Lambda, Sigma, gamma, C, varrho, Pi, kernel, kernel_scale)
        new = {k: self._params[k] for k in _structural}
 
        if self._Dn_factor is None or any(
                not np.array_equal(old[k], new[k]) for k in _structural):
            self._build_Dn()
 
        gn     = self._build_gn(E_dA)
        u_flat = 0.5 * cho_solve(self._Dn_factor, gn, check_finite=False)
        u_core = u_flat.reshape(n, self.N)
 
        u_star     = np.vstack([u_core, np.zeros((1, self.N))])
        X_star     = np.zeros((n + 1, self.N))
        X_star[0]  = self._params['X0']
        X_star[1:] = self._params['X0'] + np.cumsum(u_core, axis=0) * self.dt
 
        return u_star, X_star
    
    def build_Markowitz_optimal(self, Sigma:np.ndarray, I_t:np.ndarray, gamma:float) -> np.ndarray:
        X_star = np.zeros((self.n,self.N))
        #Sigma_inv = np.linalg.inv(Sigma)
        for k in range(self.n):
            #X_star[k,:] =  (Sigma_inv @ I_t[k,:]) / gamma 
            X_star[k,:] = np.linalg.solve(Sigma, I_t[k,:])/gamma
        return X_star
