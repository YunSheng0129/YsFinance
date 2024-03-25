from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize


class BaseOptimizer(ABC):
 
    def __init__(self, mu, cov):
        assert len(mu) == len(cov)
        assert isinstance(mu, np.ndarray)
        assert isinstance(cov, np.ndarray)
        self.mu = mu
        self.cov = cov
        self.solution = None
        self.n_assets = len(self.mu)
    
    @abstractmethod
    def objective_function(self, weights):
        raise NotImplementedError("should implement in the derived class")

    @abstractmethod
    def constraints(self):
        raise NotImplementedError("should implement in the derived class")
    
    @abstractmethod
    def bounds(self):
        return tuple((0, 1) for _ in range(self.n_assets))

    def _optimize(self):
        self.solution = minimize(
            lambda weights: self.objective_function(weights), 
            self.initial_guess(), 
            method='SLSQP', 
            bounds=self.bounds(), 
            constraints=self.constraints()
            )

    def get_optimal_weights(self):
        return self.solution.x

    def initial_guess(self):
        init = np.random.uniform(0, 1, size=self.n_assets)
        return init / np.sum(init)
    
    def run(self):
        self._optimize()
        if self.solution.x is None:
            return "FAILED"
        else:
            return "SUCCESS"

    def __call__(self):
        self._optimize()
        if self.solution.x is None:
            return "FAILED"
        else:
            return "SUCCESS"
    

class MinVarPortfolio(BaseOptimizer):
    """
    最小方差
    """

    def __init__(self, mu, cov):
        super().__init__(mu, cov)
    
    def objective_function(self, weights):
        mu = self.mu
        cov = self.cov
        portfolio_mean = weights.T @ mu
        portfolio_variance = weights.T @ cov @ weights
        min_obj = portfolio_variance
        return min_obj

    def constraints(self):
        constraints_list = []


        return ({'type': 'eq', 
                 'fun': lambda weights: np.sum(weights) - 1})
    
    def bounds(self, lower_bound=0, upper_bound=1):
        bounds_list = []
        for i in range(self.n_assets):
            bounds_list.append((lower_bound, upper_bound))
        return tuple(bounds_list)


class SharpePortfolio(BaseOptimizer):
    """
    最大夏普
    """
    def __init__(self, mu, cov, target_return):
        super().__init__(mu, cov)
        self.target_return = target_return

    def objective_function(self, weights):
        portfolio_mean = weights.T @ self.mu
        portfolio_variance = weights.T @ self.cov @ weights
        max_obj = portfolio_mean/np.sqrt(portfolio_variance)
        return - max_obj

    def constraints(self):
        return ({'type': 'eq', 
                 'fun': lambda weights: np.sum(weights) - 1},
                {'type': 'eq',
                 'fun': lambda weights: weights.T @ self.mu - self.target_return})
    
    def bounds(self):
        bounds = []
        for i in range(self.n_assets):
            upper_bound = 1
            lower_bound = 0
            bounds.append((lower_bound, upper_bound))
        return tuple(bounds)
    

class MaxUtilityPortfolio(BaseOptimizer):
    """
    最大效用U=E(r)-0.5A \sigma^2
    penalty 为风险厌恶系数
    """
    def __init__(self, mu, cov, penalty=1):
        super().__init__(mu, cov)
        self.penalty = penalty

    def objective_function(self, weights):
        portfolio_mean = weights.T @ self.mu
        portfolio_variance = weights.T @ self.cov @ weights
        max_obj =  portfolio_mean - 0.5 * self.penalty * portfolio_variance
        return - max_obj

    def constraints(self):
        return ({'type': 'eq', 
                 'fun': lambda weights: np.sum(weights) - 1})
    
    def bounds(self):
        return tuple((0, 1) for _ in range(self.n_assets))


class CustomTypeA(BaseOptimizer):

    def __init__(self, 
        mu: np.ndarray, 
        cov: np.ndarray, 
        penalty: float, 
        scores: np.ndarray,
        component_weights: np.ndarray,
        industry_dummies: np.ndarray,
    ):
        """Custom type optimizer

        Parameters
        ----------
        mu : np.ndarray
            The mean returns
        cov : np.ndarray
            The covariance of returns
        penalty : float
            The Risk aversion coefficient
        scores : np.ndarray
            Factor Exposures
        component_weights : np.ndarray
            The weights of the components of an index
        industry_dummies : np.ndarray

        Raises
        ------
        None
        
        Notes
        -----
        None

        """
        super().__init__(mu, cov)
        self.penalty = penalty
        self.scores = scores
        self.component_weights = component_weights
        self.industry_dummies = industry_dummies
    
    def objective_function(self, omega):
        mu = self.mu
        cov = self.cov
        portfolio_mean = omega.T @ mu
        portfolio_variance = omega.T @ cov @ omega
        max_obj = omega.T @ self.scores - self.penalty * portfolio_variance
        return - max_obj

    def constraints(self):
        upper_limit = 0.1
        constraints_list = []

        def constraint1(omega):
            return np.sum(omega) - 1
        
        constraints_list.append({'type': 'eq', "fun": constraint1})

        for j in range(self.industry_dummies.shape[1]):
            industry_dummy = self.industry_dummies[:, j] 
            def aux_func(omega):
                result = np.abs(np.dot(omega - self.component_weights, industry_dummy)) - upper_limit
                return -result

            constraints_list.append({'type': 'ineq', "fun": aux_func})

        return constraints_list
    
    def bounds(self):
        result = []
        upper_bound = 0.02
        for i in range(self.n_assets):
            bound_i_up = self.component_weights[i] + upper_bound
            bound_i_down = max(self.component_weights[i] - upper_bound, 0)
            result.append((bound_i_down, bound_i_up))
        return tuple(result)
    


class RiskEva(BaseOptimizer):
    """风险平价"""
    def __init__(self, mu, cov):
        super().__init__(mu, cov)

    def objective_function(self, weights):
        vec = weights * (self.cov @ weights)
        vec = np.array([vec])
        return ((vec.T-vec)**2).sum().sum()
    
    def constraints(self):
        constraints_list = []


        return ({'type': 'eq', 
                 'fun': lambda weights: np.sum(weights) - 1})
    
    def bounds(self):
        return super().bounds()
    

class MaxMulti(BaseOptimizer):
    """最分散化"""
    def __init__(self, mu, cov, stds):
        super().__init__(mu, cov)
        if len(stds) != len(mu):
            raise ValueError('length of stds not map')
        self.stds = stds
    
    def objective_function(self, weights):
        num = weights @ self.stds
        den = weights.T @ self.cov @ weights
        return -num/den
    
    def constraints(self):
        constraints_list = []


        return ({'type': 'eq', 
                 'fun': lambda weights: np.sum(weights) - 1})
    
    def bounds(self):
        return super().bounds()
    

### 对角拼接矩阵
def concatenate_diag(matrix1,matrix2):

    result = np.zeros((matrix1.shape[0] + matrix2.shape[0], matrix1.shape[1] + matrix2.shape[1]))

    result[:matrix1.shape[0], :matrix1.shape[1]] = matrix1
    result[matrix1.shape[0]:, matrix1.shape[1]:] = matrix2
    return result


class BlackLitterman(BaseOptimizer):
    """
    Black-Litterman Model

    市场权重: w
    资产协方差矩阵: \Sigma
    风险厌恶系数: \lambda
    市场观点: \Pi = \mu + \epsilon_\Pi, \epsilon_\Pi \sim N(0, tau\Sigma)
            反马科维兹优化可求出市场均衡\mu_{market}=\Pi, tau为相信程度, 一般取贴近0的小值.
    主观观点: q = P\mu + \epsilon_q, \epsilon_q \sim N(0, \Omega)
            P为k*n矩阵,k为观点数, \Omega为对观点的相信程度,一般为对角阵.
    融合观点: y = X\mu + \epsilon.
        市场和主观纵向拼接
    线性回归求出\mu作为\mu_{bl}, 利用两个协方差矩阵计算\cov_{bl}



    Params:
    -cov: np.ndarray. 资产的协方差矩阵
    -penalty: Any. 风险厌恶系数
    -component_weights: np.ndarry. 市场权重向量
    -P: np.ndarry. 主观矩阵
    -q: np.ndarry. 主观向量
    -tau: Any. 市场偏差风险.
    -omega: Any. 主观偏差风险.

    """
    def __init__(self, cov, penalty,component_weights,P:np.ndarray,q:np.ndarray,tau=0.01,omega=0.05):
        self.n_assets = len(component_weights)
        assert isinstance(component_weights,np.ndarray)
        assert isinstance(P,np.ndarray)
        assert P.shape[1] == self.n_assets
        assert P.shape[0] == len(q)
        self.cov = cov
        self.P = P
        self.q = q
        self.k = len(q)  ## 观点个数
        self.tau = tau
        self.penalty = penalty
        self.component_weights = component_weights
        self.Omega = np.eye(self.k) * omega**2
        self.set_market_expect()
        self.set_subjective_expect()
        self.set_union_expect()
        ### 利用BL结果优化权重向量
        super().__init__(self.mu_bl, self.var_bl)

    def bounds(self):
        return super().bounds()
    
    def constraints(self):
        return ({'type': 'eq', 
                 'fun': lambda weights: np.sum(weights) - 1})
    
    def objective_function(self, weights):
        portfolio_mean = weights.T @ self.mu_bl
        portfolio_variance = weights.T @ self.var_bl @ weights
        max_obj =  portfolio_mean - 0.5 * self.penalty * portfolio_variance
        return - max_obj
    
    def set_market_expect(self):
        self.mu_market = self.penalty * self.cov @ self.component_weights

    def set_subjective_expect(self):
        pass

    def set_union_expect(self):
        self.y = np.concatenate((self.mu_market,self.q))
        self.X = np.vstack((np.eye(self.n_assets),self.P))
        self.V = concatenate_diag(self.tau * self.cov, self.Omega)
        ### 线性回归计算\mu_{bl}
        self.mu_bl, _, _, _ = np.linalg.lstsq(self.X,self.y,rcond=None)
        self.var_bl = np.linalg.inv(self.X.T @ np.linalg.inv(self.V) @ self.X)




    
    
        