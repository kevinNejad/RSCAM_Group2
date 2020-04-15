import numpy as np
class AssetPrice:
    """
    This is a class for Asset Price ODE. 
    It contains methods for finding exact solution, plotting and exat mean hitting time.
    Arguments:
    a: Lower bound
    b: Upper bound
    mu: mu
    sigma: sigma
    """
    def __init__(self, a, b, mu, sigma):
        self.a = a
        self.b = b
        self.mu = mu
        self.sigma = sigma
        
    def f(self,x):
        """
        Function of f(x) in asset price stochastic differential equation (deterministic part)
        """
        return self.mu * x
    
    def g(self,x):
        """
        Function of g(x) in asset price stochastic differential equation (stochastic part)
        """
        return self.sigma * x
    
    def dg(self,x):
        """
        Function of g'(x) in asset price stochastic differential equation (stochastic part)
        """
        return self.sigma
    
    def df(self,x):
        return self.mu
  

class AssetPriceInterestRate:
    def __init__(self, lam, mu, sigma):
        self.lam = lam
        self.mu = mu
        self.sigma = sigma
        
    
    def f(self,x):
        return self.lam*(self.mu - x)
    
    def g(self,x):
        return self.sigma*np.sqrt(x)
    

class OpinionPolls:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def f(self,x):
        return -self.mu*(x/(1-x**2))
        
    def g(self,x):
        return self.sigma




class PopulationDynamic:
    def __init__(self, K, r, beta):
        self.K = K
        self.r = r
        self.beta = beta
        
    def f(self,x):
        return self.r*x*(self.K - x)
    
    def g(self,x):
        return self.beta*x



class EpidemicModel:
    
    def __init__(self, p, B, beta, alpha, rho, C):
        self.p = p
        self.B = B
        self.beta = beta
        self.alpha = alpha
        self.rho = rho 
        self.C = C
    
    def f(self, x):
        return (self.p -1)*self.B*x + (self.beta*self.C - self.alpha)*(1 - x)*x
    
    def g(self, x):
        return self.p*self.C*(1-x)*x



class PoliticalOpinion:
    def __init__(self, r, G, eps):
        self.r = r
        self.G = G
        self.eps = eps
        
    def f(self,x):
        return self.r*(self.G-x)
    
    def g(self,x):
        return np.sqrt(self.eps*x*(1-x))
    


class DoubleWellPotential:
    def __init__(self, sigma):
        self.sigma = sigma
        
    def f(self, x):
        return -8*x + 12*(x**2) - 4*(x**3)
    
    def g(self, x):
        return self.sigma
    
    def df(self, x):
        return -self._dV2(x)
    
    def dg(self, x):
        return 0



class SimpleSDE:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def f(self, x):
        return self.mu
    
    def g(self, x):
        return self.sigma
    
    def df(self, x):
        return 0
    
    def dg(self, x):
        return 0
    

class CustomSDE:
    def __init__(self):
        pass
    
    def f(self, x):
        return x - x**3
    
    def g(self, x):
        return np.sqrt(0.2)
    
    def df(self, x):
        return 1 - 3*(x**2)
    
    def dg(self, x):
        return 0

