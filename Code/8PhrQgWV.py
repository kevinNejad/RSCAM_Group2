import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sympy.solvers.inequalities import solve_univariate_inequality
from sympy import Symbol, sin, Interval, S, sqrt
from sympy.abc import x
import sympy

from multiprocessing import Pool

class ExponentialTimestepping:
    def __init__(self):
        pass
    
    
    @staticmethod
    def F(Xn, f, g):
        return f(Xn) / g(Xn)**2
    
    @staticmethod
    def N(Xn,f, g, dt):
        return np.sqrt(((2*(1/dt)) / (g(Xn)**2)) + ExponentialTimestepping.F(Xn,f,g)**2)
    
    @staticmethod
    def worker(stuff):
        print(stuff)
        X0, f, g ,dt, df, dg, V, num_itr, a, b = stuff

        breaked = 0 
        tn = 0
        Xn = X0
        temp = X0
        steps = 0
        counter = 0
        while Xn > a and Xn < b:
            counter += 1
            steps +=1
            v = np.random.uniform()
            p = -np.log(v)
            u = np.random.uniform()
            Nt = ExponentialTimestepping.N(Xn,f,g, dt)
            Ft = ExponentialTimestepping.F(Xn,f,g)
            sign = np.sign(0.5*(1 + Ft/Nt) - u)  
            
            Xn_1 = Xn + (sign*p)/(Nt-sign*Ft)
            temp = Xn_1
            w = np.random.uniform()
            if Xn_1 < a or w < np.exp(-2*Nt*(min(Xn, Xn_1)-a)) or Xn_1 > b or w < np.exp(-2*Nt*(b-max(Xn, Xn_1))):
                breaked += 1
                break
                
            Xn = Xn_1

        return (steps * (dt),counter)


    @staticmethod
    def compute_MHT_EM(X0, f, g ,dt, df, dg, V, num_itr, a=None, b=None):
        if a is None and b is None:
            assert("Please provide a boundary value")  
        if a is None:
            a = -np.inf
        if b is None:
            b = np.inf
            
        
        pool = Pool()

        results = pool.map(ExponentialTimestepping.worker,((X0, f, g ,dt, df, dg, V, num_itr, a,b) for i in range(num_itr)),chunksize=2500)
        pool.close()
        t_exit = [x[0] for x in results]
        steps_exit = [x[1] for x in results]
        
        
        return t_exit, steps_exit
    
    def plot(self,t_exit):
        histogram,bins = np.histogram(t_exit,bins=20,range=[0,20])
        midx = (bins[0:-1]+bins[1:])/2
        plt.bar(midx,histogram,label='Test')
        plt.show()


class ExponentialVTimestepping:
    def __init__(self):
        pass

    @staticmethod
    def nu(g, dt):
        return np.sqrt(2*(1/dt) / g**2)
    
    @staticmethod
    def worker(stuff):
        X0, f, g ,dt, df, dg, V, num_itr, a, b = stuff

        breaked = 0 
        tn = 0
        Xn = X0
        steps = 0
        counter = 0
        nu = ExponentialVTimestepping.nu(g(Xn), dt)
        while Xn > a and Xn < b:
            counter += 1
            steps +=1
            v = np.random.uniform()
            p = -np.log(v)
            u = np.random.uniform()
            sign = np.sign(0.5*(1 + (1/nu)*(g(Xn)**(-2))*f(Xn)) - u)
            Xn_1 = Xn + (1/nu)*sign*(p - (g(Xn)**(-2))*(V(Xn + (1/nu)*sign*p) - V(Xn)))
                                
            w = np.random.uniform()
            if Xn_1 < a or w < np.exp(-2*nu*(min(Xn, Xn_1)-a)) or Xn_1 > b or w < np.exp(-2*nu*(b - max(Xn, Xn_1))):
                breaked += 1
                break
            
            Xn = Xn_1

        return (steps * (dt),counter)

    @staticmethod
    def compute_MHT_EM(X0, dt, f, g, df, dg ,V , num_itr, a=None, b=None):
            
        if a is None and b is None:
            assert("Please provide a boundary value")  
        if a is None:
            a = -np.inf
        if b is None:
            b = np.inf

        t_exit = []
        steps_exit = []
        
        pool = Pool()

        results = pool.map(ExponentialVTimestepping.worker,((X0, f, g ,dt, df, dg, V, num_itr, a,b) for i in range(num_itr)),chunksize=2500)
        pool.close()
        t_exit = [x[0] for x in results]
        steps_exit = [x[1] for x in results]
        
        
        return t_exit, steps_exit
    
    
    def plot(self,t_exit):
        histogram,bins = np.histogram(t_exit,bins=20,range=[0,20])
        midx = (bins[0:-1]+bins[1:])/2
        plt.bar(midx,histogram,label='Test')
        plt.show()


class EulerMaryamaBoundaryCheck:
    def __init__(self):
        self.breaked = 0
        self.thres_coeff = 5
        pass
    
#     def P_hit(self, Xn,Xn_1,dt,bound,D, df):
#         return np.exp(-df(bound)/(2*D*(np.exp(2*dt*df(bound))-1))*(Xn_1-bound+(Xn-bound)*np.exp(dt*df(bound))-f(bound)/df(bound))**2 + (bound - (Xn + dt*(f(Xn)+f(Xn_1))/2))**2/4*D*dt)
    
    @staticmethod
    def P_hit(x0,xh,dt,xb,D, f_dash, f):
        return np.exp(-f_dash(xb)/(2*D*(np.exp(2*dt*f_dash(xb))-1))*(xh-xb+(x0-xb)*np.exp(dt*f_dash(xb))-f(xb)/f_dash(xb))**2 + (xb - (x0 + dt*(f(x0)+f(xh))/2))**2/4*D*dt)
    
    @staticmethod
    def worker(stuff):
        X0, f, g ,dt, df, dg, V, num_itr, a, b,thres_coeff = stuff

        breaked = 0 
        tn = 0
        Xn = X0
        counter = 0
        while Xn > a and Xn < b:
            counter += 1
            Rn = np.random.randn(1)
            Xn_1 = Xn + dt*f(Xn) + np.sqrt(dt)*Rn*g(Xn)
            D = (g(Xn)**2)/2 
            
            if Xn-a<thres_coeff*dt or b-Xn<thres_coeff*dt:
                prob_a = EulerMaryamaBoundaryCheck.P_hit(Xn,Xn_1,dt,a,D, df, f)
                prob_b = EulerMaryamaBoundaryCheck.P_hit(Xn,Xn_1,dt,b,D, df, f)
                if prob_a > np.random.uniform(0,1) or prob_b > np.random.uniform(0,1):
                    breaked += 1
                    break
                    
            tn += dt
            Xn = Xn_1

        return (tn-0.5*dt,counter)



    def compute_MHT_EM(self, X0, dt, f, g, df, dg, V, num_itr, a=None, b=None):
        if a is None and b is None:
            assert("Please provide a boundary value")  
        if a is None:
            a = -10000
        if b is None:
            b = 10000
            
        pool = Pool()

        results = pool.map(EulerMaryamaBoundaryCheck.worker,((X0, f, g ,dt, df, dg, V, num_itr, a,b,self.thres_coeff) for i in range(num_itr)),chunksize=2500)
        pool.close()
        t_exit = [x[0] for x in results]
        steps_exit = [x[1] for x in results]

        return t_exit,steps_exit
    
    def plot(self,t_exit):
        histogram,bins = np.histogram(t_exit,bins=20,range=[0,20])
        midx = (bins[0:-1]+bins[1:])/2
        plt.bar(midx,histogram,label='Test')
        plt.show()



x = Symbol('x')
class AdaptiveTimestep:
    def __init__(self, zscore=0.6):
        self.zscore = zscore
        pass
    
    def find_min(self, sols):
        vals = []
        for sol in sols:
            if sol is not None:
                if sol.is_Union or sol.is_Intersection:
                    for arg in sol.args:
                        if arg.is_Interval:
                            vals.append(float(arg.args[1]))
                        elif arg.is_Union or arg.is_Intersection:
                            for a in arg.args:
                                vals.append(float(a.args[1]))
                            
                if sol.is_Interval:
                    vals.append(float(sol.args[1]))
        
        vals = [v for v in vals if str(v) != '-oo' or str(v) != 'oo']  
        
        return min(vals)
    
    def adapt_time_solver(self, b, a, Xn, fx, gx, dt):
        f = fx(Xn)
        g = gx(Xn)
        eps = np.arange(-self.zscore,self.zscore, 0.001)
        Xn_1dist = [Xn + f*dt + g*np.sqrt(dt)*p for p in eps]
        p_max = eps[np.argmax(Xn_1dist)]
        p_min = eps[np.argmin(Xn_1dist)]
        theta = 0.001

        sol1, sol2, sol3, sol4 = None, None, None, None 
        
        if (a < Xn + f*dt + g*np.sqrt(dt)*p_min) and (Xn + f*dt + g*np.sqrt(dt)*p_max < b):
            return dt
        
        if Xn + f*dt + g*np.sqrt(dt)*p_max > b:
            sol1=solve_univariate_inequality(f*x + g*p_max*sqrt(x) + Xn - b < 0, x, relational=False)
            
        
        if Xn + f*dt + g*np.sqrt(dt)*p_min < a:
            sol2 = solve_univariate_inequality(f*x + g*p_min*sqrt(x) + Xn - a > 0, x, relational=False)

        min_sol = self.find_min([sol1, sol2])
        dt_n = min(max(min_sol,theta), dt)

        return min(max(min_sol,theta), dt)
        
    
    def compute_MHT_EM(self, X0, dt, num_itr, f, g, df, dg, V, a, b):
        """
        Method that approxiamte a solution using Euler-Maruyama method
        
        Arguments:
        f: F(x)
        g: g(x)
        
        Return: List containing Mean, STD, Confidence interval Left, Confidence interval Right
        """
        
        self.paths = []
        self.times = []
        self.timesteps = []
    
        if a is None and b is None:
            assert("Please provide a boundary value")  
        if a is None:
            a = -1000
        if b is None:
            b = 1000
            
            
        # TODO: Add threshold for situation when the loop goes forever
        t_exit = []
        steps_exit = []
        
        
        adapt_timestep = self.adapt_time_solver
        for i in tqdm(range(num_itr)):
            X = X0
            t = 0
            path = []
            time = []
            timestep = []
            counter = 0
            while X > a and X < b:
                counter += 1
                dt_new_EM = adapt_timestep(b=b,a=a, Xn=X, fx=f, gx=g, dt=dt)
                dW = np.sqrt(dt_new_EM)*np.random.randn()
                X = X + dt_new_EM*f(X) + g(X)*dW
                t += dt_new_EM
                time.append(t)
                path.append(X)
                timestep.append(dt_new_EM)
            self.paths.append(path)
            self.times.append(time)
            self.timesteps.append(timestep)
            t_exit.append(t - 0.5 * dt_new_EM)
            steps_exit.append(counter)


        
        
        return t_exit, steps_exit
    


    def plot(self,t_exit):
        histogram,bins = np.histogram(t_exit,bins=20,range=[0,20])
        midx = (bins[0:-1]+bins[1:])/2
        plt.bar(midx,histogram,label='Test')
        plt.show()



class EM_Milstein:
    def __init__(self):
        pass
    
    @staticmethod
    def compute_MHT_EM(X0 , dt, num_itr, f, g, df, dg, V, a, b):
        
        if a is None and b is None:
            assert("Please provide a boundary value")  
        if a is None:
            a = -1000
        if b is None:
            b = 1000
        
        pool = Pool()

        results = pool.map(EM_Milstein.worker1,((X0, f, g ,dt, df, dg, V, num_itr, a,b) for i in range(num_itr)),chunksize=2500)
        pool.close()
        t_exit = [x[0] for x in results]
        steps_exit = [x[1] for x in results]

        return t_exit,steps_exit
    

    @staticmethod
    def worker1(stuff):
        X0, f, g ,dt, df, dg, V, num_itr, a, b = stuff

        X = X0
        t = 0
        counter = 0
        while X > a and X < b:
            counter += 1
            dW = np.sqrt(dt) * np.random.randn()
            X = X + dt*f(X) + g(X)*dW
        return (t-0.5*dt,counter)

    @staticmethod
    def worker2(stuff):
        X0, f, g ,dt, df, dg, V, num_itr, a, b = stuff

        X = X0
        t = 0
        counter = 0
        while X > a and X < b:
            counter += 1
            dW = np.sqrt(dt) * np.random.randn()
            X = X + dt*f(X) + g(X)*dW + 0.5 * g(X)*dg(X)*(dW**2 - dt)
            t += dt

        return (t-0.5*dt,counter)



    @staticmethod
    def compute_MHT_Milstein(X0, dt, num_itr, f, g, df, dg, V, a, b):
        
        if a is None and b is None:
            assert("Please provide a boundary value")  
        if a is None:
            a = -np.inf
        if b is None:
            b = np.inf
        
        pool = Pool()

        results = pool.map(EM_Milstein.worker2,((X0, f, g ,dt, df, dg, V, num_itr, a,b) for i in range(num_itr)))

        pool.close()
        t_exit = [x[0] for x in results]
        steps_exit = [x[1] for x in results]

        return t_exit,steps_exit
    
    
    def plot(self,t_exit):
        histogram,bins = np.histogram(t_exit,bins=20,range=[0,20])
        midx = (bins[0:-1]+bins[1:])/2
        plt.bar(midx,histogram,label='Test')
        plt.show()