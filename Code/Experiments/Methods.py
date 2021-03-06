import numpy as np

from multiprocessing import Pool

class ExponentialTimestepping:
    def __init__(self):
        pass

    @staticmethod
    def F(Xn, f, g):
        return f(Xn) / g(Xn)**2
        
    @staticmethod
    def N(Xn,f, g, dt):
        return np.sqrt(((2*(1/dt)) / (g(Xn)**2)) + self.F(Xn,f,g)**2)

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
    
    def plot(t_exit):
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


    @staticmethod
    def compute_MHT_EM(self, X0, dt, f, g, df, dg, V, num_itr, a=None, b=None):
        if a is None and b is None:
            assert("Please provide a boundary value")  
        if a is None:
            a = -10000
        if b is None:
            b = 10000
            
        pool = Pool()

        results = pool.map(EulerMaryamaBoundaryCheck.worker,((X0, f, g ,dt, df, dg, V, num_itr, a,b,5) for i in range(num_itr)),chunksize=2500)
        pool.close()
        t_exit = [x[0] for x in results]
        steps_exit = [x[1] for x in results]

 
        return t_exit, steps_exit
    

class AdaptiveTimestep:
    def __init__(self):
        pass

    @staticmethod
    def solve_4_b(f,g,b,Xn, pmax, theta):
        C = b-Xn
        B = g*pmax
        if f == 0:
            return dt if B < 0 else (C/B)**2

        root2 = (C/f + (B**2)/(4*(f**2)))
        if root2 < 0:
            return theta

        if f > 0:
            return (np.sqrt(root2) - abs(B/(2*f)))**2

        return dt if B < 0 else (- np.sqrt(root2) - B/(2*f) )**2


    @staticmethod
    def solve_4_a(f,g,a,Xn, pmin, theta):
        D = a - Xn
        B = g*pmin

        if f == 0:
            return dt if B > 0 else (D/B)**2

        root2 = (D/f + (B**2)/(4*(f**2)))
        if root2 < 0:
            return theta

        if f > 0:
            return dt if B > 0 else (-np.sqrt(root2) - B/(2*f))**2

        return (np.sqrt(root2) - abs(B/(2*f)))**2

    @staticmethod
    def adapt_time_solver_EM(b, a, Xn, fx, gx, dt):
        f = fx(Xn)
        g = gx(Xn)
        eps = np.arange(-1,1, 0.1)
        Xn_1dist = [Xn + f*dt + g*np.sqrt(dt)*p for p in eps]
        p_max = eps[np.argmax(Xn_1dist)]
        p_min = eps[np.argmin(Xn_1dist)]
        theta = 0.0005

        maxdt1, maxdt2 = 0, 0 
        if (a < Xn + f*dt + g*np.sqrt(dt)*p_min) and (Xn + f*dt + g*np.sqrt(dt)*p_max < b):
            return dt
        
        if Xn + f*dt + g*np.sqrt(dt)*p_max > b:
            maxdt1 = AdaptiveTimestep.solve_4_b(f=f,g=g,b=b,Xn=Xn, pmax=p_max, theta=theta)
            
        
        if Xn + f*dt + g*np.sqrt(dt)*p_min < a:
            maxdt2 = AdaptiveTimestep.solve_4_a(f=f,g=g,a=a,Xn=Xn, pmin=p_min, theta=theta)

        return min(max(maxdt1, maxdt2,theta), dt)


    @staticmethod
    def adapt_time_solver_Milstein(b, a, Xn, fx, gx, dgx, dt):
        f = fx(Xn)
        g = gx(Xn)
        dg = dgx(Xn)
        eps = np.arange(-1,1, 0.1)
        Xn_1dist = [Xn + f*dt + g*np.sqrt(dt)*p + 0.5*g*dg*((np.sqrt(dt)*p)**2 - dt) for p in eps]
        p_max = eps[np.argmax(Xn_1dist)]
        p_min = eps[np.argmin(Xn_1dist)]
        theta = 0.0005

        maxdt1, maxdt2 = 0, 0 
        if (a < Xn + f*dt + g*np.sqrt(dt)*p_min + 0.5*g*dg*((np.sqrt(dt)*p_min)**2 - dt)) and (Xn + f*dt + g*np.sqrt(dt)*p_max + 0.5*g*dg*((np.sqrt(dt)*p_max)**2 - dt) < b):
            return dt
        
        if Xn + f*dt + g*np.sqrt(dt)*p_max + 0.5*g*dg*((np.sqrt(dt)*p_max)**2 - dt) > b:
            A = f + 0.5*g*dg*(p_max**2 - 1)
            maxdt1 = AdaptiveTimestep.solve_4_b(f=A,g=g,b=b,Xn=Xn, pmax=p_max, theta=theta)
            
        
        if Xn + f*dt + g*np.sqrt(dt)*p_min < a:
            A = f + 0.5*g*dg*(p_min**2 - 1)
            maxdt2 = AdaptiveTimestep.solve_4_a(f=A,g=g,a=a,Xn=Xn, pmin=p_min, theta=theta)

        return min(max(maxdt1, maxdt2,theta), dt)
        

    @staticmethod
    def worker1(stuff):
        X0, f, g ,dt, num_itr, a, b = stuff

        X = X0
        t = 0
        counter = 0
        while X > a and X < b:
            counter += 1
            dt_new_EM = AdaptiveTimestep.adapt_time_solver_EM(b=b,a=a, Xn=X, fx=f, gx=g, dt=dt)
            dW = np.sqrt(dt_new_EM)*np.random.randn()
            X = X + dt_new_EM*f(X) + g(X)*dW
            t += dt_new_EM
       
        return (t, counter)


    @staticmethod
    def worker2(stuff):
        X0, f, g, dg ,dt, num_itr, a, b = stuff

        X = X0
        t = 0
        counter = 0
        while X > a and X < b:
            counter += 1
            dt_new_Milstein = AdaptiveTimestep.adapt_time_solver_Milstein(b=b,a=a, Xn=X, fx=f, gx=g, dgx=dg, dt=dt)
            dW = np.sqrt(dt_new_Milstein)*np.random.randn()
            X = X + dt_new_Milstein*f(X) + g(X)*dW + 0.5*g(X)*dg(X)*(dW**2 - dt_new_Milstein)
            t += dt_new_Milstein
       
        return (t, counter)

    @staticmethod
    def compute_MHT_EM(X0, dt, num_itr, f, g, df, dg, V, a, b):
        """
        Method that approxiamte a solution using Euler-Maruyama method
        
        Arguments:
        f: F(x)
        g: g(x)
        
        Return: List containing Mean, STD, Confidence interval Left, Confidence interval Right
        """

    
        if a is None and b is None:
            assert("Please provide a boundary value")  
        if a is None:
            a = -1000
        if b is None:
            b = 1000
        

        pool = Pool(32)

        results = pool.map(AdaptiveTimestep.worker1, ((X0, f, g ,dt, num_itr, a, b) for i in range(num_itr)), chunksize=2500)
        pool.close()
        t_exit = [x[0] for x in results]
        steps_exit = [x[1] for x in results]
        
        
        return t_exit, steps_exit

    @staticmethod
    def compute_MHT_Milstein(X0, dt, num_itr, f, g, df, dg, V, a, b):
        """
        Method that approxiamte a solution using Milstein method
        
        Arguments:
        f: F(x)
        g: g(x)
        
        Return: List containing Mean, STD, Confidence interval Left, Confidence interval Right
        """
    
        if a is None and b is None:
            assert("Please provide a boundary value")  
        if a is None:
            a = -1000
        if b is None:
            b = 1000
            

        pool = Pool(32)

        results = pool.map(AdaptiveTimestep.worker2, ((X0, f, g, dg ,dt, num_itr, a, b) for i in range(num_itr)), chunksize=2500)
        pool.close()
        t_exit = [x[0] for x in results]
        steps_exit = [x[1] for x in results]
        
        return t_exit, steps_exit


class EM_Milstein:
    def __init__(self):
        pass


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
    
        return t_exit, steps_exit
    

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
         

        return t_exit, steps_exit
    
    
