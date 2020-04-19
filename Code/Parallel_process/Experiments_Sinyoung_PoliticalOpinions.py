
# # Import Modules 




import Methods as M
import SDEs as S
import pickle
import numpy as np
import os


# # Initialise Parameters of Experiment




num_itr = 100000 # Number of iteration
timesteps = np.logspace(-3, 0, 19) # Values for dt



# Instance of Double-Well Potential SDE - The SDE used in Mannella's paper
SDE = S.PoliticalOpinion(r=1 , G=0.8 ,eps=0.4 )
a = 0.01
b = 0.99
X0 = 0.05


f = SDE.f
g = SDE.g
dg = SDE.dg
df = SDE.df
V = SDE.V


# # Choose Numerical Methods




EM_Mils = M.EM_Milstein() # Euler-Maryama and Milstein
EM_BC = M.EulerMaryamaBoundaryCheck() # Mannella
EXPV = M.ExponentialVTimestepping() # ExponentialV
EXP = M.ExponentialTimestepping()  # Exponential 

methods_dict = {'EM':EM_Mils.compute_MHT_EM,
                'Milstein':EM_Mils.compute_MHT_Milstein}


# # Run Expariment



for method, fun in methods_dict.items():

    t_exits = []
    steps_exits = []
    for dt in timesteps:
        t_exit,steps_exit = fun(X0=X0,dt=dt,num_itr=num_itr, f=f, g=g, df=df, dg=dg, V=V, a=a,b=b)
        t_exits.append(t_exit)
        steps_exits.append(steps_exit)
    
    paths = AT.paths if method=='AT' else None
    times = AT.times if method=='AT' else None
    ts = AT.timesteps if method=='AT' else None
    
    results_dic = {'SDE':'PolOp', 'Method':method, 'timesteps':timesteps,
               't_exits':t_exits, 'steps_exits':steps_exits,
              'AT_paths':paths, 'AT_time':times, 'AT_timesteps':ts}

    file_name = './Results/' + results_dic['SDE'] + '_' + results_dic['Method'] + '.pickle'
    if os.path.exists(file_name):
        raise Exception('WARNING-The file already exist - Change values of SDE and Methods -WARNING')


    with open(file_name, 'wb') as file:
        pickle.dump(results_dic, file)




# To load results, use the following command. 

with open('./Results/PolOp_Milstein.pickle', 'rb') as handle:
    res = pickle.load(handle)






