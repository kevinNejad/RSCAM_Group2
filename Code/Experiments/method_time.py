import Methods as M
import SDEs as S
import pickle
import numpy as np
import os
import time

num_itr = 100000 # Number of iteration
timesteps = np.logspace(-3, 0, 19) # Values for dt

# Instance of  SDEs
SDE_EpiMod = S.EpidemicModel(p=10, B=1, beta=2, alpha=1, rho=1, C=2)
SDE_PolOpi = S.PoliticalOpinion(r=1 , G=0.8 ,eps=0.4 )
SDE_PopDyna = SDE = S.PopulationDynamic( K=1, r=1, beta=0.2)
SDE_AssetPrice = S.AssetPrice(mu=0.5, sigma=1)


SDEs_dict = {'Custom': {'a': 0, 'b':2, 'X0':0.1, 
'f':SDE_Custom.f, 'g':SDE_Custom.g, 'df':SDE_Custom.df, 'dg':SDE_Custom.dg, 'V':SDE_Custom.V}}

EM_Mils = M.EM_Milstein() # Euler-Maryama and Milstein
EM_BC = M.EulerMaryamaBoundaryCheck() # Mannella
EXPV = M.ExponentialVTimestepping() # ExponentialV
EXP = M.ExponentialTimestepping()  # Exponential 
AT = M.AdaptiveTimestep()

methods_dict = {'EM':EM_Mils.compute_MHT_EM,
                'EM_BC':EM_BC.compute_MHT_EM,
                'EXPV': EXPV.compute_MHT_EM,
		        'AT_EM':AT.compute_MHT_EM}

for key, value in SDEs_dict.items():
	a, b, X0, f, g, df, dg, V = value.values()

	for method, fun in methods_dict.items():

	    t_exits = []
	    steps_exits = []
	    elapsed_time = []
	    for dt in timesteps:
	    	start_t = time.time()
	        t_exit,steps_exit = fun(X0=X0,dt=dt,num_itr=num_itr, f=f, g=g, df=df, dg=dg, V=V, a=a,b=b)
	        t_exits.append(t_exit)
	        steps_exits.append(steps_exit)
	        elapsed_time.append(round(time.time() - start_t, 3))
	    
	#     paths = AT.paths if method=='AT' else None
	#     times = AT.times if method=='AT' else None
	#     ts = AT.timesteps if method=='AT' else None
	    paths, times, ts = None, None, None
	    
	    results_dic = {'SDE':key, 'Method':method, 'timesteps':timesteps, 'elapsed_time':elapsed_time,
	               't_exits':t_exits, 'steps_exits':steps_exits,
	              'AT_paths':paths, 'AT_time':times, 'AT_timesteps':ts}

	    file_name = './Results/' + results_dic['SDE'] + '_' + results_dic['Method'] + '.pickle'
	    if os.path.exists(file_name):
	        raise Exception('WARNING-The file already exist - Change values of SDE and Methods -WARNING')


	    with open(file_name, 'wb') as file:
	        pickle.dump(results_dic, file)