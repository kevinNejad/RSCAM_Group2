import Methods as M
import SDEs as S
import pickle
import numpy as np
import os

num_itr = 100000 # Number of iteration
timesteps = np.logspace(-3, 0, 19) # Values for dt

# Instance of  SDEs
SDE_EpiMod = S.EpidemicModel(p=10, B=1, beta=2, alpha=1, rho=1, C=2)
SDE_PolOpi = S.PoliticalOpinion(r=1 , G=0.8 ,eps=0.4 )
SDE_PopDyna = SDE = S.PopulationDynamic( K=1, r=1, beta=0.2)
SDE_AssetPrice = S.AssetPrice(mu=0.5, sigma=1)


SDEs_dict = {'EpiMod': {'a': 0.1, 'b':0.9, 'X0':0.13, 
'f':SDE_EpiMod.f, 'g':SDE_EpiMod.g, 'df':SDE_EpiMod.df, 'dg':SDE_EpiMod.dg, 'V':SDE_EpiMod.V},
'PolOpi': {'a': 0.01, 'b':0.99, 'X0':0.05, 
'f':SDE_PolOpi.f, 'g':SDE_PolOpi.g, 'df':SDE_PolOpi.df, 'dg':SDE_PolOpi.dg, 'V':SDE_PolOpi.V},
'PopDyna': {'a': 1, 'b':10, 'X0':9.95, 
'f':SDE_PopDyna.f, 'g':SDE_PopDyna.g, 'df':SDE_PopDyna.df, 'dg':SDE_PopDyna.dg, 'V':SDE_PopDyna.V},
'AssetPrice': {'a': 0.5, 'b':2, 'X0':1, 
'f':SDE_AssetPrice.f, 'g':SDE_AssetPrice.g, 'df':SDE_AssetPrice.df, 'dg':SDE_AssetPrice.dg, 'V':SDE_AssetPrice.V}}

AT = M.AdaptiveTimestep()

methods_dict = {'AT_EM':AT.compute_MHT_EM,
                'AT_Mils': AT.compute_MHT_Milstein}

for key, value in SDEs_dict.items():
	a, b, X0, f, g, df, dg, V = value.values()

	for method, fun in methods_dict.items():

	    t_exits = []
	    steps_exits = []
	    for dt in timesteps:
	        t_exit,steps_exit = fun(X0=X0,dt=dt,num_itr=num_itr, f=f, g=g, df=df, dg=dg, V=V, a=a,b=b)
	        t_exits.append(t_exit)
	        steps_exits.append(steps_exit)
	    
	#     paths = AT.paths if method=='AT' else None
	#     times = AT.times if method=='AT' else None
	#     ts = AT.timesteps if method=='AT' else None
	    paths, times, ts = None, None, None
	    
	    results_dic = {'SDE':key, 'Method':method, 'timesteps':timesteps,
	               't_exits':t_exits, 'steps_exits':steps_exits,
	              'AT_paths':paths, 'AT_time':times, 'AT_timesteps':ts}

	    file_name = './Results/' + results_dic['SDE'] + '_' + results_dic['Method'] + '.pickle'
	    if os.path.exists(file_name):
	        raise Exception('WARNING-The file already exist - Change values of SDE and Methods -WARNING')


	    with open(file_name, 'wb') as file:
	        pickle.dump(results_dic, file)