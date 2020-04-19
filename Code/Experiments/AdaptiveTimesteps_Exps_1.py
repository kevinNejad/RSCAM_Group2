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
SDE_AssetPriceInt = S.AssetPriceInterestRate(lam=1, mu=0.5, sigma=0.3)
SDE_DoubleWell = S.DoubleWellPotential(sigma=1.2)
SDE_OpiPoll = S.OpinionPolls(mu=5, sigma=1)
SDE_Custom = S.CustomSDE()

SDEs_dict = {'AssetPriceInt': {'a': 1, 'b':2, 'X0':1.9, 
'f':SDE_AssetPriceInt.f, 'g':SDE_AssetPriceInt.g, 'df':SDE_AssetPriceInt.df, 'dg':SDE_AssetPriceInt.dg, 'V':SDE_AssetPriceInt.V},
'DoubleWell': {'a': -0.5, 'b':1.5, 'X0':0.25, 
'f':SDE_DoubleWell.f, 'g':SDE_DoubleWell.g, 'df':SDE_DoubleWell.df, 'dg':SDE_DoubleWell.dg, 'V':SDE_DoubleWell.V},
'OpiPoll': {'a': -0.4, 'b':0.9, 'X0':0.4, 
'f':SDE_OpiPoll.f, 'g':SDE_OpiPoll.g, 'df':SDE_OpiPoll.df, 'dg':SDE_OpiPoll.dg, 'V':SDE_OpiPoll.V},
'Custom': {'a': 0, 'b':2, 'X0':0.1, 
'f':SDE_Custom.f, 'g':SDE_Custom.g, 'df':SDE_Custom.df, 'dg':SDE_Custom.dg, 'V':SDE_Custom.V}}

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