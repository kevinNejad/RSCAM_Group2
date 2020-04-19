# Instance of  SDEs
SDE_EpiMod = S.EpidemicModel(p=10, B=1, beta=2, alpha=1, rho=1, C=2)
SDE_PolOpi = S.PoliticalOpinion(r=1 , G=0.8 ,eps=0.4 )
SDE_PopDyna = SDE = S.PopulationDynamic( K=1, r=1, beta=0.2)
SDE_AssetPrice = S.AssetPrice(mu=0.5, sigma=1)
SDE_AssetPriceInt = S.AssetPriceInterestRate(lam=1, mu=0.5, sigma=0.3)
SDE_DoubleWell = .DoubleWellPotential(sigma=1.2)
SDE_OpiPoll = S.OpinionPolls(mu=5, sigma=1)
SDE_Custom = S.CustomSDE()

SDEs_dict = {'EpiMod': {'a': 0.1, 'b':0.9, 'X0':0.13, 
'f':SDE_EpiMod.f, 'g':SDE_EpiMod.g, 'df':SDE_EpiMod.df, 'dg':SDE_EpiMod.dg, 'V':SDE_EpiMod.V},
'SDE_PolOpi': {'a': 0.01, 'b':0.99, 'X0':0.05, 
'f':SDE_PolOpi.f, 'g':SDE_PolOpi.g, 'df':SDE_PolOpi.df, 'dg':SDE_PolOpi.dg, 'V':SDE_PolOpi.V},
'SDE_PopDyna': {'a': 1, 'b':10, 'X0':9.95, 
'f':SDE_PopDyna.f, 'g':SDE_PopDyna.g, 'df':SDE_PopDyna.df, 'dg':SDE_PopDyna.dg, 'V':SDE_PopDyna.V},
'SDE_AssetPrice': {'a': 0.5, 'b':2, 'X0':1, 
'f':SDE_AssetPrice.f, 'g':SDE_AssetPrice.g, 'df':SDE_AssetPrice.df, 'dg':SDE_AssetPrice.dg, 'V':SDE_AssetPrice.V},
'SDE_AssetPriceInt': {'a': 1, 'b':2, 'X0':1.9, 
'f':SDE_AssetPriceInt.f, 'g':SDE_AssetPriceInt.g, 'df':SDE_AssetPriceInt.df, 'dg':SDE_AssetPriceInt.dg, 'V':SDE_AssetPriceInt.V},
'SDE_DoubleWell': {'a': -0.5, 'b':1.5, 'X0':0.25, 
'f':SDE_DoubleWell.f, 'g':SDE_DoubleWell.g, 'df':SDE_DoubleWell.df, 'dg':SDE_DoubleWell.dg, 'V':SDE_DoubleWell.V},
'SDE_OpiPoll': {'a': -0.4, 'b':0.9, 'X0':0.4, 
'f':SDE_OpiPoll.f, 'g':SDE_OpiPoll.g, 'df':SDE_OpiPoll.df, 'dg':SDE_OpiPoll.dg, 'V':SDE_OpiPoll.V},
'SDE_Custom': {'a': 0, 'b':2, 'X0':0.1, 
'f':SDE_Custom.f, 'g':SDE_Custom.g, 'df':SDE_Custom.df, 'dg':SDE_Custom.dg, 'V':SDE_Custom.V}}