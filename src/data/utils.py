import pandas as pd
import numpy as np
import scipy.stats

def realization(num, hist_Disc, hist_DH, value_dollar=True):
    mean = []
    Pg = []
    size = []
    label = []

    num = num
    count = 0

    while count < num:
        P90 = np.random.triangular(4,5,8)
        P10 = np.random.triangular(20,30,60)
        norm_mu = .5*(np.log(P10)+np.log(P90))
        norm_sig = (np.log(P10)-np.log(P90))/(1.2815*2)
    
        size.append(scipy.stats.lognorm.rvs(s = norm_sig, scale = np.exp(norm_mu), size=1).item())
        mean.append(np.exp(norm_mu+.5*norm_sig**2))
        Pg.append(hist_Disc.rvs(size=1).item())
        label.append(1)
        count+=1

    count = 0

    while count < num:
        P90 = np.random.triangular(4,5,8)
        P10 = np.random.triangular(20,30,60)
        norm_mu = .5*(np.log(P10)+np.log(P90))
        norm_sig = (np.log(P10)-np.log(P90))/(1.2815*2)
    
        size.append(scipy.stats.lognorm.rvs(s = norm_sig, scale = np.exp(norm_mu), size=1).item())
        mean.append(np.exp(norm_mu+.5*norm_sig**2))
        Pg.append(hist_DH.rvs(size=1).item())
        label.append(0)
        count+=1

    zippedList =  list(zip(mean, Pg, label, size))
    df_ML = pd.DataFrame(zippedList, columns = ['Mean' , 'Pg', 'Label','ActualSize']) 

    df_ML['RiskedPg'] = df_ML['Mean']* df_ML['Pg']
    df_ML['Acutal'] = df_ML['ActualSize']*df_ML['Label']
    df_ML['Value'] = df_ML['Acutal'].apply(lambda x: x*10 - 60)

    df_ML.sort_values(by='RiskedPg', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')
    df_ML.reset_index(inplace=True, drop=True)
    df_ML['CumResource']=df_ML['Acutal'].cumsum(axis = 0)
    df_ML['CumValue'] = df_ML['Value'].cumsum(axis=0)
    if value_dollar==True:
        out = df_ML['CumValue'].values
    else:
        out = df_ML['CumResource'].values
    return(out)

def portfolio_simulation(num_simulations, num_prospects, hist_discovery, hist_dryhole, value_dollar=False):
    q = realization(num_prospects, hist_Disc=hist_discovery, hist_DH=hist_dryhole, value_dollar=value_dollar)

    count = 0

    while count < num_simulations:
        q = q + realization(num_prospects, hist_Disc=hist_discovery, hist_DH=hist_dryhole, value_dollar=value_dollar)
        count +=1

    out = q/(num_simulations+1)

    return(out)

def kernel_generator(bins, bin_counts):  
    test = []
    for a, b in enumerate(zip(bins, bin_counts)):
        limit = b[1]
        count = 0
        while count < limit:
            test.append(b[0]+.01)
            count+=1
            #
    hist = np.histogram(test, bins=bins, density=True)
    hist_out = scipy.stats.rv_histogram(hist)
    return (hist_out)
