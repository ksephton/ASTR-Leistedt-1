# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:42:58 2020

@author: kvsep
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from numpy.linalg import inv
from numpy.linalg import slogdet
import pandas as pd
#%% Load full data set
data = np.load('spec4000error.npz')

#%% Calculating wavlength grid from coefficients

log_wavelengths = data['coeff0'] + data['coeff1'] * np.arange(0, 1000, 1)
wavelengths = [10**i for i in log_wavelengths]

#%% Selecting only galaxies (spec_cln = 2)

galaxy_ind = data['spec_cln'] == 2
X = data['spectra'][galaxy_ind]
subclass = data['lineindex_cln'][galaxy_ind]
z = data['z'][galaxy_ind]
z_err = data['zerr'][galaxy_ind]
spec_err = data['spec_err'][galaxy_ind]

#%% Set any negative spikes in flux to zero and set the corresponding error to zero

neg_ind = X <= 0.
X[neg_ind] = 0.
spec_err[neg_ind] = 0.

#%% Set zero fluxes to NaN

X_nonan = X.copy()
zero_ind = X == 0.
X[zero_ind] = np.NaN

#%% Set all zero flux errors to NaN

zero_err_ind = spec_err == 0.
spec_err[zero_err_ind] = np.NaN

#%% Normalise spectrum
X_normal, norm = preprocessing.normalize(X_nonan, return_norm=True)
X_norm_zeros = np.copy(X_normal)

#%% Plot an example spectrum in the data
plt.figure()
plt.plot(wavelengths,X_normal[4])
plt.show()

#%% Set all zero normalised fluxes to nan
zero_norm_ind = X_normal == 0.
X_normal[zero_norm_ind] = np.NaN

#%% Transform errors due to corresponding normalisation
spec_err_T = np.transpose(spec_err)
spec_err_norm_T = np.divide(spec_err_T,norm)
spec_err_norm = np.transpose(spec_err_norm_T)

#%% Spectra errors with infs instead of nans
df = pd.DataFrame(spec_err_norm)
df_inf = df.fillna(np.inf)
spec_err_norm_inf = df_inf.to_numpy()


#%% Plot mean spectrum
mu = np.nanmean(X_normal, axis=0)
std = np.nanstd(X_normal, axis=0)
#mu = X_norm_zeros.mean(0)
#std = X_norm_zeros.std(0)
mean_err = np.nanmean(spec_err_norm, axis=0)
plt.figure()
plt.plot(wavelengths, mu, color = 'black')
plt.fill_between(wavelengths, mu - std, mu + std , color = 'lightgrey')
plt.fill_between(wavelengths, mu - mean_err, mu + mean_err , color = 'grey')
plt.xlim(wavelengths[0], wavelengths[-1])
plt.ylim(0,0.06)
plt.xlabel('Wavelength [$\AA$]')
plt.ylabel('Scaled flux')
plt.title('Mean spectrum')
plt.show()

#%% Apply PCA
pca = PCA(n_components=1000)
X_red = pca.fit_transform(X_norm_zeros)

#%%
W = pca.components_
W_T = np.transpose(W)

#%%

#sigma_n = np.diagflat(spec_err_norm_inf[0]**2)
#sigma_n_nan = np.diagflat(spec_err_norm[0]**2)

sigma_n = np.diagflat(spec_err_norm_inf[n]**2)
sigma_n_nan = np.diagflat(spec_err_norm[n]**2)

sig_inv_n = inv(sigma_n)

M = np.identity(1000) + np.matmul(W_T,W) 
M_inv = inv(M)

#%%
C_n = np.matmul(W,W_T) + sigma_n_nan
C_inv_n = sig_inv_n -  np.matmul(sig_inv_n,np.matmul(W,np.matmul(M_inv,W_T)))

#%%
n= 0
l_n = -500*np.log(2*np.pi) - 0.5*np.log(np.abs(C_n[n])) - 0.5*np.transpose(X[n]-mu)*C_inv_n[n]*(X[n]-mu)

#%% Fill spectra minus mean to be filled with zeros at nan values

X_mu = X_normal - mu
df2 = pd.DataFrame(X_mu)
df2_zeros = df2.fillna(0.)
X_mu_zeros = df2_zeros.to_numpy()

#%%
n = 2001

sigma_n = np.diagflat(spec_err_norm_inf[n]**2)
sigma_n_nan = np.diagflat(spec_err_norm[n]**2)
sig_inv_n = inv(sigma_n)
    
W = pca.components_
W_T = np.transpose(W)
M = np.identity(np.shape(W)[0]) + np.matmul(W_T,np.matmul(sig_inv_n,W)) 

M_inv = inv(M)

C_n = np.matmul(W,W_T) + sigma_n
C_inv_n = sig_inv_n -  np.matmul(sig_inv_n,np.matmul(W,np.matmul(M_inv,np.matmul(W_T,sig_inv_n))))
   
sign_M, logdet_M = slogdet(M)
logdet_sig = np.nansum(np.log(spec_err_norm[n]**2))
    
l_n = -500*np.log(2*np.pi) - 0.5*(sign_M*logdet_M + logdet_sig) - 0.5*np.matmul(np.array([(X_mu_zeros[n])]),np.matmul(C_inv_n,(X_mu_zeros[n])))[0]
    
print(l_n)

#%% From identities

def Bayesian(n):
    sigma_n = np.diagflat(spec_err_norm_inf[n]**2)
    sigma_n_nan = np.diagflat(spec_err_norm[n]**2)
    sig_inv_n = inv(sigma_n)
    
    W = pca.components_
    W_T = np.transpose(W)
    M = np.identity(np.shape(W)[0]) + np.matmul(W_T,np.matmul(sig_inv_n,W)) 
    M_inv = inv(M)
    
    C_n = np.matmul(W,W_T) + sigma_n
    C_inv_n = sig_inv_n -  np.matmul(sig_inv_n,np.matmul(W,np.matmul(M_inv,np.matmul(W_T,sig_inv_n))))
    
    sign_M, logdet_M = slogdet(M)
    logdet_sig = np.nansum(np.log(spec_err_norm[n]**2))
    
    l_n = -500*np.log(2*np.pi) - 0.5*(sign_M*logdet_M + logdet_sig) - 0.5*np.matmul(np.array([(X_mu_zeros[n])]),np.matmul(C_inv_n,(X_mu_zeros[n])))[0]

    return l_n

