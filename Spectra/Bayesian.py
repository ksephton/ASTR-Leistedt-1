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
from scipy.optimize import minimize

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

#%% Set all zero and negative flux errors to NaN

zero_err_ind = spec_err <= 0.
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

# %% Cap errors at a lower limit of 1e-5 times the flux
cap_counter = 0
for spectra in range(len(spec_err_norm)):
    for pixel in range(len(spec_err_norm[spectra])):
        if np.isnan(spec_err_norm[spectra][pixel]) == False and spec_err_norm[spectra][pixel] < 1e-5 * X_normal[spectra][pixel]:
            spec_err_norm[spectra][pixel] = 1e-5 * X_normal[spectra][pixel]
            cap_counter += 1
print("Number of capped errors", cap_counter)

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
logdet_sig = np.nansum(np.log(spec_err_norm[n]**(-2)))
    
l_n = -500*np.log(2*np.pi) - 0.5*(sign_M*logdet_M - logdet_sig) - 0.5*np.matmul(np.array([(X_mu_zeros[n])]),np.matmul(C_inv_n,(X_mu_zeros[n])))[0]
    
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
    logdet_sig = np.nansum(np.log(spec_err_norm[n]**(-2)))
    
    l_n = -500*np.log(2*np.pi) - 0.5*(sign_M*logdet_M - logdet_sig) - 0.5*np.matmul(np.array([(X_mu_zeros[n])]),np.matmul(C_inv_n,(X_mu_zeros[n])))[0]

    return l_n

#%% Attempt to use matrices
a = np.array([1,2,3])
b = np.array([4,5,6])
both = np.array([a,b])

#ind = np.arange(0,np.shape(both)[0])
ind = np.array([0,1])
#ind = np.ones(np.shape(both)[0],dtype=bool)
#ind = [True,True]
#ind = [0,1]

both_diag = np.zeros([2,3,3])
both_flat = np.zeros([2,3**2])

both_flatdiag = np.diagflat(both)


n = np.arange(0,9,3)

both_diag[0] = np.transpose(both_flatdiag[n[0]:n[1]])[n[0]:n[1]]     #works but cant use a list of indices
both_diag[1] = np.transpose(both_flatdiag[n[1]:n[2]])[n[1]:n[2]]

#both_flat[ind] = np.diagflat(both[ind]).flatten()

#both_diag[0] = np.diagflat(both[0])       #works but cant use a list of indices
#both_diag[1] = np.diagflat(both[1])

#i = np.array([True,True],dtype=bool)
#both_diag[i] = np.diagflat(both[i])

#%% With for loop because matrices wouldn't work 
a = np.array([1,2,3])
b = np.array([4,5,6])
both = np.array([a,b])

both_diag = np.zeros([2,3,3])

for i in range(2):
    both_diag[i] = np.diagflat(both[i])
    


#%% For all n at once (matrices)

W_0 = pca.components_

def objective_function(W):
    n = 2
    W_T = np.transpose(W)

    spec_err1 = spec_err_norm_inf[:n]
    spec_err1_nan = spec_err_norm[:n]
    
    sig_diag = np.zeros([np.shape(spec_err1)[0],np.shape(spec_err1)[1],np.shape(spec_err1)[1]])
    sig_inv = np.zeros([np.shape(spec_err1)[0],np.shape(spec_err1)[1],np.shape(spec_err1)[1]])
    M_mul = np.zeros([np.shape(spec_err1)[0],np.shape(W)[0],np.shape(W)[0]])
    M_mul_inv = np.zeros([np.shape(spec_err1)[0],np.shape(W)[0],np.shape(W)[0]])
    C_inv = np.zeros([np.shape(spec_err1)[0],np.shape(spec_err1)[1],np.shape(spec_err1)[1]])
    sign_mul_M = np.zeros([np.shape(spec_err1)[0]])
    logdet_mul_M = np.zeros([np.shape(spec_err1)[0]])
    logdet_mul_sig = np.zeros([np.shape(spec_err1)[0]])
    l_n = np.zeros([np.shape(spec_err1)[0]])
    
    for i in range(np.shape(spec_err1)[0]):
        sig_diag[i] = np.diagflat(spec_err1[i]**2)
        sig_inv[i] = inv(np.diagflat(spec_err1[i]**2))
        M_mul[i] = np.identity(np.shape(W)[0]) + np.matmul(W_T,np.matmul(sig_inv[i],W)) 
        M_mul_inv[i] = inv(np.identity(np.shape(W)[0]) + np.matmul(W_T,np.matmul(sig_inv[i],W)))
        C_inv[i] = sig_inv[i] -  np.matmul(sig_inv[i],np.matmul(W,np.matmul(M_mul_inv[i],np.matmul(W_T,sig_inv[i]))))
        sign_mul_M[i], logdet_mul_M[i] = slogdet(M_mul[i])
        logdet_mul_sig[i] = np.nansum(np.log(spec_err1_nan[i]**(-2)))
        l_n[i] = -500*np.log(2*np.pi) - 0.5*(sign_mul_M[i]*logdet_mul_M[i] - logdet_mul_sig[i]) - 0.5*np.matmul(np.array([(X_mu_zeros[i])]),np.matmul(C_inv[i],(X_mu_zeros[i])))[0]
    
    ln = np.nansum(l_n)
    
    return -ln

#%% Try decrease saved values and therefore RAM used
W_0 = pca.components_

def objective_function1(W,X_mu_zeros,n1,n2):
    W_T = np.transpose(W)

    spec_err1 = spec_err_norm_inf[n1:n2]
    spec_err1_nan = spec_err_norm[n1:n2]
    
    l_n = np.zeros([np.shape(spec_err1)[0]])
    
    for i in range(np.shape(spec_err1)[0]):
        sig_inv = np.diagflat(spec_err1[i]**(-2))
        M = np.identity(np.shape(W)[0]) + np.matmul(W_T,np.matmul(sig_inv,W))
    #    print(i)
        M_inv = inv(np.identity(np.shape(W)[0]) + np.matmul(W_T,np.matmul(sig_inv,W)))

        C_inv = sig_inv -  np.matmul(sig_inv,np.matmul(W,np.matmul(M_inv,np.matmul(W_T,sig_inv))))

        sign_M, logdet_M = slogdet(M)

        logdet_sig = np.nansum(np.log(spec_err1_nan[i]**(-2)))
        l_n[i] = -500*np.log(2*np.pi) - 0.5*(sign_M*logdet_M - logdet_sig) - 0.5*np.matmul(np.array([(X_mu_zeros[i])]),np.matmul(C_inv,(X_mu_zeros[i])))[0]
    
    ln = np.nansum(l_n)
    
    return -ln

#%% Optimise for W by maximising ln

n1 = 0
n2 = 100

res = minimize(
    objective_function,
    x0=W_0,
    args=(X_mu_zeros,n1,n2),
)