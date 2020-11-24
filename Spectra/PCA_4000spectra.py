# Recreating https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/astronomy/dimensionality_reduction.html
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
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

#%%
plt.figure()
plt.plot(wavelengths,X_normal[4])

#%% Set all zero normalised fluxes to nan
zero_norm_ind = X_normal == 0.
X_normal[zero_norm_ind] = np.NaN

#%% Transform errors due to corresponding normalisation
spec_err_T = np.transpose(spec_err)
spec_err_norm_T = np.divide(spec_err_T,norm)
spec_err_norm = np.transpose(spec_err_norm_T)

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

#%%
def PCA_fs(X,n_components=None):
    ''' PCA function adapted from https://www.askpython.com/python/examples/principal-component-analysis
    Input:
    X: numpy nd.array
    n_components: Denotes the number of principal components; can be integer or None with default as n_components = None.
                  If None, function will automatically the optimal number of principal components based by finding the maximum
                  decrease in added variance %
    Returns:
    X_reduced: The dataset with redu

    '''
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    #Step-2
    S = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    evals , evecs = np.linalg.eigh(S)
    evals = np.flip(evals)
    evecs = np.flip(evecs, axis = 1)
     
    #Step-4
    var = evals/np.sum(evals)
    cum_var = np.cumsum(var)
    
    plt.figure()
    plt.plot(np.array(range(1,len(evals)+1)), var, label = 'Variance %', color = 'tab:blue')
    plt.plot(np.array(range(1,len(evals)+1)), var, 'x', color = 'tab:blue', markersize = 10)
    plt.plot(np.array(range(1,len(evals)+1)), cum_var, label = 'Cumulative variance %', color = 'tab:green')
    plt.plot(np.array(range(1,len(evals)+1)), cum_var, '+', color = 'tab:green', markersize = 10)
    plt.xlabel('Number of principal components')
    plt.ylabel('Variance %')
    plt.xticks(np.array(range(1,len(evals)+1)))
    plt.legend()
    plt.show()
    
    var_diff = [var[i] - var[i+1] for i in range(len(var)-1)]
    if n_components == None:
        n_components = var_diff.index(max(var_diff))+1
        print('Optimal number of principal components:', n_components)
    
    #Step-5
    evecs_subset = evecs[:,0:n_components]
    X_reduced = np.dot(evecs_subset.transpose(), X_meaned.transpose()).transpose()
     
    return X_reduced, evals, evecs

#%% Apply PCA
pca = PCA(n_components=4)
X_red = pca.fit_transform(X_norm_zeros)

# X_red, evals,evecs = PCA_fs(X, n_components=4)

#%%
plt.figure()
plt.scatter(X_red[:, 0], X_red[:, 1], c = subclass, s=4, lw=0, vmin=2, vmax=6)
plt.colorbar()
plt.xlabel('coefficient 1')
plt.ylabel('coefficient 2')
plt.title('PCA projection of Spectra')

#%%
plt.figure()
l = plt.plot(wavelengths, pca.mean_ - 0.15)
c = l[0].get_color()
plt.text(7000, -0.16, "mean", color=c)
for i in range(4):
    l = plt.plot(wavelengths, pca.components_[i] + 0.15 * i)
    c = l[0].get_color()
    plt.text(7000, -0.01 + 0.15 * i, "component %i" % (i + 1), color=c)
#plt.ylim(-0.2, 0.6)
plt.xlabel('wavelength (Angstroms)')
plt.ylabel('scaled flux + offset')
plt.title('Mean Spectrum and Eigen-spectra')

# plt.figure()
# l = plt.plot(wavelengths, X_normal.mean(axis=0) - 0.15)
# c = l[0].get_color()
# plt.text(7000, -0.16, "mean", color=c)
# for i in range(4):
#     l = plt.plot(wavelengths, evecs[i] + 0.15 * i)
#     c = l[0].get_color()
#     plt.text(7000, -0.01 + 0.15 * i, "component %i" % (i + 1), color=c)
# plt.ylim(-0.2, 0.6)
# plt.xlabel('wavelength (Angstroms)')
# plt.ylabel('scaled flux + offset')
# plt.title('Mean Spectrum and Eigen-spectra')
