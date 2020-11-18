# Recreating https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/astronomy/dimensionality_reduction.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

# %% Load data (spec4000.npz saved from fetch_and_shift_spectra() in compute_sdss_pca.py)
data = np.load('spec4000.npz')

# %% Set X and wavelengths (from wavelength coefficients)
X = data['spectra']
log_wavelengths = data['coeff0'] + data['coeff1'] * np.arange(0, 1000, 1)
wavelengths = [10**i for i in log_wavelengths]

# %% Plot mean spectrum
X_normal = preprocessing.normalize(X)
mu = X_normal.mean(0)
std = X_normal.std(0)
plt.figure()
plt.plot(wavelengths, mu, color = 'black')
plt.fill_between(wavelengths, mu - std, mu + std , color = 'lightgrey')
plt.xlim(wavelengths[0], wavelengths[-1])
plt.ylim(0,0.06)
plt.xlabel('Wavelength [$\AA$]')
plt.ylabel('Scaled flux')
plt.title('Mean spectrum')
plt.show()

# %%
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

# %% Apply PCA
pca = PCA(n_components=4)
X_red = pca.fit_transform(X_normal)

# X_red, evals,evecs = PCA_fs(X, n_components=4)

# %%
plt.figure()
plt.scatter(X_red[:, 0], X_red[:, 1], s=4, lw=0, vmin=2, vmax=6)
plt.xlabel('coefficient 1')
plt.ylabel('coefficient 2')
plt.title('PCA projection of Spectra')

# %%
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
