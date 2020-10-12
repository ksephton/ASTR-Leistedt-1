import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table, hstack

# %%
north = Table.read('survey-dr8-north-specObj-dr14.fits',format='fits')
south = Table.read('survey-dr8-south-specObj-dr14.fits',format='fits')
spectro = Table.read('specObj-dr14.fits',format='fits')

# %%
if True:
    ind = south['OBJID'] == -1
    south[ind] = north[ind]
    
# %%
photo = south

# %%
full = hstack([spectro, photo])

# %%
bands = np.array(['G','R','Z'])
for i in bands:
    full['FLUX_'+i] = full['FLUX_'+i]/full['MW_TRANSMISSION_'+i]
    full['FLUX_IVAR_'+i] = full['FLUX_IVAR_'+i] * (full['MW_TRANSMISSION_'+i]**2)
    
# %%
impt_cols = np.array(['MJD','PLATE','FIBERID','RA','DEC','Z','ZWARNING','CLASS'])
for i in bands:
    impt_cols = np.append(impt_cols,'FLUX_'+i)
    impt_cols = np.append(impt_cols,'FLUX_IVAR_'+i)
    impt_cols = np.append(impt_cols,'MW_TRANSMISSION_'+i)
    
# %%
full.rename_column('PLATE_1', 'PLATE')
full.rename_column('MJD_1', 'MJD')
full.rename_column('FIBERID_1', 'FIBERID')

# %% Drop unnecessary columns
impt_cols_list = impt_cols.tolist()
full_short = full[impt_cols_list]

# %% Plot position in sky
# Code from https://stackoverflow.com/questions/29525356/produce-a-ra-vs-dec-equatorial-coordinates-plot-with-python

ra_rad = [x / 180.0 * np.pi for x in full_short['RA']]

# Make plot.
fig= plt.figure()
ax = plt.subplot(polar=True)

# Set x,y ticks
angs = np.array([330., 345., 0., 15., 30., 45., 60., 75., 90., 105., 120.])
ax.set_rlabel_position(120)


cm = plt.cm.get_cmap('RdYlBu_r')
SC = ax.scatter(ra_rad, full_short['DEC'], marker='o', c=full_short['Z'], s=10, lw=0., cmap=cm)
cbar = plt.colorbar(SC, shrink=1., pad=0.05)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('colorbar', fontsize=8)

plt.show()
# %% Save position in sky figure
plt.savefig('ra_dec_plot.png', dpi=300)

# %% Plot colour distribution 
plt.figure()
plt.plot(full_short['Z'],-2.5*np.log10(full_short['FLUX_R']/full_short['FLUX_G']), 'o')
plt.show()

# %% Plot colour distribution for galaxies
galaxy = full_short['CLASS'] == 'GALAXY'
full_short_g = full_short[galaxy]
plt.figure()
plt.plot(full_short_g['Z'],-2.5*np.log10(full_short_g['FLUX_R']/full_short_g['FLUX_G']), 'o')
plt.show()
