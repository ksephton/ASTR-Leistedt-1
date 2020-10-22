import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# %% Read in data
full = Table.read('full.fits', format = 'fits')

# %% Plot position in sky
# Code from https://stackoverflow.com/questions/29525356/produce-a-ra-vs-dec-equatorial-coordinates-plot-with-python

ra_rad = [x / 180.0 * np.pi for x in full['RA']]

# Make plot.
fig= plt.figure()
ax = plt.subplot(polar=True)

# Set x,y ticks
ax.set_rlabel_position(120)

# SC = ax.scatter(ra_rad, full['DEC'], marker='o')
                
cm = plt.cm.get_cmap('RdYlBu_r')
SC = ax.scatter(ra_rad, full['DEC'], marker='o', c=full['Z'], s=10, lw=0., cmap=cm)
cbar = plt.colorbar(SC, shrink=1., pad=0.05)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Redshift z', fontsize=8)
plt.savefig('position_full.png')
plt.show()

# %% Make subset with only objid = -1
with_objid = full[full['OBJID']!= -1]

# %% Plot position in sky for objid != -1
# Code from https://stackoverflow.com/questions/29525356/produce-a-ra-vs-dec-equatorial-coordinates-plot-with-python

ra_rad = [x / 180.0 * np.pi for x in with_objid['RA']]

# Make plot.
fig= plt.figure()
ax = plt.subplot(polar=True)

# Set x,y ticks
ax.set_rlabel_position(120)

SC = ax.scatter(ra_rad, with_objid['DEC'], marker='o')
                
cm = plt.cm.get_cmap('RdYlBu_r')
SC = ax.scatter(ra_rad, with_objid['DEC'], marker='o', c=with_objid['Z'], s=10, lw=0., cmap=cm)
cbar = plt.colorbar(SC, shrink=1., pad=0.05)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Redshift z', fontsize=8)
plt.savefig('position_full_wobjid.png')
plt.show()

# %% Repeat with smaller dataset (~4000 points)
small_full = full[::1000]
small_with_objid = small_full[small_full['OBJID']!= -1]

# %% Plot position in sky small
# Code from https://stackoverflow.com/questions/29525356/produce-a-ra-vs-dec-equatorial-coordinates-plot-with-python

ra_rad = [x / 180.0 * np.pi for x in small_full['RA']]

# Make plot.
fig= plt.figure()
ax = plt.subplot(polar=True)

# Set x,y ticks
ax.set_rlabel_position(120)

# SC = ax.scatter(ra_rad, small_full['DEC'], marker='o')
                
cm = plt.cm.get_cmap('RdYlBu_r')
SC = ax.scatter(ra_rad, small_full['DEC'], marker='o', c=small_full['Z'], s=10, lw=0., cmap=cm)
cbar = plt.colorbar(SC, shrink=1., pad=0.05)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('colorbar', fontsize=8)
plt.savefig('position_smallfull.png')
plt.show()

# %% Plot position in sky for objid != -1 small
# Code from https://stackoverflow.com/questions/29525356/produce-a-ra-vs-dec-equatorial-coordinates-plot-with-python

ra_rad = [x / 180.0 * np.pi for x in small_with_objid['RA']]

# Make plot.
fig= plt.figure()
ax = plt.subplot(polar=True)

# Set x,y ticks
ax.set_rlabel_position(120)

# SC = ax.scatter(ra_rad, small_with_objid['DEC'], marker='o')
                
cm = plt.cm.get_cmap('RdYlBu_r')
SC = ax.scatter(ra_rad, small_with_objid['DEC'], marker='o', c=small_with_objid['Z'], s=10, lw=0., cmap=cm)
cbar = plt.colorbar(SC, shrink=1., pad=0.05)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('colorbar', fontsize=8)
plt.savefig('position_smallfull_wobjid.png')
plt.show()


