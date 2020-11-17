# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:00:37 2020
"""
#%% from https://www.astroml.org/examples/datasets/plot_sdss_spectrum.html

from matplotlib import pyplot as plt
from astroML.datasets import fetch_sdss_spectrum
import numpy as np

#------------------------------------------------------------
# Fetch single spectrum
plate = 1732
mjd = 53501
fiber = 156

spec = fetch_sdss_spectrum(plate, mjd, fiber)

#------------------------------------------------------------
# Plot the resulting spectrum
plt.figure()
ax = plt.axes()
ax.plot(np.log10(spec.wavelength()), spec.spectrum, '-k', label='spectrum')
ax.plot(np.log10(spec.wavelength()), spec.error, '-', color='gray', label='error')
#ax.plot(spec.wavelength(), spec.spectrum, '-k', label='spectrum')
#ax.plot(spec.wavelength(), spec.error, '-', color='gray', label='error')

ax.legend(loc=4)

ax.set_title('Plate = %(plate)i, MJD = %(mjd)i, Fiber = %(fiber)i' % locals())

#ax.text(0.05, 0.95, 'z = %.2f' % spec.z, size=16,
#        ha='left', va='top', transform=ax.transAxes)

ax.set_xlabel(r'log $\lambda (\AA)$')
ax.set_ylabel('Flux')

ax.set_ylim(-5, 35)


plt.show()
