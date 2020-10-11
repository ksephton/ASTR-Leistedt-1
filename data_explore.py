import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table, hstack

# %%
north = Table.read('survey-dr8-north-specObj-dr14.fits',format='fits')
south = Table.read('survey-dr8-south-specObj-dr14.fits',format='fits')
spectro = Table.read('specObj-dr14.fits',format='fits')
