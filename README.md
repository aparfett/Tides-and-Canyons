# Tides-and-Canyons

Script DSload_and_calculations.py postprocesses idealized ROMS output runs of canyon upwelling and tides experiments and calculates energy terms (APE, KE, N2), Energy fluxes and energy conversion using functions defined in dsload_dask.py. The calculations are saved as new NetCDF files. 

It uses Dask to parallelize calculations over time windows. 
See function ``calc_all`` for details on time windowing.

Scripts to make various figures are saved under figures/. They are a bit messy right now.