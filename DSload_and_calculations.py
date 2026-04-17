# !/usr/bin/env python3
# Postprocess idealized ROMS output runs of canyon upwelling and tides experiments
# and calculate energy terms (APE, KE, N2), Energy fluxes and energy conversion using
# functions defined in dsload_dask.py. The calculations are saved as new NetCDF files.
# Uses Dask to parallelize calculations over time windows. 
# See function 'calc_all' for details on time windowing.

import dsload_dask as dsd
import xarray as xr
import time


# List of experiments and corresponding output file names for the processed datasets. The original datasets are loaded, 
# sliced, and processed in this loop, and the results are saved to new NetCDF files with the specified names.
experiments = ["ocean_his_exp3_obc_pert1.nc", 
               #"ocean_his_exp4_obc_pert1.nc", 
               "ocean_his_exp3_obc_pert1_M2.nc", 
               #"ocean_his_exp4_obc_pert1_M2.nc",
               #"ocean_his_exp2_obc_pert1.nc",
               #"ocean_his_exp2_obc_pert1_M2.nc",
               #"ocean_his_exp6_obc_pert1.nc",
               #"ocean_his_exp6_obc_pert1_M2.nc",
               ]
file_names = ["ds3_wnd5_st25.nc", 
              #"ds4_wnd5_st25.nc", 
              "ds3t_wnd5_st25.nc", 
              #"ds4t_wnd5_st25.nc",
              #"ds2_wnd5_st25.nc",
              #"ds2t_wnd5_st25.nc",
              #"ds6_wnd5_st25.nc",
              #"ds6t_wnd5_st25.nc"
              ]

# Time-averaging parameters for rolling window calculations (Aidan's defaults are window=3, step=10, end=22)
window=5*10 # n days * 10 timeslices
step=25
end= int((251-window)/step)+1

# Whether to apply low-pass filter to the data before computing energy terms (default = False). If True, 
# applies a low-pass Butterworth filter with a cutoff period of 40 hours to the u, v, and rho fields before 
# computing the energy terms. This can help to isolate the tidal signal by removing higher-frequency variability,
#  but it also increases the computational cost.
filt=False


for exp, file_name in zip(experiments, file_names):
    ds = xr.open_dataset("../../Upwelling_and_Tides/ROMS_Canyon_Idealized/" + exp, decode_times=False)
    print(f"{exp}: Started processing dataset at {time.ctime()}")
    ds_sliced = dsd.uv_rho(ds).sel(xi_rho=slice(20,121),eta_rho=slice(500,700),
                                   xi_u=slice(20,120),eta_u=slice(500,700),
                                   xi_v=slice(20,121),eta_v=slice(500,700))
    del ds
    ds_filtered = dsd.z_rho(dsd.w_rho(ds_sliced))
    del ds_sliced
    #ds_filtered.to_netcdf(sliced_filenames[file_names.index(file_name)])
            
    # Other calculations
    ds_1 = dsd.APE_c(dsd.sigma(dsd.dz(ds_filtered.drop_vars(["u", "v", "ubar", "vbar", "mask_rho", 
                                                           "mask_u", "mask_v", "mask_psi", "zeta"]))),
                                                           window, step, end, filt)
    del ds_filtered
    ds_2 = dsd.tke_bar(dsd.KE_c(ds_1, window, step, end, filt))
    del ds_1
    ds_2.to_netcdf(file_name)
    del ds_2
            
       