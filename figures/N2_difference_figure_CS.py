import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cmocean as cmo
import matplotlib as mpl

# Load exp with M2 tide
dst=xr.open_dataset("ds2t_wnd5_st25.nc",decode_times=False)
ist=xr.open_dataset("ds4t_wnd5_st25.nc",decode_times=False)
sst=xr.open_dataset("ds6t_wnd5_st25.nc",decode_times=False)

# Load exp without tide
dsn=xr.open_dataset("ds2_wnd5_st25.nc",decode_times=False)
isn=xr.open_dataset("ds4_wnd5_st25.nc",decode_times=False)
ssn=xr.open_dataset("ds6_wnd5_st25.nc",decode_times=False)

# Grid and bathymetry
x_rho = isn["x_rho"].values
y_rho = isn["y_rho"].values
x_u = isn["x_u"].values
y_v = isn["y_v"].values
y_r = (y_rho[:,1]/1000)-300 # center around canyon
x_r = (x_rho[1,:]/1000)-155

h_ds=dsn.h.values
h_is=isn.h.values
h_ss=ssn.h.values

# Useful constants
g = 9.81
rho0 = isn.rho0.values
omega_M2 = 2 * np.pi / (12.4206 * 3600.)
f = 1.0284e-04

# plotting parameters
save = 1 # 1 or 0 to save or not save figures
td_phase = 1
adv_phase = 0
variable = "N2" # u or w, (n, s_rho, eta_rho, xi_rho)
percentege = 0 # choose 0 to plot velocity difference or 1 to plot percent difference (relative to the no-tide case)
y_index = 100  #100 - axis choose an alongshore section
cmap = cmo.cm.dense
cmap2 = cmo.cm.curl
bgcolor='tan'
clim = 1 # 
clim2 = 1 #
rho_levs=[24.6,24.75,24.9,25.05,25.2,25.35,25.5,25.65,25.8,25.95,26.1,26.25,26.4,26.55,26.7,26.85]
rho_color='k'

## --- Calculate averages -----
if td_phase == 1:
    phase_str = "time dependent phase"
    t1= 5 #2 only tides # 5 day 12.5 
    t1r = 14*10#2*10 only tides #14*10 # day 14
    t2r = 16*10 #5*10 #16*10 # day 16
elif adv_phase == 1:
    phase_str = "advective phase"
    t1=8 # day 20.0
    t1r = 22*10 # day 22
    t2r = 24*10 # day 24
else:
    ValueError("Must set either td_phase or adv_phase to 1")


if variable == "N2":
    var_name = "N2"
    udsn = dsn.N2.isel(n=t1,eta_rho=y_index).values
    udst = dst.N2.isel(n=t1,eta_rho=y_index).values
    uisn = isn.N2.isel(n=t1,eta_rho=y_index).values
    uist = ist.N2.isel(n=t1,eta_rho=y_index).values
    ussn = ssn.N2.isel(n=t1,eta_rho=y_index).values
    usst = sst.N2.isel(n=t1,eta_rho=y_index).values
    fact = 4 
elif variable == "KE":
    var_name = "KE"
    udsn = dsn.KE.isel(n=t1,eta_rho=y_index).values
    udst = dst.KE.isel(n=t1,eta_rho=y_index).values
    uisn = isn.KE.isel(n=t1,eta_rho=y_index).values
    uist = ist.KE.isel(n=t1,eta_rho=y_index).values
    ussn = ssn.KE.isel(n=t1,eta_rho=y_index).values
    usst = sst.KE.isel(n=t1,eta_rho=y_index).values
    fact = 0 
else:
    ValueError("variable must be 'N2' or 'KE'")
#

sigdsn = dsn.sigma.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values
sigdst = dst.sigma.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values
sigisn = isn.sigma.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values
sigist = ist.sigma.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values
sigssn = ssn.sigma.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values
sigsst = sst.sigma.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values

zdsn = dsn.z_rho.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values
zdst = dst.z_rho.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values
zisn = isn.z_rho.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values
zist = ist.z_rho.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values
zssn = ssn.z_rho.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values
zsst = sst.z_rho.isel(ocean_time=slice(t1r,t2r),eta_rho=y_index).mean(dim='ocean_time').values

# --- Figure ------
x = x_r[:]
y = y_r[y_index]

fig = plt.figure(figsize=(8, 10)) # constrained_layout helps with spacing
gs = GridSpec(nrows=3, ncols=3, figure=fig) # Example: 2 rows, 2 columns

# ---- first row ----
ax1 = fig.add_subplot(gs[0,0])
ax1.pcolormesh(x, zdsn[:,:], udsn[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap, vmin=0, vmax=clim)
ax1.contour(np.tile(x,(30,1)), zdsn[:,:], sigdsn[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)

ax2 = fig.add_subplot(gs[0, 1])
ax2.pcolormesh(x, zdst[:,:], udst[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap,vmin=0,vmax=clim)
ax2.contour(np.tile(x,(30,1)), zdst[:,:], sigdst[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)
ax2.plot(x, -h_ds[50,:], linewidth=0.5, color='0.5')

norm = mpl.colors.Normalize(vmin=0, vmax=clim)
cbar_ax = fig.add_axes([0.14, 0.05, 0.46, 0.01])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=cbar_ax, orientation='horizontal', format='%1.1f', 
                  label=r'10$^{-' + str(fact) + '}$ 1/s$^2$')
cbar_ax.xaxis.set_tick_params(pad=0)
   

ax3 = fig.add_subplot(gs[0, 2])

if percentege == 1:
    ax3.pcolormesh(x, zdsn[:,:], ((udsn[:,:] - udst[:,:]) / udsn[:,:]) , shading="auto", 
               cmap=cmap2,vmin=-1,vmax=1)
    norm2 = mpl.colors.Normalize(vmin=-2, vmax=2)
    cbar_ax2 = fig.add_axes([0.67, 0.05, 0.23, 0.01])
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2),
                  cax=cbar_ax2, orientation='horizontal', format='%1.1f', 
                  label=r'% difference from no-tide case')

else:
    ax3.pcolormesh(x, zdsn[:,:], (udsn[:,:] - udst[:,:]) * (10**(fact)), shading="auto", 
               cmap=cmap2,vmin=-clim2,vmax=clim2)
    norm2 = mpl.colors.Normalize(vmin=-clim2, vmax=clim2)
    cbar_ax2 = fig.add_axes([0.67, 0.05, 0.23, 0.01])
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2),
                  cax=cbar_ax2, orientation='horizontal', format='%1.1f', 
                  label=r'10$^{-' + str(fact) + '}$ 1/s$^2$')

ax3.plot(x, -h_ds[50,:], linewidth=0.5, color='0.5')
cbar_ax2.xaxis.set_tick_params(pad=0)

# --- second row ----
ax4 = fig.add_subplot(gs[1,0])
ax4.pcolormesh(x, zisn[:,:], uisn[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap, vmin=0, vmax=clim)
ax4.contour(np.tile(x,(30,1)), zisn[:,:], sigisn[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)
ax4.plot(x, -h_is[50,:], linewidth=0.5, color='0.5')

ax5 = fig.add_subplot(gs[1, 1])
ax5.pcolormesh(x, zist[:,:], uist[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap,vmin=0,vmax=clim)
ax5.contour(np.tile(x,(30,1)), zist[:,:], sigist[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)
ax5.plot(x, -h_is[50,:], linewidth=0.5, color='0.5')

ax6 = fig.add_subplot(gs[1, 2])
if percentege == 1:
    ax6.pcolormesh(x, zisn[:,:], ((uisn[:,:] - uist[:,:]) / uisn[:,:]) , shading="auto", 
               cmap=cmap2,vmin=-2,vmax=2)
else:
    ax6.pcolormesh(x, zisn[:,:], (uisn[:,:] - uist[:,:]) * (10**(fact)), shading="auto", 
               cmap=cmap2,vmin=-clim2,vmax=clim2)

ax6.plot(x, -h_is[50,:], linewidth=0.5, color='0.5')

# --- third row ----

ax7 = fig.add_subplot(gs[2,0])
ax7.pcolormesh(x, zssn[:,:], ussn[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap, vmin=0, vmax=clim)
ax7.contour(np.tile(x,(30,1)), zssn[:,:], sigssn[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)
ax7.plot(x, -h_ss[50,:], linewidth=0.5, color='0.5')

ax8 = fig.add_subplot(gs[2, 1])
ax8.pcolormesh(x, zsst[:,:], usst[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap,vmin=0,vmax=clim)
ax8.contour(np.tile(x,(30,1)), zsst[:,:], sigsst[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)
ax8.plot(x, -h_ss[50,:], linewidth=0.5, color='0.5')

ax9 = fig.add_subplot(gs[2, 2])
if percentege == 1:
    ax9.pcolormesh(x, zssn[:,:], ((ussn[:,:] - usst[:,:]) / ussn[:,:]) , shading="auto", 
               cmap=cmap2,vmin=-2,vmax=2)
else:
    ax9.pcolormesh(x, zssn[:,:], (ussn[:,:] - usst[:,:]) * (10**(fact)), shading="auto", 
               cmap=cmap2,vmin=-clim2,vmax=clim2)
ax9.plot(x, -h_ss[50,:], linewidth=0.5, color='0.5')


# --- axis aesthetics ---
title = f"{var_name} during the {phase_str}"
ax2.annotate(title, xy=(0.5, 1.15), xycoords="axes fraction", fontsize=12, ha='center')

ax1.set_title("No tide")
ax2.set_title("Tide")
ax3.set_title("Difference (No tide - Tide)")

for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]: 
    ax.set_facecolor(bgcolor)
    ax.set_ylim([-200,0])
    ax.set_xlim([-45,0]) 
    ax.xaxis.set_inverted(True)

for ax, exp in zip([ax1, ax4, ax7], ['DS','IS','SS']):
    ax.set_ylabel("Depth (m)")
    ax.annotate(exp, xy=(0.1, 0.1), xycoords="axes fraction", fontsize=12, fontweight='bold')
for ax in [ax2,ax3,ax5,ax6,ax8,ax9]:
    ax.set_yticklabels([])

for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
    ax.set_xticklabels([])

for ax in [ax7, ax8, ax9]:
    ax.set_xlabel("x (km)")

if save == 1:
    if percentege == 1:
        fig.savefig(f"{var_name}_{phase_str}_CS_y{y_index}_pcdiff.png")
    else:
        fig.savefig(f"{var_name}_{phase_str}_CS_y{y_index}.png")
    
else:
    plt.show()
