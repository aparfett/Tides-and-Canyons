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
td_phase = 0
adv_phase = 1
variable = "tke" # u, w or tke
percentege = 0 # choose 0 to plot velocity difference or 1 to plot percent difference (relative to the no-tide case)
x_index = 84  # choose a cross-shore section
cmap = cmo.cm.speed #cmo.cm.balance
cmap2 = cmo.cm.curl
bgcolor='tan'
clim = 2.5 # 10 for u td cm/s, 20 for adv, 1.0 for td w, 1.5 for adv w, 
clim2 = 0.75 # 1 for u td cm/s, 2 for adv, 0.3 for td w, 0.5 for adv w
rho_levs=[24.6,24.75,24.9,25.05,25.2,25.35,25.5,25.65,25.8,25.95,26.1,26.25,26.4,26.55,26.7,26.85]
rho_color='k'
## --- Calculate averages -----
if td_phase == 1:
    phase_str = "time dependent phase"
    phase_name = "time_dependent"
    t1=14*10 # day 14
    t2=16*10+2 # 
elif adv_phase == 1:
    phase_str = "advective phase"
    phase_name = "advective"
    t1=22*10 # day 22
    t2=24*10+2 # 
else:
    ValueError("Must set either td_phase or adv_phase to 1")


if variable == "u":
    var_name = "Cross-shelf"
    udsn = dsn.u_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    udst = dst.u_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    uisn = isn.u_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    uist = ist.u_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    ussn = ssn.u_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    usst = sst.u_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    fact = 2 # for u, multiply by 10^2 to get cm/s
elif variable == "w":
    var_name = "Vertical"
    udsn = dsn.w_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    udst = dst.w_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    uisn = isn.w_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    uist = ist.w_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    ussn = ssn.w_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    usst = sst.w_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    fact = 3 # for w, multiply by 10^3 to get mm/s 
elif variable == "tke":
    var_name = "Turbulent kinetic energy"
    udsn = dsn.tke.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    udst = dst.tke.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    uisn = isn.tke.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    uist = ist.tke.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    ussn = ssn.tke.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    usst = sst.tke.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    fact = 4 # for tke, multiply by 10^6 to get m^2/s^2
else:
    ValueError("variable must be 'u' or 'w' or 'tke'")
#

sigdsn = dsn.sigma.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
sigdst = dst.sigma.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
sigisn = isn.sigma.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
sigist = ist.sigma.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
sigssn = ssn.sigma.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
sigsst = sst.sigma.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values

zdsn = dsn.z_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
zdst = dst.z_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
zisn = isn.z_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
zist = ist.z_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
zssn = ssn.z_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
zsst = sst.z_rho.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values

if variable == "tke":
    zwdsn = dsn.z_w.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    zwdst = dst.z_w.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    zwisn = isn.z_w.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    zwist = ist.z_w.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    zwssn = ssn.z_w.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values
    zwsst = sst.z_w.isel(ocean_time=slice(t1,t2),xi_rho=x_index).mean(dim='ocean_time').values

# --- Figure ------
x = x_r[x_index]
y = y_r[:]

fig = plt.figure(figsize=(8, 10)) # constrained_layout helps with spacing
gs = GridSpec(nrows=3, ncols=3, figure=fig) # Example: 2 rows, 2 columns

# ---- first row ----
ax1 = fig.add_subplot(gs[0,0])
if variable == "tke":
    ax1.pcolormesh(y, zwdsn[:,:], udsn[:,:] * (10**(fact)), shading="auto", 
                 cmap=cmap, vmin=0, vmax=clim)
else:
    ax1.pcolormesh(y, zdsn[:,:], udsn[:,:] * (10**(fact)), shading="auto", 
                 cmap=cmap, vmin=-clim, vmax=clim)
ax1.contour(np.tile(y,(30,1)), zdsn[:,:], sigdsn[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)

ax2 = fig.add_subplot(gs[0, 1])
if variable == "tke":
    ax2.pcolormesh(y, zwdst[:,:], udst[:,:] * (10**(fact)), shading="auto", 
                 cmap=cmap, vmin=0, vmax=clim)
else:
    ax2.pcolormesh(y, zdst[:,:], udst[:,:] * (10**(fact)), shading="auto", 
                 cmap=cmap, vmin=-clim, vmax=clim)
ax2.contour(np.tile(y,(30,1)), zdst[:,:], sigdst[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)

norm = mpl.colors.Normalize(vmin=0, vmax=clim)
cbar_ax = fig.add_axes([0.14, 0.05, 0.46, 0.01])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=cbar_ax, orientation='horizontal', format='%1.1f', 
                  label=r'$10^{-' + str(fact) + '}$ m$^2$s$^{-2}$')
cbar_ax.xaxis.set_tick_params(pad=0)
   

ax3 = fig.add_subplot(gs[0, 2])

if percentege == 1:
    ax3.pcolormesh(y, zdsn[:,:], ((udsn[:,:] - udst[:,:]) / udsn[:,:]) , shading="auto", 
               cmap=cmap2,vmin=-1,vmax=1)
    norm2 = mpl.colors.Normalize(vmin=-2, vmax=2)
    cbar_ax2 = fig.add_axes([0.67, 0.05, 0.23, 0.01])
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2),
                  cax=cbar_ax2, orientation='horizontal', format='%1.1f', 
                  label=r'% difference from no-tide case')

else:
    if variable == "tke":
        ax3.pcolormesh(y, zwdsn[:,:], (udsn[:,:] - udst[:,:]) * (10**(fact)), shading="auto", 
                   cmap=cmap2,vmin=-clim2,vmax=clim2)
    else:
        ax3.pcolormesh(y, zdsn[:,:], (udsn[:,:] - udst[:,:]) * (10**(fact)), shading="auto", 
                  cmap=cmap2,vmin=-clim2,vmax=clim2)
    norm2 = mpl.colors.Normalize(vmin=-clim2, vmax=clim2)
    cbar_ax2 = fig.add_axes([0.67, 0.05, 0.23, 0.01])
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2),
                  cax=cbar_ax2, orientation='horizontal', format='%1.1f', 
                  label=r'$10^{-' + str(fact) + '}$ m$^2$s$^{-2}$')

cbar_ax2.xaxis.set_tick_params(pad=0)

# --- second row ----
ax4 = fig.add_subplot(gs[1,0])
if variable == "tke":
    ax4.pcolormesh(y, zwisn[:,:], uisn[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap, vmin=0, vmax=clim)
else:
    ax4.pcolormesh(y, zisn[:,:], uisn[:,:] * (10**(fact)), shading="auto", 
                  cmap=cmap, vmin=-clim, vmax=clim)
ax4.contour(np.tile(y,(30,1)), zisn[:,:], sigisn[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)

ax5 = fig.add_subplot(gs[1, 1])
if variable == "tke":
    ax5.pcolormesh(y, zwist[:,:], uist[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap, vmin=0, vmax=clim)
else:
    ax5.pcolormesh(y, zist[:,:], uist[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap,vmin=-clim,vmax=clim)
ax5.contour(np.tile(y,(30,1)), zist[:,:], sigist[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)

ax6 = fig.add_subplot(gs[1, 2])
if percentege == 1:
    ax6.pcolormesh(y, zisn[:,:], ((uisn[:,:] - uist[:,:]) / uisn[:,:]) , shading="auto", 
               cmap=cmap2,vmin=-2,vmax=2)
else:
    if variable == "tke":
        ax6.pcolormesh(y, zwisn[:,:], (uisn[:,:] - uist[:,:]) * (10**(fact)), shading="auto", 
                   cmap=cmap2,vmin=-clim2,vmax=clim2)
    else:
        ax6.pcolormesh(y, zisn[:,:], (uisn[:,:] - uist[:,:]) * (10**(fact)), shading="auto", 
               cmap=cmap2,vmin=-clim2,vmax=clim2)


# --- third row ----

ax7 = fig.add_subplot(gs[2,0])
if variable == "tke":
    ax7.pcolormesh(y, zwssn[:,:], ussn[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap, vmin=0, vmax=clim)
else:
    ax7.pcolormesh(y, zssn[:,:], ussn[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap, vmin=-clim, vmax=clim)
ax7.contour(np.tile(y,(30,1)), zssn[:,:], sigssn[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)

ax8 = fig.add_subplot(gs[2, 1])
if variable == "tke":
    ax8.pcolormesh(y, zwsst[:,:], usst[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap, vmin=0, vmax=clim)
else:
    ax8.pcolormesh(y, zsst[:,:], usst[:,:] * (10**(fact)), shading="auto", 
               cmap=cmap,vmin=-clim,vmax=clim)
ax8.contour(np.tile(y,(30,1)), zsst[:,:], sigsst[:,:],
            levels=rho_levs,colors=rho_color,linewidths=0.5)

ax9 = fig.add_subplot(gs[2, 2])
if percentege == 1:
    ax9.pcolormesh(y, zssn[:,:], ((ussn[:,:] - usst[:,:]) / ussn[:,:]) , shading="auto", 
               cmap=cmap2,vmin=-2,vmax=2)
else:
    if variable == "tke":
        ax9.pcolormesh(y, zwssn[:,:], (ussn[:,:] - usst[:,:]) * (10**(fact)), shading="auto", 
                   cmap=cmap2,vmin=-clim2,vmax=clim2)
    else:
        ax9.pcolormesh(y, zssn[:,:], (ussn[:,:] - usst[:,:]) * (10**(fact)), shading="auto", 
               cmap=cmap2,vmin=-clim2,vmax=clim2)   
   

# --- axis aesthetics ---
title = f"{var_name} during the {phase_str}"
ax2.annotate(title, xy=(0.5, 1.15), xycoords="axes fraction", fontsize=12, ha='center')

ax1.set_title("No tide")
ax2.set_title("Tide")
ax3.set_title("Difference (No tide - Tide)")

for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]: 
    ax.set_facecolor(bgcolor)
    ax.set_ylim([-200,0])
    ax.set_xlim([-30,30])
    ax.xaxis.set_inverted(True)

for ax, exp in zip([ax1, ax4, ax7], ['DS','IS','SS']):
    ax.set_ylabel("Depth (m)")
    ax.annotate(exp, xy=(0.1, 0.1), xycoords="axes fraction", fontsize=12, fontweight='bold')
for ax in [ax2,ax3,ax5,ax6,ax8,ax9]:
    ax.set_yticklabels([])

for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
    ax.set_xticklabels([])

for ax in [ax7, ax8, ax9]:
    ax.set_xlabel("y (km)")

if save == 1:
    if percentege == 1:
        fig.savefig(f"{var_name}_{phase_name}_pcdiff.png", dpi=300)
    else:
        fig.savefig(f"{var_name}_{phase_name}.png", dpi=300)
    

plt.show()
