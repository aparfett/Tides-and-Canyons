import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cmocean as cmo
import matplotlib as mpl
import seaborn as sns

sns.set_context('notebook', font_scale=1.2)
isc=xr.open_dataset("ds4_wnd5_st25.nc",decode_times=False)
isct=xr.open_dataset("ds4t_wnd5_st25.nc",decode_times=False)
isn=xr.open_dataset("ds3_wnd5_st25.nc",decode_times=False)
isnt=xr.open_dataset("ds3t_wnd5_st25.nc",decode_times=False)

# Grid and bathymetry
x_rho = isc["x_rho"].values
y_rho = isc["y_rho"].values
x_u = isc["x_u"].values
y_v = isc["y_v"].values
y_r = (y_rho[:,1]/1000)-300 # center around canyon
x_r = (x_rho[1,:]/1000)-155
h_c=isc.h.values
h_n=isn.h.values

## --- Plot bathymetry ---

fig=plt.figure(figsize=(4, 6.6))
ax = plt.gca()
plt.pcolor((x_rho[1,:]/1000)-155,(y_rho[:,1]/1000)-300,h_c,shading='auto',cmap=cmo.cm.deep)
plt.colorbar(label="Depth (m)")
plt.clim([50,500])
plt.contour((x_rho[1,:]/1000)-155,(y_rho[:,1]/1000)-300, h_c,levels=[150,200,300,400],colors=['coral','r','r','r'])
plt.xlim([-50,0])
plt.ylim([-50,50])
plt.ylabel("y (km)")
plt.xlabel("x (km)")
ax.axvline(x_rho[0,84]/1000-155, color='k', linestyle='--', linewidth=3)
#ax.axvline(x_rho[0,70]/1000-155, color='k', linestyle='--')

ax.set_aspect('equal')
fig.savefig("bathy_canyon_x84.png",bbox_inches='tight', dpi=300)
plt.show()

## -----
## -------------------------------------------------------
## Plot energy conversion at different windows -----------
## -------------------------------------------------------
# cmap = cmo.cm.balance

# # choose windows to plot
# win_index = [2, 5, 7]
# h_nc = isn.h.values
# h_c = isc.h.values
# x_index = 84  # choose a cross-shore section
# window = 50 # 5 days * 10 outputs per day
# step = 25 # 2.5 days * 10 per day

# for win in win_index:
#     # Create figure and axes once
#     fig = plt.figure(figsize=(10,6))
#     gs = GridSpec(nrows=1, ncols=4, figure=fig )
    
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[0, 2])
#     ax4 = fig.add_subplot(gs[0, 3])

#     # Create colorbar axes
#     #fig.subplots_adjust(right=0.8)
#     #plt.subplots_adjust(wspace=0.4)
#     cbar_ax1 = fig.add_axes([0.22, 0.08, 0.4, 0.02])
#     cbar_ax2 = fig.add_axes([0.73, 0.08, 0.17, 0.02])

#     # Calculate title and time slice
#     title = f"{(step*win)*(2.4/24):.1f}-{((step*win)+window)*(2.4/24):.1f} days"
        
#     # Extract data for this frame
#     Fu3_slice = isn.C.values[win, :, :]
#     Fu4_slice = isc.C.values[win, :, :]
#     Fu3t_slice = isnt.C.values[win, :, :]
#     Fu4t_slice = isct.C.values[win, :, :]
    
#     # Plot 1: No canyon, no tide
#     ax1.pcolormesh(x_r, y_r, Fu3_slice*1E3, shading="auto", cmap=cmap, vmin=-0.025, vmax=0.025)
#     ax1.set_title("No canyon, no tide")
#     ax1.contour(x_r, y_r, h_nc,levels=[150,200,300,400],colors='0.5',linewidths=0.5)
#     ax1.set_aspect(1)
#     ax1.set_ylabel("y (km)")
#     ax1.set_xlabel("x (km)")
#     ax1.annotate(title, xy=(-0.2, 1.1), xycoords="axes fraction", fontweight='bold', fontsize=14)
    
#     # Plot 2: No canyon, tide
#     ax2.pcolormesh(x_r, y_r, Fu3t_slice*1E3, shading="auto", cmap=cmap, vmin=-0.025, vmax=0.025)
#     ax2.contour(x_r, y_r, h_nc,levels=[150,200,300,400],colors='0.5',linewidths=0.5)
#     ax2.set_title("No canyon, tide")
#     ax2.tick_params(axis='y', labelleft=False)
#     ax2.set_xlabel("x (km)")
#     ax2.set_aspect(1) 
    
#     # Plot 3: Canyon, no tide
#     im1 = ax3.pcolormesh(x_r, y_r, Fu4_slice*1E3, shading="auto", cmap=cmap, vmin=-0.025, vmax=0.025)
#     ax3.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='0.5',linewidths=0.5)
#     ax3.set_title("Canyon, no tide")
#     ax3.tick_params(axis='y', labelleft=False)
#     ax3.set_xlabel("x (km)")
#     ax3.set_aspect(1)
    
#     # Plot 4: Canyon, tide
#     im2 = ax4.pcolormesh(x_r,y_r, Fu4t_slice*1E3, shading="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
#     ax4.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='0.5',linewidths=0.5)
#     ax4.set_title("Canyon, tide")
#     ax4.tick_params(axis='y', labelleft=False)
#     ax4.set_xlabel("x (km)")
#     ax4.set_aspect(1)
    
#     # Add colorbars
#     cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
#     cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
#     cbar_ax1.text(-0.35,-0.1,r"10$^{-3}$ Wm$^{-2}$",transform=cbar_ax1.transAxes, fontsize=13, fontweight='bold')   
#     fig.savefig(f'energy_conversion_win_num_{win}.png', dpi=200)
# plt.show()

## --------------------------------------------------------------------------------------------------
## Plot criticality at different windows
## --------------------------------------------------------------------------------------------------
# def char(ds):
#     w = 2 * np.pi / (12.4206 * 3600.)
#     f = ds.f
#     N2 = ds.N2
#     char = xr.ufuncs.sqrt(xr.ufuncs.abs((w**2-f**2)/(N2-w**2)))
#     alphax = xr.ufuncs.abs((ds.h.differentiate('xi_rho') * ds.pm)) / char
#     alphax = alphax.transpose("n", "s_rho", "eta_rho", "xi_rho")
#     char = char.transpose("n", "s_rho", "eta_rho", "xi_rho")

#     char.name = "Characterstic ray path"
#     alphax.name = "Criticality"
    
#     alphax.attrs.update({
#         'Description': 'Criticality for waves propogating in x-direction',
#     })
#     return alphax

# cmap = cmo.cm.delta

# alpha3 = char(isn)
# alpha4 = char(isc)
# alpha4t = char(isct)
# alpha3t = char(isnt)

# rho4t = isct.sigma.values
# rho3t = isnt.sigma.values
# rho4 = isc.sigma.values
# rho3 = isn.sigma.values

# # choose windows to plot
# zlev=0
# win_index = [2, 5, 7]
# h_nc = isn.h.values
# h_c = isc.h.values
# window = 50 # 5 days * 10 outputs per day
# step = 25 # 2.5 days * 10 per day
# for win in win_index:
#     # Create figure and axes once
#     fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(10,6), sharey=True)  
#     # Calculate title and time slice
#     title = f"{(step*win)*(2.4/24):.1f}-{((step*win)+window)*(2.4/24):.1f} days"      
#     ax1.pcolormesh(x_r, y_r, alpha3.values[win,zlev,:,:], shading="auto", cmap=cmap,vmin=0,vmax=2)
#     ax1.contour(x_r, y_r, alpha3.values[win,zlev,:,:], levels=[1], colors=['k'])
#     ax1.set_ylim([-50,50])
#     ax1.set_xlim([-40,0])
#     ax1.contour(x_r, y_r, h_nc,levels=[150,200,300,400],colors='0.5',linewidths=1)
#     ax1.set_ylabel("y (km)")
#     ax1.set_aspect(1)
#     ax1.set_title("No canyon, no tide", fontsize=12)
#     ax2.pcolormesh(x_r, y_r, alpha3t.values[win,zlev,:,:], shading="auto", cmap=cmap,vmin=0,vmax=2)
#     ax2.contour(x_r, y_r, alpha3t.values[win,zlev,:,:], levels=[1], colors=['k'])
#     ax2.set_ylim([-50,50])
#     ax2.set_xlim([-40,0])
#     ax2.contour(x_r, y_r, h_nc,levels=[150,200,300,400],colors='0.5',linewidths=1)
#     ax2.set_title("No canyon, tide",fontsize=12)
#     ax2.set_aspect(1)
#     ax3.pcolormesh(x_r, y_r, alpha4.values[win,zlev,:,:], shading="auto", cmap=cmap,vmin=0,vmax=2)
#     ax3.contour(x_r, y_r, alpha4.values[win,zlev,:,:], levels=[1], colors=['k'])
#     ax3.set_ylim([-50,50])
#     ax3.set_xlim([-40,0])
#     ax3.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='0.5',linewidths=1)
#     ax3.set_xlabel("x (km)")
#     ax3.set_title("Canyon, no tide",fontsize=12)
#     ax3.set_aspect(1)
#     im=ax4.pcolormesh(x_r, y_r, alpha4t.values[win,zlev,:,:], shading="auto", cmap=cmap,vmin=0,vmax=2)
#     ax4.contour(x_r, y_r, alpha4t.values[win,zlev,:,:], levels=[1], colors=['k'])
#     ax4.set_ylim([-50,50])
#     ax4.set_xlim([-40,0])
#     ax4.contour(x_r, y_r, h_c, levels=[150,200,300,400],colors='0.5',linewidths=1)
#     ax4.set_title("Canyon, tide",fontsize=12)
#     ax4.set_aspect(1)
#     cbar_ax1 = fig.add_axes([0.91, 0.1, 0.008, 0.75])
#     cb=fig.colorbar(im, cax=cbar_ax1)
#     cb.set_label(label='Criticality')
#     ax1.text(0.1,1.1,title,transform=ax1.transAxes)
#     # Add colorbars
#     fig.savefig(f'criticality_{win}.png', dpi=200)
# plt.show()
## --------------------------------------------------------------------------------------------------
 ## -------------------------------------------------------
# ## Energy conversion and criticality
# ## -------------------------------------------------------
# cmap = cmo.cm.balance

# # choose windows to plot
# win_index = [2, 5, 7]
# h_nc = isn.h.values
# h_c = isc.h.values
# x_index = 84  # choose a cross-shore section
# window = 50 # 5 days * 10 outputs per day
# step = 25 # 2.5 days * 10 per day

# for win in win_index:
#     # Create figure and axes once
#     fig = plt.figure(figsize=(10,6))
#     gs = GridSpec(nrows=1, ncols=4, figure=fig )
    
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[0, 2])
#     ax4 = fig.add_subplot(gs[0, 3])

#     # Create colorbar axes
#     #fig.subplots_adjust(right=0.8)
#     #plt.subplots_adjust(wspace=0.4)
#     cbar_ax1 = fig.add_axes([0.22, 0.08, 0.4, 0.02])
#     cbar_ax2 = fig.add_axes([0.73, 0.08, 0.17, 0.02])

#     # Calculate title and time slice
#     title = f"{(step*win)*(2.4/24):.1f}-{((step*win)+window)*(2.4/24):.1f} days"
        
#     # Extract data for this frame
#     Fu3_slice = isn.C.values[win, :, :]
#     Fu4_slice = isc.C.values[win, :, :]
#     Fu3t_slice = isnt.C.values[win, :, :]
#     Fu4t_slice = isct.C.values[win, :, :]
    
#     # Plot 1: No canyon, no tide
#     ax1.pcolormesh(x_r, y_r, Fu3_slice*1E3, shading="auto", cmap=cmap, vmin=-0.025, vmax=0.025)
#     ax1.contour(x_r, y_r, alpha3.values[win,zlev,:,:], levels=[0.8,1,1.2], colors=['b','k','g'])
#     ax1.set_title("No canyon, no tide")
#     ax1.contour(x_r, y_r, h_nc,levels=[150,200,300,400],colors='0.5',linewidths=0.5)
#     ax1.set_aspect(1)
#     ax1.set_ylabel("y (km)")
#     ax1.set_xlabel("x (km)")
#     ax1.annotate(title, xy=(-0.2, 1.1), xycoords="axes fraction", fontweight='bold', fontsize=14)
    
#     # Plot 2: No canyon, tide
#     ax2.pcolormesh(x_r, y_r, Fu3t_slice*1E3, shading="auto", cmap=cmap, vmin=-0.025, vmax=0.025)
#     ax2.contour(x_r, y_r, h_nc,levels=[150,200,300,400],colors='0.5',linewidths=0.5)
#     ax2.contour(x_r, y_r, alpha3t.values[win,zlev,:,:], levels=[0.8,1,1.2], colors=['b','k','g'])
#     ax2.set_title("No canyon, tide")
#     ax2.tick_params(axis='y', labelleft=False)
#     ax2.set_xlabel("x (km)")
#     ax2.set_aspect(1) 
    
#     # Plot 3: Canyon, no tide
#     im1 = ax3.pcolormesh(x_r, y_r, Fu4_slice*1E3, shading="auto", cmap=cmap, vmin=-0.025, vmax=0.025)
#     ax3.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='0.5',linewidths=0.5)
#     ax3.contour(x_r, y_r, alpha4.values[win,zlev,:,:], levels=[0.8,1,1.2], colors=['b','k','g'])
#     ax3.set_title("Canyon, no tide")
#     ax3.tick_params(axis='y', labelleft=False)
#     ax3.set_xlabel("x (km)")
#     ax3.set_aspect(1)
    
#     # Plot 4: Canyon, tide
#     im2 = ax4.pcolormesh(x_r,y_r, Fu4t_slice*1E3, shading="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
#     ax4.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='0.5',linewidths=0.5)
#     ax4.contour(x_r, y_r, alpha4t.values[win,zlev,:,:], levels=[0.8,1,1.2], colors=['b','k','g'])
#     ax4.set_title("Canyon, tide")
#     ax4.tick_params(axis='y', labelleft=False)
#     ax4.set_xlabel("x (km)")
#     ax4.set_aspect(1)
    
#     # Add colorbars
#     cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
#     cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
#     cbar_ax1.text(-0.35,-0.1,r"10$^{-3}$ Wm$^{-2}$",transform=cbar_ax1.transAxes, fontsize=13, fontweight='bold')   
#     fig.savefig(f'energy_conversion_alphax_win_num_{win}.png', dpi=200)
# plt.show()
## ----------------------------------------------------
##---------- Plot tides ----------------
## ----------------------------------------------------
# y_r = (y_rho[:,1]/1000)-300
# t1 = 26
# t2 = 120
# t3 = 200

# td1 = 14*10
# td2 = 16*10+1
# ad1 = 22*10
# ad2 = 24*10+1

# fig = plt.figure(figsize=(10, 3)) # c
# gs = GridSpec(nrows=1, ncols=1, figure=fig)
# time_tot = isct.ocean_time.values/(3600*24)
# tide = np.nanmean(isct.u_rho.values[:,:,0,0], axis=1)

# ax1 = fig.add_subplot(gs[0, :])
# ax1.plot(time_tot,tide, 'b.-')
# ax1.axvline(x=time_tot[t1],color='r')
# ax1.axvline(x=time_tot[t2],color='r')
# ax1.axvline(x=time_tot[t3],color='r')
# ax1.axvline(x=time_tot[td1],color='orange')
# ax1.axvline(x=time_tot[td2],color='orange')
# ax1.axvline(x=time_tot[ad1],color='g')
# ax1.axvline(x=time_tot[ad2],color='g')

# ax1.set_ylabel("U (m/s)")
# ax1.set_xlabel("Time (days)")

# fig.savefig("tide.png", dpi=200)
# plt.show()

# ### ---- density ---------
# td1 = 14*10
# td2 = 16*10+1
# ad1 = 22*10
# ad2 = 24*10+1
# rho4ti = np.nanmean(isct.sigma.values[slice(ad1,ad2),0,:,:], axis=0)
# rho3ti = np.nanmean(isnt.sigma.values[slice(ad1,ad2),0,:,:], axis=0)
# rho4i = np.nanmean(isc.sigma.values[slice(ad1,ad2),0,:,:], axis=0)
# rho3i = np.nanmean(isn.sigma.values[slice(ad1,ad2),0,:,:], axis=0)

# fig = plt.figure(figsize=(8, 10)) # constrained_layout helps with spacing
# gs = GridSpec(nrows=2, ncols=3, figure=fig) # Example: 2 rows, 2 columns

# ax2 = fig.add_subplot(gs[0,0])
# ax2.pcolormesh(x_r, y_r, rho3i, shading="auto", cmap=cmo.cm.dense,vmin=25.2,vmax=26.5)
# ax2.set_ylim([-50,50])
# ax2.set_xlim([-40,0])
# ax2.contour(x_r, y_r, h_n,levels=[150,200,300,400],colors='red',linewidths=0.5)
# ax2.set_ylabel("y (km)")
# ax2.set_title("No tide",fontsize=12)

# ax3 = fig.add_subplot(gs[0,1])
# ax3.pcolormesh(x_r, y_r, rho3ti, shading="auto", cmap=cmo.cm.dense,vmin=25.2,vmax=26.5)
# ax3.set_ylim([-50,50])
# ax3.set_xlim([-40,0])
# ax3.contour(x_r, y_r, h_n,levels=[150,200,300,400],colors='red',linewidths=0.5)
# ax3.set_title("Tide",fontsize=12)

# ax4 = fig.add_subplot(gs[0,2])
# ax4.pcolormesh(x_r, y_r, rho3i-rho3ti, shading="auto", cmap=cmo.cm.curl,vmin=-0.2,vmax=0.2)
# ax4.set_ylim([-50,50])
# ax4.set_xlim([-40,0])
# ax4.contour(x_r, y_r, h_n,levels=[150,200,300,400],colors='red',linewidths=0.5)
# ax4.set_title("No tide - tide",fontsize=12)

# ax5 = fig.add_subplot(gs[1,0])
# ax5.pcolormesh(x_r, y_r, rho4i, shading="auto", cmap=cmo.cm.dense,vmin=25.2,vmax=26.5)
# ax5.set_ylim([-50,50])
# ax5.set_xlim([-40,0])
# ax5.set_ylabel("y (km)")
# ax5.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='red',linewidths=0.5)
# #ax5.set_title("Canyon, no tide",fontsize=12)

# ax6 = fig.add_subplot(gs[1,1])
# im1=ax6.pcolormesh(x_r, y_r, rho4ti, shading="auto", cmap=cmo.cm.dense,vmin=25.2,vmax=26.5)
# ax6.set_ylim([-50,50])
# ax6.set_xlim([-40,0])
# ax6.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='red',linewidths=0.5)

# ax7 = fig.add_subplot(gs[1,2])
# im2=ax7.pcolormesh(x_r, y_r, rho4i-rho4ti, shading="auto", cmap=cmo.cm.curl,vmin=-0.2,vmax=0.2)
# ax7.set_ylim([-50,50])
# ax7.set_xlim([-40,0])
# ax7.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='red',linewidths=0.5)

# for ax in (ax2,ax3,ax4,ax5,ax6,ax7):
#     ax.set_aspect(1)
# cbar_ax1 = fig.add_axes([0.15, 0.06, 0.45, 0.01])
# cb1=fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
# #cb1.ax.tick_params(labelsize=12)
# cb1.set_label(label='Potential Density (kg m$^{-3}$)')
# cbar_ax2 = fig.add_axes([0.7, 0.06, 0.2, 0.01])
# cb2=fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
# cb2.set_label(label='Potential Density (kg m$^{-3}$)')
# #cb2.ax.tick_params(labelsize=12)
# plt.subplots_adjust(wspace=0.4, hspace=0.12) # Increase wspace and hspace
# fig.savefig("Density_topview_adv.png")
# plt.show()


## ----------------------------------------------------
##---------- Plot tides ----------------
## ----------------------------------------------------
# y_r = (y_rho[:,1]/1000)-300
# t1 = 26
# t2 = 120
# t3 = 200

# td1 = 14*10
# td2 = 16*10+1
# ad1 = 22*10
# ad2 = 24*10+1

# fig = plt.figure(figsize=(10, 3)) # c
# gs = GridSpec(nrows=1, ncols=1, figure=fig)
# time_tot = isct.ocean_time.values/(3600*24)
# tide = np.nanmean(isct.u_rho.values[:,:,0,0], axis=1)

# ax1 = fig.add_subplot(gs[0, :])
# ax1.plot(time_tot,tide, 'b.-')
# ax1.axvline(x=time_tot[t1],color='r')
# ax1.axvline(x=time_tot[t2],color='r')
# ax1.axvline(x=time_tot[t3],color='r')
# ax1.axvline(x=time_tot[td1],color='orange')
# ax1.axvline(x=time_tot[td2],color='orange')
# ax1.axvline(x=time_tot[ad1],color='g')
# ax1.axvline(x=time_tot[ad2],color='g')

# ax1.set_ylabel("U (m/s)")
# ax1.set_xlabel("Time (days)")

# fig.savefig("tide.png", dpi=200)
# plt.show()

# ### ---- TKE ---------
td1 = 14*10
td2 = 16*10+1
ad1 = 22*10
ad2 = 24*10+1
rho4ti = np.nanmean(isct.tke.values[slice(ad1,ad2),0,:,:], axis=0)*1E4
rho3ti = np.nanmean(isnt.tke.values[slice(ad1,ad2),0,:,:], axis=0)*1E4
rho4i = np.nanmean(isc.tke.values[slice(ad1,ad2),0,:,:], axis=0)*1E4
rho3i = np.nanmean(isn.tke.values[slice(ad1,ad2),0,:,:], axis=0)*1E4

fig = plt.figure(figsize=(8, 10)) # constrained_layout helps with spacing
gs = GridSpec(nrows=2, ncols=3, figure=fig) # Example: 2 rows, 2 columns

ax2 = fig.add_subplot(gs[0,0])
ax2.pcolormesh(x_r, y_r, rho3i, shading="auto", cmap=cmo.cm.speed,vmin=0,vmax=5)
ax2.set_ylim([-50,50])
ax2.set_xlim([-40,0])
ax2.contour(x_r, y_r, h_n,levels=[150,200,300,400],colors='red',linewidths=0.5)
ax2.set_ylabel("y (km)")
ax2.set_title("No tide",fontsize=12)

ax3 = fig.add_subplot(gs[0,1])
ax3.pcolormesh(x_r, y_r, rho3ti, shading="auto", cmap=cmo.cm.speed,vmin=0,vmax=5)
ax3.set_ylim([-50,50])
ax3.set_xlim([-40,0])
ax3.contour(x_r, y_r, h_n,levels=[150,200,300,400],colors='red',linewidths=0.5)
ax3.set_title("Tide",fontsize=12)

ax4 = fig.add_subplot(gs[0,2])
ax4.pcolormesh(x_r, y_r, rho3i-rho3ti, shading="auto", cmap=cmo.cm.curl,vmin=-0.75,vmax=0.75)
ax4.set_ylim([-50,50])
ax4.set_xlim([-40,0])
ax4.contour(x_r, y_r, h_n,levels=[150,200,300,400],colors='red',linewidths=0.5)
ax4.set_title("No tide - tide",fontsize=12)

ax5 = fig.add_subplot(gs[1,0])
ax5.pcolormesh(x_r, y_r, rho4i, shading="auto", cmap=cmo.cm.speed,vmin=0,vmax=5)
ax5.set_ylim([-50,50])
ax5.set_xlim([-40,0])
ax5.set_ylabel("y (km)")
ax5.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='red',linewidths=0.5)
#ax5.set_title("Canyon, no tide",fontsize=12)

ax6 = fig.add_subplot(gs[1,1])
im1=ax6.pcolormesh(x_r, y_r, rho4ti, shading="auto", cmap=cmo.cm.speed,vmin=0,vmax=5)
ax6.set_ylim([-50,50])
ax6.set_xlim([-40,0])
ax6.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='red',linewidths=0.5)

ax7 = fig.add_subplot(gs[1,2])
im2=ax7.pcolormesh(x_r, y_r, rho4i-rho4ti, shading="auto", cmap=cmo.cm.curl,vmin=-0.75,vmax=0.75)
ax7.set_ylim([-50,50])
ax7.set_xlim([-40,0])
ax7.contour(x_r, y_r, h_c,levels=[150,200,300,400],colors='red',linewidths=0.5)

for ax in (ax2,ax3,ax4,ax5,ax6,ax7):
    ax.set_aspect(1)
cbar_ax1 = fig.add_axes([0.15, 0.06, 0.45, 0.01])
cb1=fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
#cb1.ax.tick_params(labelsize=12)
cb1.set_label(label='TKE (10$^{-4}$ m$^2$s$^{-2}$)')
cbar_ax2 = fig.add_axes([0.7, 0.06, 0.2, 0.01])
cb2=fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
cb2.set_label(label='TKE (10$^{-4}$ m$^2$s$^{-2}$)')
#cb2.ax.tick_params(labelsize=12)
plt.subplots_adjust(wspace=0.4, hspace=0.12) # Increase wspace and hspace
fig.savefig("TKE_topview_adv.png")
plt.show()
