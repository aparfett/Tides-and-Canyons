import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import cmocean as cmo
import matplotlib as mpl
import seaborn as sns

sns.set_context('notebook', font_scale=1.15)

# Load datasets
dst = xr.open_dataset("ds2t_wnd5_st25.nc", decode_times=False)
ist = xr.open_dataset("ds4t_wnd5_st25.nc", decode_times=False)

dsn = xr.open_dataset("ds2_wnd5_st25.nc", decode_times=False)
isn = xr.open_dataset("ds4_wnd5_st25.nc", decode_times=False)

# Grid and bathymetry
x_rho = isn["x_rho"].values
y_rho = isn["y_rho"].values
x_r = (x_rho[1, :]/1000) - 155
y_r = (y_rho[:, 1]/1000) - 300

# Load bathymetry for masking land
h_isn = -isn["h"].values  # (y, x)
h_dsn = -dsn["h"].values  # (y, x)

# Parameters
variable = "w"  # Choose "u" for cross-shelf velocity or "w" for vertical velocity
z_target = -100.0  # Target depth (negative)
cmap = cmo.cm.balance
cmap2 = cmo.cm.curl
bgcolor = 'tan'
clim = 20. if variable == "u" else 2.0
clim2 = 4. if variable == "u" else 1.0
fact = 2 if variable == "u" else 3  # for u: cm/s, w: mm/s

# Create land mask (land where h >= -100)
land_mask_isn = h_isn >= z_target
land_mask_dsn = h_dsn >= z_target

# Time array for wind forcing
wind_hours = np.arange(0, 25*24 + 1, 0.1)
wind_days = wind_hours / 24.0

def wind_forcing(t_days):
    if t_days <= 5:
        return 0.0
    elif t_days <= 10:
        return 0.03 * ((t_days - 5.0) / 5.0)
    else:
        return 0.03

wind_values = np.vectorize(wind_forcing)(wind_days)

# Time steps and frame settings
n_days = 25
frame_duration_hours = 2.5
total_hours = n_days * 24
total_frames = int(total_hours / frame_duration_hours)

# Function to extract data at a target depth
def get_data_at_depth(ds, var, t, z_target, land_mask):
    """
    Extract variable at a specific depth by finding closest z_rho level
    
    Parameters:
    -----------
    ds : xarray.Dataset
    var : str, "u" or "w"
    t : int, time index
    z_target : float, target depth (negative, e.g., -100)
    land_mask : boolean 2D array, True where land
    
    Returns:
    --------
    data : 2D array (y, x) of variable at target depth (NaN over land)
    """
    z = ds.z_rho.isel(ocean_time=t).values  # (s_rho, y, x)
    
    if var == "u":
        field = ds.u_rho.isel(ocean_time=t).values  # (s_rho, y, x)
    elif var == "w":
        field = ds.w_rho.isel(ocean_time=t).values  # (s_rho, y, x)
    
    uu = ds.u_rho.isel(ocean_time=t).values
    vv = ds.v_rho.isel(ocean_time=t).values
    # For each (y, x) point, find the s_rho level closest to z_target
    ny, nx = z.shape[1], z.shape[2]
    data_at_depth = np.full((ny, nx), np.nan)
    u_depth = np.full((ny, nx), np.nan)
    v_depth = np.full((ny, nx), np.nan)

    for j in range(ny):
        for i in range(nx):
            # Skip land points
            if land_mask[j, i]:
                continue
            z_profile = z[:, j, i]
            # Find index of z value closest to z_target
            valid_mask = ~np.isnan(z_profile)
            if np.any(valid_mask):
                idx = np.nanargmin(np.abs(z_profile - z_target))
                data_at_depth[j, i] = field[idx, j, i]
                u_depth[j, i] = uu[idx, j, i]
                v_depth[j, i] = vv[idx, j, i]
    return data_at_depth, u_depth, v_depth

# Create figure with top row wind and bottom row velocity panels
fig = plt.figure(figsize=(8, 8), constrained_layout=False)
gs = GridSpec(nrows=2, ncols=3, height_ratios=[1, 4], figure=fig)

ax_wind = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, 0])  # No tide
ax2 = fig.add_subplot(gs[1, 1])  # Tide
ax3 = fig.add_subplot(gs[1, 2])  # Difference

# Wind panel setup
ax_wind.set_xlim(0, 25)
ax_wind.set_ylim(0, 0.035)
ax_wind.set_ylabel('Wind stress (N/m$^2$)')
ax_wind.set_xlabel('Time (days)')
ax_wind.plot(wind_days, wind_values, color='royalblue', lw=2)
line_wind = ax_wind.axvline(0, color='red', lw=2)
ax_wind.grid(True, alpha=0.3)

# Colorbars
norm = mpl.colors.Normalize(vmin=-clim, vmax=clim)
cbar_ax = fig.add_axes([0.17, 0.07, 0.4, 0.02])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=cbar_ax, orientation='horizontal', format='%1.1f',
                  label=r'$10^{-' + str(fact) + '}$ m/s')

norm2 = mpl.colors.Normalize(vmin=-clim2, vmax=clim2)
cbar_ax2 = fig.add_axes([0.69, 0.07, 0.2, 0.02])
cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2),
                   cax=cbar_ax2, orientation='horizontal', format='%1.1f',
                   label=r'$10^{-' + str(fact) + '}$ m/s')

# Update function for animation
def update(frame):
    # Each frame advances 2.5 hours
    t_days = frame * frame_duration_hours / 24.0
    t_index = int(round(frame * frame_duration_hours / 2.4))
    t_index = min(t_index, isn.sizes['ocean_time'] - 1)

    # Clear axes
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Update wind time line
    line_wind.set_xdata([t_days, t_days])

    # Get data at target depth for this time step (with land mask applied)
    uisn, udepn, vdepn = get_data_at_depth(isn, variable, t_index, z_target, land_mask_isn)
    uist, udept, vdept = get_data_at_depth(ist, variable, t_index, z_target, land_mask_isn)

    # Plot no tide (horizontal slice)
    ax1.pcolormesh(x_r, y_r, uisn * (10**fact), shading="auto",
                   cmap=cmap, vmin=-clim, vmax=clim)
    ax1.contour(x_r, y_r, dsn.h, levels=[50,100], colors=['r'])
    ax1.set_title("IS - No tide")
    ax1.set_facecolor(bgcolor)
    ax1.set_xlabel("x (km)")
    ax1.set_ylabel("y (km)")
    ax1.set_xlim([x_r.min(), x_r.max()])
    ax1.set_ylim([y_r.min(), y_r.max()])
    ax1.set_aspect(1)
    
    # Add quiver arrows
    sub = max(1, int(len(x_r) / 25))
    X, Y = np.meshgrid(x_r, y_r)
    ax1.quiver(X[::sub, ::sub], Y[::sub, ::sub], udepn[::sub, ::sub], vdepn[::sub, ::sub],
               scale=1, width=0.0025, color='k', alpha=0.8)

    # Plot tide
    ax2.pcolormesh(x_r, y_r, uist * (10**fact), shading="auto",
                   cmap=cmap, vmin=-clim, vmax=clim)
    ax2.set_title("IS - Tide")
    ax2.set_facecolor(bgcolor)
    ax2.set_xlabel("x (km)")
    ax2.set_yticklabels([])
    ax2.set_xlim([x_r.min(), x_r.max()])
    ax2.set_ylim([y_r.min(), y_r.max()])
    ax2.set_aspect(1)

    # Add quiver
    ax2.quiver(X[::sub, ::sub], Y[::sub, ::sub], udept[::sub, ::sub], vdept[::sub, ::sub],
               scale=1, width=0.0025, color='k', alpha=0.8)
    
    # Plot difference
    diff = (uisn - uist) * (10**fact)
    ax3.pcolormesh(x_r, y_r, diff, shading="auto",
                   cmap=cmap2, vmin=-clim2, vmax=clim2)
    ax3.set_title(" Diff (No tide - Tide)")
    ax3.set_facecolor(bgcolor)
    ax3.set_xlabel("x (km)")
    ax3.set_yticklabels([])
    ax3.set_xlim([x_r.min(), x_r.max()])
    ax3.set_ylim([y_r.min(), y_r.max()])
    ax3.set_aspect(1)

    # Add quiver as difference vector field
    ax3.quiver(X[::sub, ::sub], Y[::sub, ::sub], (udepn - udept)[::sub, ::sub], (vdepn - vdept)[::sub, ::sub],
               scale=1, width=0.0025, color='k', alpha=0.8)
    
    # Title with time
    ax1.text(1.8, 1.7, f"t = {t_days:.2f} days - {variable.lower()} velocity at z = {z_target} m", 
             ha='center', va='center', fontsize=14, transform=ax1.transAxes)

    return [line_wind]

# Create animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=500,  # milliseconds between frames
    blit=False,
    repeat=True
)

# Save to MP4
writer = animation.FFMpegWriter(fps=4, metadata={"title": f"{variable.upper()} at Depth {z_target}m"})
ani.save(f"canyon_circulation_horizontal_{variable}_z{int(abs(z_target))}_animation.mp4", writer=writer, dpi=200)

plt.show()
