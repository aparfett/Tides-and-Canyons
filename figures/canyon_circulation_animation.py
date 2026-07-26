import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import cmocean as cmo
import matplotlib as mpl
import seaborn as sns

sns.set_context('notebook', font_scale=1.2)
# Load datasets
dst = xr.open_dataset("ds2t_wnd5_st25.nc", decode_times=False)
ist = xr.open_dataset("ds4t_wnd5_st25.nc", decode_times=False)

dsn = xr.open_dataset("ds2_wnd5_st25.nc", decode_times=False)
isn = xr.open_dataset("ds4_wnd5_st25.nc", decode_times=False)

# Grid and bathymetry
x_rho = isn["x_rho"].values
y_rho = isn["y_rho"].values
y_r = (y_rho[:, 1]/1000) - 300  # center around canyon
x_r = (x_rho[1, :]/1000) - 155

# Parameters
variable = "w"  # Choose "u" for cross-shelf velocity or "w" for vertical velocity
x_index = 84 #84 o 70  # choose a cross-shore section
cmap = cmo.cm.balance
cmap2 = cmo.cm.curl
bgcolor = 'tan'
clim = 20. if variable == "u" else 2.0  # adjust based on variable
clim2 = 4. if variable == "u" else 1.0
rho_levs = [24.6, 24.75, 24.9, 25.05, 25.2, 25.35, 25.5, 25.65, 25.8, 25.95, 26.1, 26.25, 26.4, 26.55, 26.7, 26.85]
rho_color = 'k'
fact = 2 if variable == "u" else 3  # for u: cm/s, w: mm/s

# Time steps and frame settings
n_days = 25
frame_duration_hours = 2.5
total_hours = n_days * 24
# frame step corresponds to 2.5h; total frames = 240 for 25 days
total_frames = int(total_hours / frame_duration_hours)

# Pre-compute data for IS shelf
def get_variable_data(ds, var, t):
    if var == "u":
        return ds.u_rho.isel(ocean_time=t, xi_rho=x_index).values
    elif var == "w":
        return ds.w_rho.isel(ocean_time=t, xi_rho=x_index).values

def get_sigma_data(ds, t):
    return ds.sigma.isel(ocean_time=t, xi_rho=x_index).values

def get_z_data(ds, t):
    return ds.z_rho.isel(ocean_time=t, xi_rho=x_index).values

# Create time array for wind forcing
wind_hours = np.arange(0, 25*24 + 1, 0.1)  # fine resolution (0.1 h)
wind_days = wind_hours / 24.0

def wind_forcing(t_days):
    if t_days <= 5:
        return 0.0
    elif t_days <= 10:
        return 0.03 * ((t_days - 5.0) / 5.0)
    else:
        return 0.03

wind_values = np.vectorize(wind_forcing)(wind_days)

# Create figure with top row wind and bottom row velocity panels
total_height = 7.0
fig = plt.figure(figsize=(13.0, total_height), constrained_layout=False)
gs = GridSpec(nrows=2, ncols=3, height_ratios=[1, 4], figure=fig)

ax_wind = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, 0])  # No tide
ax2 = fig.add_subplot(gs[1, 1])  # Tide
ax3 = fig.add_subplot(gs[1, 2])  # Difference
ax1.set_aspect(0.25) #0.25 0 0.15
ax2.set_aspect(0.25)
ax3.set_aspect(0.25)

# Add annotated labels for wind axis
ax_wind.set_xlim(0, 25)
ax_wind.set_ylim(0, 0.035)
ax_wind.set_ylabel('Wind stress (N/m$^2$)')
ax_wind.set_xlabel('Time (days)')
ax_wind.plot(wind_days, wind_values, color='royalblue', lw=2)
line_wind = ax_wind.axvline(0, color='red', lw=2)
ax_wind.grid(True, alpha=0.3)

# Colorbars
norm = mpl.colors.Normalize(vmin= -clim, vmax=clim)
cbar_ax = fig.add_axes([0.17, 0.1, 0.4, 0.02])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=cbar_ax, orientation='horizontal', format='%1.1f',
                  label=r'$10^{-' + str(fact) + '}$ m/s')

norm2 = mpl.colors.Normalize(vmin=-clim2, vmax=clim2)
cbar_ax2 = fig.add_axes([0.69, 0.1, 0.2, 0.02])
cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2),
                   cax=cbar_ax2, orientation='horizontal', format='%1.1f',
                   label=r'$10^{-' + str(fact) + '}$ m/s')

# Update function for animation
def update(frame):
    # Each frame advances 2.5 hours
    t_days = frame * frame_duration_hours / 24.0
    t_index = int(round(frame * frame_duration_hours / 2.4))
    t_index = min(t_index, isn.sizes['ocean_time'] - 1)

    # Clear velocity axes
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Update wind time line (line_xdata must be sequence)
    line_wind.set_xdata([t_days, t_days])

    # Get data for this time step (nearest model step)
    uisn = get_variable_data(isn, variable, t_index)
    uist = get_variable_data(ist, variable, t_index)
    sigisn = get_sigma_data(isn, t_index)
    sigist = get_sigma_data(ist, t_index)
    zisn = get_z_data(isn, t_index)
    zist = get_z_data(ist, t_index)

    y = y_r[:]

    # Plot no tide
    ax1.pcolormesh(y, zisn, uisn * (10**fact), shading="auto",
                   cmap=cmap, vmin=-clim, vmax=clim)
    ax1.contour(np.tile(y, (zisn.shape[0], 1)), zisn, sigisn,
                levels=rho_levs, colors=rho_color, linewidths=0.7)
    ax1.set_title("IS - No tide")
    ax1.set_facecolor(bgcolor)
    ax1.set_ylim([-200, 0])
    ax1.set_xlim([-30, 30])
    ax1.xaxis.set_inverted(True)
    ax1.set_ylabel("Depth (m)")
    ax1.set_xlabel("y (km)")

    # Plot tide
    ax2.pcolormesh(y, zist, uist * (10**fact), shading="auto",
                   cmap=cmap, vmin=-clim, vmax=clim)
    ax2.contour(np.tile(y, (zist.shape[0], 1)), zist, sigist,
                levels=rho_levs, colors=rho_color, linewidths=0.7)
    ax2.set_title("IS - Tide")
    ax2.set_facecolor(bgcolor)
    ax2.set_ylim([-200, 0])
    ax2.set_xlim([-30, 30])
    ax2.xaxis.set_inverted(True)
    ax2.set_yticklabels([])
    ax2.set_xlabel("y (km)")

    # Plot difference
    diff = (uisn - uist) * (10**fact)
    ax3.pcolormesh(y, zisn, diff, shading="auto",
                   cmap=cmap2, vmin=-clim2, vmax=clim2)
    ax3.set_title("IS - Difference (No tide - Tide)")
    ax3.set_facecolor(bgcolor)
    ax3.set_ylim([-200, 0])
    ax3.set_xlim([-30, 30])
    ax3.xaxis.set_inverted(True)
    ax3.set_yticklabels([])
    ax3.set_xlabel("y (km)")

    # Title with day
    ax1.text(1, 2, f"t = {t_days:.2f} days - {variable.lower()} velocity", ha='center', va='center', fontsize=16, transform=ax1.transAxes, fontweight='bold')

    return [line_wind]

# Create animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=200,  # milliseconds between frames
    blit=False,
    repeat=True
)

# Save to MP4
writer = animation.FFMpegWriter(fps=4, metadata={"title": f"{variable.upper()} Velocity Animation"})
ani.save(f"canyon_circulation_{variable}_animation.mp4", writer=writer, dpi=200 )

plt.show()