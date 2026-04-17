# This module is Aidan's notbook of teh same name split into functions.

import xarray as xr
import numpy as np
from numpy.linalg import lstsq
from scipy.signal import butter, filtfilt
import gsw


def uv_rho(ds, fill='nan'):
    """
    Add u and v velocities interpolated from staggered u/v-points
    onto rho-points.
    
    Parameters
    ----------
    ds : xarray.Dataset.
    fill : str, optional
        How to handle padding at domain edges:
        - 'nan' (default): pad with NaN
        - 'edge': copy nearest edge values

    Returns
    -------
    ds : xarray.Dataset
        Dataset with new variables `u_rho` and `v_rho`.
    """

    u = ds['u']
    v = ds['v']
    ubar = ds['ubar']
    vbar = ds['vbar']

    # ---- U → rho ----
    u_rho = 0.5 * (u.isel(xi_u=slice(0,-1)) + u.isel(xi_u=slice(1,None)))
    ubar_rho = 0.5 * (ubar.isel(xi_u=slice(0,-1)) + ubar.isel(xi_u=slice(1,None)))

    if fill == 'nan':
        u_rho = u_rho.pad({'xi_u': (1,1)}, constant_values=np.nan)
        ubar_rho = ubar_rho.pad({'xi_u': (1,1)}, constant_values=np.nan)
    elif fill == 'edge':
        u_rho = u_rho.pad({'xi_u': (1,1)}, mode='edge')
        ubar_rho = ubar_rho.pad({'xi_u': (1,1)}, mode='edge')

    u_rho = u_rho.rename({'xi_u': 'xi_rho'})
    u_rho = u_rho.rename({'eta_u': 'eta_rho'})
    ubar_rho = ubar_rho.rename({'xi_u': 'xi_rho'})
    ubar_rho = ubar_rho.rename({'eta_u': 'eta_rho'})

    # ---- V → rho ----
    v_rho = 0.5 * (v.isel(eta_v=slice(0,-1)) + v.isel(eta_v=slice(1,None)))
    vbar_rho = 0.5 * (vbar.isel(eta_v=slice(0,-1)) + vbar.isel(eta_v=slice(1,None)))

    if fill == 'nan':
        v_rho = v_rho.pad({'eta_v': (1,1)}, constant_values=np.nan)
        vbar_rho = vbar_rho.pad({'eta_v': (1,1)}, constant_values=np.nan)
    elif fill == 'edge':
        v_rho = v_rho.pad({'eta_v': (1,1)}, mode='edge')
        vbar_rho = vbar_rho.pad({'eta_v': (1,1)}, mode='edge')

    v_rho = v_rho.rename({'eta_v': 'eta_rho'})
    v_rho = v_rho.rename({'xi_v': 'xi_rho'})
    vbar_rho = vbar_rho.rename({'eta_v': 'eta_rho'})
    vbar_rho = vbar_rho.rename({'xi_v': 'xi_rho'})

    # ---- Add attributes ----
    u_rho.attrs.update({
        'long_name': 'u-velocity interpolated onto rho-points',
        'units': 'm/s'
    })
    v_rho.attrs.update({
        'long_name': 'v-velocity interpolated onto rho-points',
        'units': 'm/s'
    })
    
    # ---- Add to dataset ----
    ds = ds.assign(u_rho=u_rho, v_rho=v_rho, ubar_rho=ubar_rho, vbar_rho=vbar_rho)

    return ds


def w_rho(ds):
    """
    Add w velocity interpolated from staggered w-points
    onto rho-points.
    
    Parameters
    ----------
    ds : xarray.Dataset.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with new variables `w_rho`.
    """
    
    w = ds['w']
    
    # ---- W → rho ----
    w_rho = 0.5 * (w.isel(s_w=slice(0, -1)).assign_coords(s_w=ds.s_rho.values) + w.isel(s_w=slice(1, None)).assign_coords(s_w=ds.s_rho.values))
    w_rho = w_rho.rename({'s_w': 's_rho'})

    # ---- Add attributes ----
    w_rho.attrs.update({
        'long_name': 'w-velocity interpolated onto rho-points',
        'units': 'm/s'
    })

    # ---- Add to dataset ----
    ds = ds.assign(w_rho=w_rho)

    return ds


def s_to_z(s, Cs, h, zeta, Vtransform, hc):
    """
    Compute z_rho (depths of s-levels) following ROMS vertical transformation.

    Parameters
    ----------
    s : xarray.DataArray, shape (s_rho,)
        Non-dimensional vertical coordinate
    Cs : xarray.DataArray, shape (s_rho,)
        Vertical stretching function
    h : xarray.DataArray, shape (eta_rho, xi_rho)
        Bathymetry
    zeta : xarray.DataArray, shape (ocean_time, eta_rho, xi_rho)
        Free-surface height
    Vtransform : int
        ROMS vertical transform flag (1 or 2)
    hc : float
        Critical depth parameter

    Returns
    -------
    z : xarray.DataArray, shape (ocean_time, s_rho, eta_rho, xi_rho)
        Depth of each rho level [meters]
    """
    if Vtransform == 1:
        z0 = (s - Cs) * hc + Cs * h
        z = z0 + zeta * (1 + z0 / h)
    elif Vtransform == 2:
        z0 = (hc * s + Cs * h) / (hc + h)
        z = zeta + (zeta + h) * z0
    else:
        raise ValueError("Vtransform must be 1 or 2")

    return z


def z_rho(ds):
    """
    Implements s_to_z function using xarray.

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    ds : xarray.Dataset
        Dataset with new variables `z_rho`.
    """
    
    # ---- z → rho ----
    s_r = ds["s_rho"]
    Cs_r = ds["Cs_r"]
    h = ds["h"]
    zeta = ds["zeta"]
    Vtransform = ds["Vtransform"].values
    hc = ds["hc"].values

    s_w = ds["s_w"]
    Cs_w = ds["Cs_w"]
    
    z_rho = xr.apply_ufunc(
        s_to_z,
        s_r, Cs_r, h, zeta,
        input_core_dims=[['s_rho'], ['s_rho'], [], []],
        output_core_dims=[['s_rho']],
        kwargs={'Vtransform': int(Vtransform), 'hc': float(hc)},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
    )

    z_w = xr.apply_ufunc(
        s_to_z,
        s_w, Cs_w, h, zeta,
        input_core_dims=[['s_w'], ['s_w'], [], []],
        output_core_dims=[['s_w']],
        kwargs={'Vtransform': int(Vtransform), 'hc': float(hc)},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
    )

    z_rho.attrs.update({
        'long_name': 'Depth at rho-points',
        'units': 'meter'
    })

    z_w.attrs.update({
        'long_name': 'Depth at w-points',
        'units': 'meter'
    })

    z_rho = z_rho.transpose("ocean_time","s_rho","eta_rho", "xi_rho")
    z_w = z_w.transpose("ocean_time","s_w","eta_rho", "xi_rho")
    
    ds = ds.assign(z_rho=z_rho, z_w=z_w)

    return ds


def dz(ds):
    """
    Computes vertical grid size dz.

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    ds : xarray.Dataset
        Dataset with new variables `dz`.
    """
    dz = ds.z_w.diff('s_w')
    
    dz = xr.DataArray(
        dz.values,
        dims=('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
        coords={
            'ocean_time': ds.z_rho['ocean_time'],
            's_rho': ds.z_rho['s_rho'],
            'eta_rho': ds.z_rho['eta_rho'],
            'xi_rho': ds.z_rho['xi_rho'],
        },
        attrs={'long_name': 'Grid cell thickness', 'units': 'meter'}
    )

    ds=ds.assign(dz=dz)

    return ds


def filt(data):
    """
    Low-pass filters the data.

    Parameters
    ----------
    data : xarray.Dataset

    Returns
    -------
    ds : xarray.Dataset
        Dataset with new variables `dz`.
    """
    
    dt = 8640
    fs = 1.0 / dt  # Hz (1/s)
    nyq = 0.5 * fs

    # ---------------------------------------------------------------------
    # Low-pass filter (only if requested)
    # ---------------------------------------------------------------------

    cutoff_hr = 40.0
    low = (1 / (cutoff_hr * 3600.0)) / nyq
    b, a = butter(4, low, btype='low')

    return filtfilt(b, a, data, axis=0)


def harmonic_fit_ts(ts, t, omega):
    """
    Perform harmonic fit A*cos(ωt)+B*sin(ωt)+C0
    along time axis (axis=0).

    Parameters
    ----------
    ts : np.ndarray
        Input array of variable being filtered over time.
    t : np.ndarray
        1D array of times (in seconds) for the current window.
    omega : float
        Tidal frequency (rad/s).

    Returns
    -------
    amp : float
        tidally fitted amplitude.
    phase : float
        tidally fitted phase
    """

    mask = np.isfinite(ts)
    if np.count_nonzero(mask) < 3:
        return np.nan, np.nan

    ts = ts[mask]
    t = t[mask]
    
    M = np.vstack([np.ones_like(t), np.cos(omega*t), np.sin(omega*t)]).T
    coefs, *_ = lstsq(M, ts, rcond=None)
    c0, A, B = coefs
    amp = np.sqrt(A**2 + B**2)
    phase = np.arctan2(-B, A)  # matches cos(omega t - phase)

    return amp, phase


def fit_KE(u_prime, v_prime, w, rho, rhot, z_rho, dz, time, omega):
    """
    Computes Tidal Kinetic Energy and Available Potential Energy (APE)

    Parameters
    ----------
    u_prime : np.ndarray
        Baroclinic u velocity (u - depth-averaged u)
    v_prime : np.ndarray
        Baroclinic v velocity (v - depth-averaged v)
    w: np.ndarray
        Vertical velocity
    rho: np.ndarray
        Density
    rhot: float
        time-averaged density
    z_rho: np.ndarray
        z depth coordinate on rho coordinates
    dz: np.ndarray
        vertical grid cell sizes
    time: np.ndarray
        time
    omega : float
        Tidal frequency (rad/s).

    Returns
    -------
    KE : np.ndarray
        Tidal Kinetic Energy.
    KEbar : np.ndarray
        Depth-averaged Tidal Kinetic Energy.
    APE : np.ndarray
        Available Potential Energy.
    APEbar : np.ndarray
        Depth-averaged Available Potential Energy.
    N2 : np.ndarray
        Stratification.
    """
    g = 9.81
    
    # -------------------------------------------------------------------------
    # 1. Energy terms
    # -------------------------------------------------------------------------
    def tidal_energy(ua, va): # Calculates tidal kinetic energy
        rho0 = 1025
        return rho0 * 0.25 * (ua**2 + va**2)

    def calc_APE(zetaa, N2, omega): # Calculates APE
        rho0 = 1025
        N2_msk = N2.where(N2 > 0)
        return rho0 * 0.25 * N2_msk * (zetaa**2) # Following Hall and Carter 2010

    # -------------------------------------------------------------------------
    # 2. Derived fields and fits
    # -------------------------------------------------------------------------
    N2 = -(g/1025) * ((rhot.diff('s_rho') / z_rho.diff('s_rho')).pad(s_rho=(0,1))).mean('ocean_time') # Buoyancy stratification based on background density
    N2 = N2.assign_coords(s_rho=z_rho.s_rho.values)
    b = -(g/1025) * (rho - rhot) # buyoancy
    zeta = -b/N2 # isopycnal displacement
    
    ua, up = xr.apply_ufunc(
        harmonic_fit_ts,
        u_prime, time,
        input_core_dims=[['ocean_time'], ['ocean_time']],
        output_core_dims=[[], []],
        kwargs={'omega': omega},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    ) # u prime
    va, vp = xr.apply_ufunc(
        harmonic_fit_ts,
        v_prime, time,
        input_core_dims=[['ocean_time'], ['ocean_time']],
        output_core_dims=[[], []],
        kwargs={'omega': omega},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    ) # v prime
    wa, wp = xr.apply_ufunc(
        harmonic_fit_ts,
        w, time,
        input_core_dims=[['ocean_time'], ['ocean_time']],
        output_core_dims=[[], []],
        kwargs={'omega': omega},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float], 
    )
    zetaa, zetap = xr.apply_ufunc(
        harmonic_fit_ts,
        zeta, time,
        input_core_dims=[['ocean_time'], ['ocean_time']],
        output_core_dims=[[], []],
        kwargs={'omega': omega},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float], 
    )

    dz_mean = dz.mean(dim='ocean_time')
    
    KE = tidal_energy(ua, va)
    APE = calc_APE(zetaa, N2, omega)
    APEbar = (APE * dz_mean).sum(dim='s_rho') / dz_mean.sum(dim='s_rho') # Depth-averaging APE
    KEbar = (KE * dz_mean).sum(dim='s_rho') / dz_mean.sum(dim='s_rho') # Depth-averaging Tidal KE

    return KE, KEbar, APE, APEbar, N2


def fit_c(rho, rhot, u_rho, ut, v_rho, vt, dz, ubar_rho, vbar_rho, h, pm, pn, time, omega):
    """
    Computes Barotropic-to-Baroclinic energy conversion and M2 energy flux

    Parameters
    ----------
    rho: np.ndarray
        Density
    rhot: float
        time-averaged density
    u_rho: np.ndarray
        u velocity on density coordinates
    ut: float
        time-averaged u velocity
    v_rho: np.ndarray
        v velocity on density coordinates
    vt: float
        time-averaged v velocity
    dz: np.ndarray
        vertical grid cell sizes
    ubar_rho: np.ndarray
        depth-averaged u velocity on density coordinates
    vbar_rho: np.ndarray
        depth-averaged v velocity on density coordinates
    h: np.ndarray
        bathymetry
    pm: np.ndarray
        1/dx where dx is horizontal grid spacing
    pn: np.ndarray
        1/dy where dy is horizontal grid spacing
    time: np.ndarray
        time
    omega : float
        Tidal frequency (rad/s).

    Returns
    -------
    C : np.ndarray
        Barotropic-to-baroclinic energy conversion.
    Fu : np.ndarray
        M2 Energy flux in u-direction.
    Fv : np.ndarray
        M2 Energy flux in v direction.
    Fubar : np.ndarray
        Depth-averaged M2 Energy flux in u-direction.
    Fvbar : np.ndarray
        Depth-averaged M2 Energy flux in v-direction.
    """

    g = 9.81

    # -------------------------------------------------------------------------
    # 1. Barotropic component removal
    # -------------------------------------------------------------------------
    def u_perturb(da, dat, dz):
        u_anam = da - dat
        u_0 = (u_anam * dz).sum(dim=['s_rho']) / dz.sum(dim=['s_rho']) # Depth-averaging perturbation velocity
        return (u_anam - u_0) # Removing depth-averaged perturbation velocity

    # -------------------------------------------------------------------------
    # 2. Perturbation pressure (hydrostatic)
    # -------------------------------------------------------------------------
    def calc_wB(h,pm,pn,ubar,vbar):

        dH_dx = h.differentiate('xi_rho') * pm # Multiplying by pm = 1/dx, because derivative is being taken with respect to xi_rho' coordinate i.e. xi_rho = 1,2,3,..., and need in grid coordinates
        dH_dy = h.differentiate('eta_rho') * pn
    
        wB = (ubar * dH_dx + vbar * dH_dy) # Maybe multiply by -1?

        return wB
    
    def calc_p_b_prime(rho, rhot, dz): # Calculate perturbation pressure
        rho_anom = rho - rhot
        p1 = (g * rho_anom * dz).cumsum(dim=['s_rho'])
        p2 = (p1 * dz).sum(dim=['s_rho']) / dz.sum(['s_rho'])
        return (p1 - p2)

    # -------------------------------------------------------------------------
    # 3. Functions to calculate energy conversion and energy flux
    # -------------------------------------------------------------------------

    def calc_C_harm(pa, pp, wa, wp): # Calculate barotropic-to-baroclinic energy conversion
        pa_H = pa.isel(s_rho=0)
        pp_H = pp.isel(s_rho=0)
        return 0.5 * pa_H * wa * xr.apply_ufunc(np.cos, pp_H - wp)

    def calc_F(dz, ua, up, va, vp, pa, pp): # Calculate M2 tidal energy flux
        F_u = 0.5 * pa * ua * xr.apply_ufunc(np.cos, pp - up)
        F_v = 0.5 * pa * va * xr.apply_ufunc(np.cos, pp - vp)
        dz_mean = dz.mean(dim=['ocean_time'])
        Fubar = (F_u * dz_mean).sum(dim=['s_rho']) / dz_mean.sum(dim=['s_rho'])
        Fvbar = (F_v * dz_mean).sum(dim=['s_rho']) / dz_mean.sum(dim=['s_rho'])
        return F_u, F_v, Fubar, Fvbar

    # -------------------------------------------------------------------------
    # 4. Applying functions using xarray
    # -------------------------------------------------------------------------
    u_pert = u_perturb(u_rho, ut, dz)
    v_pert = u_perturb(v_rho, vt, dz)
    p_prime = calc_p_b_prime(rho, rhot, dz)
    wB = calc_wB(h,pm,pn,ubar_rho,vbar_rho)

    upa, upp = xr.apply_ufunc(
        harmonic_fit_ts,
        u_pert, time,
        input_core_dims=[['ocean_time'], ['ocean_time']],
        output_core_dims=[[], []],
        kwargs={'omega': omega},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    )
    vpa, vpp = xr.apply_ufunc(
        harmonic_fit_ts,
        v_pert, time,
        input_core_dims=[['ocean_time'], ['ocean_time']],
        output_core_dims=[[], []],
        kwargs={'omega': omega},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    ) # v perturbation
    pa, pp = xr.apply_ufunc(
        harmonic_fit_ts,
        p_prime, time,
        input_core_dims=[['ocean_time'], ['ocean_time']],
        output_core_dims=[[], []],
        kwargs={'omega': omega},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    ) # p perturbation
    wa, wp = xr.apply_ufunc(
        harmonic_fit_ts,
        wB, time,
        input_core_dims=[['ocean_time'], ['ocean_time']],
        output_core_dims=[[], []],
        kwargs={'omega': omega},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float],
    ) # baroclinic bottom velocity

    C = calc_C_harm(pa, pp, wa, wp)
    Fu, Fv, Fubar, Fvbar = calc_F(dz, upa, upp, vpa, vpp, pa, pp)

    return (
        C,
        Fu,
        Fv,
        Fubar,
        Fvbar,
    )


def KE_c(ds,window=50, step=25, end=9,filt=False):
    """
    Runs function fit_c with different time-windows with xarray to compute tidal energy conversion and flux

    Parameters
    ----------
    ds : xarray.Dataset
    window : int, optional
        Size of time window in days (default = 5 days)
    step : int, optional
        Step size in time-index for rolling windows (default = 15 time steps, which corresponds to 2.5 day if time steps are 2.4 hours)
    end : int, optional
        Number of windows to compute (default =  windows)
    filt : bool, optional
        Whether to apply low-pass filter to the data before computing energy conversion and flux (default = False). If True, 
        applies a low-pass Butterworth filter with a cutoff period of 40 hours to the u, v, and rho fields before computing 
        the energy conversion and flux. This can help to isolate the tidal signal by removing higher-frequency variability, 
        but it also increases the computational cost.

    Returns
    -------
    ds: xarray.Dataset
        Dataset with C, Fu, Fv, Fubar, Fvbar, u', v' for each time window.
    """

    def calc_all(ds, window, step, end, filt): 
        # --- Constants ---
        omega_M2 = 2 * np.pi / (12.4206 * 3600.0)
        h = ds.h
        pm = ds.pm
        pn = ds.pn
        if filt: # Choose whether to run low-pass filter. Default = False
            u = xr.apply_ufunc(
                filt, 
                ds.u_rho,
                input_core_dims=[['ocean_time']],
                output_core_dims=[['ocean_time']],
                vectorize=True, dask='parallelized',
                output_dtypes=[ds.u_rho.dtype],
            )
            v = xr.apply_ufunc(
                filt, 
                ds.v_rho,
                input_core_dims=[['ocean_time']],
                output_core_dims=[['ocean_time']],
                vectorize=True, dask='parallelized',
                output_dtypes=[ds.v_rho.dtype],
            )
            rho = xr.apply_ufunc(
                filt,
                ds.rho,
                input_core_dims=[['ocean_time']],
                output_core_dims=[['ocean_time']],
                vectorize=True, dask='parallelized',
                output_dtypes=[ds.rho.dtype],
            )
            ds["u_rho"] = u
            ds["v_rho"] = v
            ds["rho"] = rho
            ds["ubar_rho"] = (u * ds.dz).sum(dim=['s_rho']) / ds.dz.sum(dim=['s_rho']) # Computes depth-averaged u on density coordinates
            ds["vbar_rho"] = (v * ds.dz).sum(dim=['s_rho']) / ds.dz.sum(dim=['s_rho']) # Computes depth-averaged v on density coordinates

        u_prime = ds.u_rho - ds.ubar_rho # Baroclinic u velocity
        v_prime = ds.v_rho - ds.vbar_rho # Baroclinic v velocity
        ds=ds.assign(u_prime=u_prime,v_prime=v_prime)

        C_list = []
        Fu_list = []
        Fv_list = []
        Fubar_list = []
        Fvbar_list = []

        for i in range(end): # Run time-windows
            ds_roll = ds.isel(ocean_time=slice(step*i, (step*i)+window)) # Take slices based on time-window
            rhot = ds.rho.isel(ocean_time=slice(step*i, (step*i)+window)).mean(dim=['ocean_time'])
            ut = ds.u_rho.isel(ocean_time=slice(step*i, (step*i)+window)).mean(dim=['ocean_time'])
            vt = ds.v_rho.isel(ocean_time=slice(step*i, (step*i)+window)).mean(dim=['ocean_time'])
    
            # --- Apply over windows ---
            C_i, Fu_i, Fv_i, Fubar_i, Fvbar_i = fit_c(ds_roll.rho, rhot, ds_roll.u_rho, ut, ds_roll.v_rho, vt, ds_roll.dz, ds_roll.ubar_rho, ds_roll.vbar_rho, h, pm, pn, ds_roll.ocean_time,omega_M2)

            C_list.append(C_i)
            Fu_list.append(Fu_i)
            Fv_list.append(Fv_i)
            Fubar_list.append(Fubar_i)
            Fvbar_list.append(Fvbar_i)

        C = xr.concat(C_list, dim='n')
        Fu = xr.concat(Fu_list, dim='n')
        Fv = xr.concat(Fv_list, dim='n')
        Fubar = xr.concat(Fubar_list, dim='n')
        Fvbar = xr.concat(Fvbar_list, dim='n')

        # --- Assign window index (n) and coordinates ---
        ds_out = xr.Dataset(
            {
                "C": C,
                "Fu": Fu,
                "Fv": Fv,
                "Fubar": Fubar,
                "Fvbar": Fvbar,
                "u_prime": ds.u_prime,
                "v_prime": ds.v_prime,
            },
        )

        # --- Metadata ---
        ds_out.C.attrs.update(units="W/m²", long_name="Barotropic-to-baroclinic energy conversion")
        ds_out.Fu.attrs.update(units="W/m²", long_name="Zonal tidal energy flux")
        ds_out.Fv.attrs.update(units="W/m²", long_name="Meridional tidal energy flux")
        ds_out.Fubar.attrs.update(units="W/m", long_name="Depth-integrated zonal flux")
        ds_out.Fvbar.attrs.update(units="W/m", long_name="Depth-integrated meridional flux")
        ds_out.u_prime.attrs.update(units="m/s", long_name="Baroclinic U")
        ds_out.v_prime.attrs.update(units="m/s", long_name="Baroclinic V")

        return ds_out

    ds_out=calc_all(ds, window, step, end, filt)

    return xr.merge([ds, ds_out])


def APE_c(ds, window=5, step=25, end=9, filt=False):
    """
    Compute tidal kinetic energy, APE, and N2 terms over rolling time windows.

    Parameters
    ----------
    ds : xarray.Dataset
        ROMS dataset with rho, u_rho, v_rho, dz, ubar_rho, vbar_rho, h, pm, pn.
    window : int, optional
        Length of rolling window in time steps.
    step : int, optional
        Step size between rolling windows.
    end : int, optional
        Number of rolling windows to compute.
    filt : bool, optional
        Whether to apply low-pass filter to the data before computing energy terms (default = False). 
        If True, applies a low-pass Butterworth filter with a cutoff period of 40 hours to the u, v, 
        and rho fields before computing the energy terms. This can help to isolate the tidal signal 
        by removing higher-frequency variability, but it also increases the computational cost.  

    Returns
    -------
    xarray.Dataset
        Dataset with KE, KEbar, APE, APEbar, N2 for each time window.
    """

    def calc_all(ds, window, step, end, filt ): # end=22
        # --- Constants ---
        omega_M2 = 2 * np.pi / (12.4206 * 3600.0)
        h = ds.h
        pm = ds.pm
        pn = ds.pn
        if filt:
            u = xr.apply_ufunc(
                filt, 
                ds.u_rho,
                input_core_dims=[['ocean_time']],
                output_core_dims=[['ocean_time']],
                vectorize=True, dask='parallelized',
                output_dtypes=[ds.u_rho.dtype],
            )
            v = xr.apply_ufunc(
                filt, 
                ds.v_rho,
                input_core_dims=[['ocean_time']],
                output_core_dims=[['ocean_time']],
                vectorize=True, dask='parallelized',
                output_dtypes=[ds.v_rho.dtype],
            )
            rho = xr.apply_ufunc(
                filt,
                ds.rho,
                input_core_dims=[['ocean_time']],
                output_core_dims=[['ocean_time']],
                vectorize=True, dask='parallelized',
                output_dtypes=[ds.rho.dtype],
            )
            ds["u_rho"] = u
            ds["v_rho"] = v
            ds["rho"] = rho
            ds["ubar_rho"] = (u * ds.dz).sum(dim=['s_rho']) / ds.dz.sum(dim=['s_rho'])
            ds["vbar_rho"] = (v * ds.dz).sum(dim=['s_rho']) / ds.dz.sum(dim=['s_rho'])

        u_prime = ds.u_rho - ds.ubar_rho # Baroclinic u velocity
        v_prime = ds.v_rho - ds.vbar_rho # Barolcinic v velocity
        ds=ds.assign(u_prime=u_prime,v_prime=v_prime)

        KE_list = []
        KEbar_list = []
        APE_list = []
        APEbar_list = []
        N2_list = []

        for i in range(end):
            ds_roll = ds.isel(ocean_time=slice(step*i, (step*i)+window))
            rhot = ds.rho.isel(ocean_time=slice(step*i, (step*i)+window)).mean(dim=['ocean_time'])
           
            # --- Apply over window ---
            KE_i, KEbar_i, APE_i, APEbar_i, N2_i = fit_KE(ds_roll.u_prime, ds_roll.v_prime, ds_roll.w, ds_roll.rho, rhot, ds_roll.z_rho, ds_roll.dz, ds_roll.ocean_time, omega_M2)
            
            KE_list.append(KE_i)
            KEbar_list.append(KEbar_i)
            APE_list.append(APE_i)
            APEbar_list.append(APEbar_i)
            N2_list.append(N2_i)
            
        KE = xr.concat(KE_list, dim='n')
        KEbar = xr.concat(KEbar_list, dim='n')
        APE = xr.concat(APE_list, dim='n')
        APEbar = xr.concat(APEbar_list, dim='n')
        N2 = xr.concat(N2_list, dim='n')

        # --- Assign window index (n) and coordinates ---
        ds_out = xr.Dataset(
            {
                "KE": KE,
                "KEbar": KEbar,
                "APE": APE,
                "APEbar": APEbar,
                "N2": N2,
            },
        )

        # --- Metadata ---
        ds_out.KE.attrs.update(units="J/kg", long_name="Tidal kinetic energy")
        ds_out.KEbar.attrs.update(units="J/kg", long_name="Depth averaged tidal kinetic energy")
        ds_out.APE.attrs.update(units="J/kg", long_name="Tidal available potential energy")
        ds_out.APEbar.attrs.update(units="J/kg", long_name="Depth averaged tidal available potential energy")
        ds_out.N2.attrs.update(units="1/s²", long_name="Buyoancy stratification")

        return ds_out

    ds_out=calc_all(ds, window, step, end, filt)

    return xr.merge([ds, ds_out])


def sigma(ds): # Calculates potential density
    p = gsw.conversions.p_from_z(ds.z_rho,45)
    SA = gsw.conversions.SA_from_SP(ds.salt,p,0,45)
    CT = gsw.conversions.CT_from_t(SA,ds.temp,p)
    sigma = gsw.sigma0(SA,CT)
        
    ds = ds.assign(sigma=sigma,SA=SA,CT=CT,p=p)
    return ds


def char(ds): # Calcualtes characteristic ray path and criticality in x direction
    w = 2 * np.pi / (12.4206 * 3600.)
    f = ds.f
    N2 = ds.N2
    char = xr.ufuncs.sqrt(xr.ufuncs.abs((w**2-f**2)/(N2-w**2)))
    alphax = xr.ufuncs.abs((ds.h.differentiate('xi_rho') * ds.pm)) / char
    alphax = alphax.transpose("n", "s_rho", "eta_rho", "xi_rho")
    char = char.transpose("n", "s_rho", "eta_rho", "xi_rho")

    char.name = "Characterstic ray path"
    alphax.name = "Criticality"
    
    alphax.attrs.update({
        'Description': 'Criticality for waves propogating in x-direction',
    })
    
    ds = ds.assign(char=char,alphax=alphax)
    return ds


def tke_bar(ds): # Calculates depth averaged turbulent kinetic energy below 60m depth
    tke=ds.tke.where(ds.z_w < -60) # Taking points below 60m
    tke_rho=(tke.isel(s_w=slice(1,None)).values+tke.isel(s_w=slice(0,-1)).values) / 2 # Putting onto density coordinates
    coords = {"ocean_time": ds.ocean_time.values, "s_rho": ds.s_rho.values, "eta_rho": ds.eta_rho.values, "xi_rho": ds.xi_rho.data}
    tke_rho_da = xr.DataArray(tke_rho,coords=coords)
    tke_bar = (tke_rho_da * ds.dz).sum('s_rho') / ds.dz.sum('s_rho') # depth-averaging

    tke_bar.name = "Depth averaged TKE"
    
    ds = ds.assign(tke_bar=tke_bar)
    return ds


def uv_bar_u60(ds): # Finding depth-aveaged u and v velocities below 60m depth
    u_rho=ds.u_rho.where(ds.z_rho < -60)
    v_rho=ds.v_rho.where(ds.z_rho < -60)
    coords = {"ocean_time": ds.ocean_time.values, "s_rho": ds.s_rho.values, "eta_rho": ds.eta_rho.values, "xi_rho": ds.xi_rho.data}
    u_rho_da = xr.DataArray(u_rho,coords=coords)
    v_rho_da = xr.DataArray(v_rho,coords=coords)
    ubar_u60 = (u_rho_da * ds.dz).sum('s_rho') / ds.dz.sum('s_rho')
    vbar_u60 = (v_rho_da * ds.dz).sum('s_rho') / ds.dz.sum('s_rho')

    ubar_u60.name = "Depth averaged u (below 60m)"
    vbar_u60.name = "Depth averaged v (below 60m)"
    
    ds = ds.assign(ubar_u60=ubar_u60)
    ds = ds.assign(vbar_u60=vbar_u60)
    return ds