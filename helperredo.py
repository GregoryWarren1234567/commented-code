import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import math
from astropy.cosmology import FlatLambdaCDM
# function to print status with timestamp
def print_status(message:str):
    """
    Print status with timestamp
    """
    print(f"=> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def rms(data):
    return np.sqrt(np.mean(np.square(data)))

def rtsf(x, n=3):
    """Rounds a number to a specified number of significant figures.

    Args:
        x: The number to be rounded.
        n: The number of significant figures to round to.

    Returns:
        The rounded number.
    """
    if x == 0:
        return 0
    
    magnitude = math.floor(math.log10(abs(x)))
    factor = 10 ** (n - 1 - magnitude)
    return round(x * factor) / factor

def binvals(r, y, nbins, log):
    if log:
        r=np.log10(r)
    x1=min(r)
    x2=max(r)
    dx=(x2-x1)/nbins
    x=x1+dx*(np.arange(nbins)) # This is the midpoint
    n=np.zeros(nbins)
    mu=np.zeros(nbins)
    sig=np.zeros(nbins)
    for i in range(nbins):
        #id=where(ibin == i)[0]
        mask=(r<(x1+(i+1)*dx))*(r>(x1+(i*1)*dx))
        rm=r[mask]
        ym=y[mask]

        n[i] = len(rm)
        mu[i] = np.sum(ym)/len(ym)
        sig[i] = np.std(ym)
    return x, n, mu, sig

# defining helper functions
def flexion_normal(f1: np.ndarray, f2: np.ndarray, dx:np.ndarray, dy:np.ndarray) -> np.ndarray:
    """
    Compute the E-mode flexion
    """
    phi = np.arctan2(dy, dx)
    return -f1*np.cos(phi) - f2*np.sin(phi)

def flexion_tangent(f1: np.ndarray, f2: np.ndarray, dx:np.ndarray, dy:np.ndarray) -> np.ndarray:
    """
    Computes the B-mode flexion
    """
    phi = np.arctan2(dy, dx)
    return f1*np.sin(phi) - f2*np.cos(phi)

def shear_tangential(e1: np.ndarray, e2: np.ndarray, dx:np.ndarray, dy:np.ndarray) -> np.ndarray:
    """
    Computes the tangential shear component
    """
    phi = np.arctan2(dy, dx)
    return -e1*np.cos(2*phi) - e2*np.sin(2*phi)

def shear_cross(e1: np.ndarray, e2: np.ndarray, dx:np.ndarray, dy:np.ndarray) -> np.ndarray:
    """
    Computes the cross shear component
    """
    phi = np.arctan2(dy, dx)
    return e1*np.sin(2*phi) - e2*np.cos(2*phi)

def importDataa(galpairs, upairs, bmask, type):
    """
    Extract weak lensing signals and the positions of galaxies for the given type of galaxies
    """
    if type != 'background' and type != 'foreground' and type != 'all':
        print("Invalid **type** of galaxies. Choose either 'background', 'foreground', or 'all'")
        return
    
    idx1 = upairs['idx1'].to_numpy()
    idx2 = upairs['idx2'].to_numpy()

    # extract the data
    ra = galpairs['ra'].to_numpy(); dec = galpairs['dec'].to_numpy()
    z = galpairs['z'].to_numpy(); a = galpairs['a'].to_numpy()
    e1 = galpairs['e1'].to_numpy(); e2 = galpairs['e2'].to_numpy()
    f1 = galpairs['f1'].to_numpy(); f2 = galpairs['f2'].to_numpy()

    # create empty array to store the pointers
    pointers = np.array([], dtype=int); pointersf=np.array([], dtype=int)
    dx = np.array([], dtype=float); dy = np.array([], dtype=float)
    dtheta = np.array([], dtype=float)

    # compute delta y and delta x
    del_y = dec[idx1] - dec[idx2]
    del_x = -np.cos((dec[idx1]+dec[idx2])/2/206265)*(ra[idx1] - ra[idx2])
    theta = np.sqrt(del_x**2 + del_y**2)

    # now create a mask that follows the following conditions:
    # 1. z > 0
    # 2. a > 0.2 and a < 3
    # 3. abs(z[idx1] - z[idx2]) > 0.2
    # 4. To select background galaxies, we require that pair is at higher redshift
    # 5. To select foreground galaxies, we require that pair is at lower redshift
    # mask = (z[idx1] > 0) * (z[idx2] > 0) * (theta > thetamin)
    # mask *= (a[idx1] > amin) * (a[idx1] < amax) * (a[idx2] > amin) * (a[idx2] < amax)
    dz=0.2; amin=0.2; amax=2.5
    # mask *= (a[idx1]>0.2)*(a[idx2]>0.2)*(a[idx1]<3.0)*(a[idx2]<3.0)
    mask=(theta>4.5)*(theta<5.5)
    # mask*=((a[idx1]+a[idx2])/2<theta)
    # mask*=(a[idx1]>amin)*(a[idx2]>amin)*(a[idx1]<amax)*(a[idx2]<amax)
    mask*=(z[idx1]>0)*(z[idx2]>0)

    if bmask is not None:
        mask *= bmask[idx1] * bmask[idx2]

    bkg_idx1 = mask.copy()
    bkg_idx2 = mask.copy()
    frg_idx1 = mask.copy()
    frg_idx2 = mask.copy()

    if type == 'all':
        bkg_idx1[:] = True
        bkg_idx2[:] = True

    if type=='background':
        bkg_idx1 *= (z[idx1] > z[idx2])
        bkg_idx2 *= (z[idx2] > z[idx1])
        frg_idx1 *= (z[idx1] < z[idx2])
        frg_idx2 *= (z[idx2] < z[idx1])

    if type=='foreground':
        bkg_idx1 *= ~(z[idx1] > z[idx2])
        bkg_idx2 *= ~(z[idx2] > z[idx1])
        frg_idx1 *= ~(z[idx1] < z[idx2])
        frg_idx2 *= ~(z[idx2] < z[idx1])

    #near pointers
    pointers = np.append(pointers, idx1[bkg_idx1])
    pointers = np.append(pointers, idx2[bkg_idx2])
    #far pointer
    pointersf= np.append(pointersf, idx1[frg_idx1])
    pointersf= np.append(pointersf, idx2[frg_idx2])


    dx = np.append(dx, del_x[bkg_idx1])
    dx = np.append(dx, -del_x[bkg_idx2])
    dy = np.append(dy, del_y[bkg_idx1])
    dy = np.append(dy, -del_y[bkg_idx2])

    ra_out = ra[pointers]; dec_out = dec[pointers]
    e1_out = e1[pointers]; e2_out = e2[pointers]
    f1_out = f1[pointers]; f2_out = f2[pointers]
    a_out=a[pointers]
    zl_out=z[pointersf]; zs_out=z[pointers]
    # Dl=zl_out
    # Ds=zs_out
    # Dls=Ds-Dl
    # LF_out = Dls/Ds
    # LF=Dls/Ds
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)  # adjust to your cosmology
    # Only keep entries where source redshift > lens redshift
    valid = zs_out > zl_out

    # Apply the filter
    zl_out = zl_out[valid]
    zs_out = zs_out[valid]
    f1_out = f1_out[valid]
    f2_out = f2_out[valid]
    dx = dx[valid]
    dy = dy[valid]
    a_out = a_out[valid]

    # Now compute distances

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    Dl = cosmo.angular_diameter_distance(zl_out).value
    Ds = cosmo.angular_diameter_distance(zs_out).value
    Dls = cosmo.angular_diameter_distance_z1z2(zl_out, zs_out).value

    LF_out = Dls / Ds  # No need to clip to 0, because we filtered
    return f1_out, f2_out, dx, dy, a_out, zl_out, zs_out, LF_out

#not used at the moment
def filtering(fv, fvf):
    med=np.median(fv)
    ###background filtering
    maskg=(fv>med); maskl=(fv<med)
    #foreground filtering
    fmed=np.median(fvf)
    fmaskg=(fvf>fmed); fmaskl=(fvf<fmed)
    return maskg, maskl, fmaskg, fmaskl

#bins a 2d grid and averages a 3rd variable
#mostly used for the colormaps
def bin_xy_avg_z(x, y, z, nbins_x, nbins_y, logx=False, logy=False):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    if logx:
        valid = x > 0
        x = np.log10(x[valid])
        y = y[valid]
        z = z[valid]
    if logy:
        valid = y > 0
        y = np.log10(y[valid])
        x = x[valid]
        z = z[valid]

    x_edges = np.linspace(x.min(), x.max(), nbins_x + 1)
    y_edges = np.linspace(y.min(), y.max(), nbins_y + 1)
    
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    
    z_sum = np.zeros((nbins_x, nbins_y))
    counts = np.zeros((nbins_x, nbins_y), dtype=int)
    z_avg = np.full((nbins_x, nbins_y), np.nan)

    x_idx = np.digitize(x, x_edges) - 1
    y_idx = np.digitize(y, y_edges) - 1

    for i in range(len(z)):
        xi, yi = x_idx[i], y_idx[i]
        if 0 <= xi < nbins_x and 0 <= yi < nbins_y:
            z_sum[xi, yi] += z[i]
            counts[xi, yi] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        z_avg = z_sum / counts

    return x_centers, y_centers, z_avg, counts