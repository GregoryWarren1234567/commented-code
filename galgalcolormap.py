import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helperredo import *
from astropy.cosmology import FlatLambdaCDM

plt.rcParams['mathtext.fontset'] = 'cm'

# Change tick direction globally
plt.rcParams['xtick.direction'] = 'in'  # Tick direction for x-axis
plt.rcParams['ytick.direction'] = 'in'  # Tick direction for y-axis
plt.rcParams['xtick.top'] = True        # Enable top ticks
plt.rcParams['ytick.right'] = True      # Enable right ticks
plt.rcParams['xtick.major.size'] = 7    # Major tick length for x-axis
plt.rcParams['ytick.major.size'] = 7    # Major tick length for y-axis
plt.rcParams['xtick.minor.size'] = 4    # Minor tick length for x-axis
plt.rcParams['ytick.minor.size'] = 4    # Minor tick length for y-axis
plt.rcParams['axes.grid'] = True

# read the data
# path to the data files
path = './'

# define output directory path
outputdir = './'

# define filename
galpairs_name = path + 'pairs.pkl'

# read unique indices for unique galaxy pairs
uidx_name = path + 'unique_pairs.pkl'

# read mask of likely blended galaxies if provided
cfact = 1.5
mask_name = path + str(cfact) + 'x_a1pa2.pkl'

# print status
print_status(f"Reading the data from {galpairs_name}")
galpairs = pd.read_pickle(galpairs_name)

print_status(f"Reading the unique indices from {uidx_name}")
upairs = pd.read_pickle(uidx_name)

print_status(f"Reading the mask of likely blended galaxies from {mask_name}")
MASK = pd.read_pickle(mask_name)
MASK = MASK['MASK'].to_numpy()

print_status("Data read successfully")

avg = False; dz = 0.2
MASK = None
f1, f2, dx, dy, a, zl, zs, LF= importDataa(galpairs, upairs, MASK, 'background')
f1f, f2f, dxf, dyf, af, zlf, zsf, LFf= importDataa(galpairs, upairs, MASK, 'foreground')
fn = flexion_normal(f1, f2, dx, dy)
fnf = flexion_normal(f1f, f2f, dxf, dyf)
print(np.median(fn), np.mean(fn))
theta=np.sqrt(dx**2+dy**2)
thetaf=np.sqrt(dxf**2+dyf**2)
nbins=20
r=5
est=(10**-1)/zl*r
valid=(a<est)
a=a[valid]; fn=fn[valid]; zl=zl[valid]
print(np.median(fn), np.mean(fn))

xbin, ybin, zbin, count = bin_xy_avg_z(a, zl, fn, nbins, nbins)
xbin = np.repeat(xbin, nbins)
ybin = np.tile(ybin, nbins)
# zbin=zbin/zbinf
cleanz=zbin[~np.isnan(zbin)]
colormax=np.mean(cleanz)+np.std(cleanz)
extent = [xbin[0], xbin[-1], ybin[0], ybin[-1]]
print("Color map a vs theta")

fig, ax1 = plt.subplots(1)
im = ax1.imshow(zbin.T, origin='lower', cmap='magma', extent=extent, aspect='auto', vmin=0, vmax=colormax)
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label(r'Average ${\cal F}_n$')

ax1.set_xlabel(r'$a$')
ax1.set_ylabel(r'$z_l$')
# ax1.set_xscale('log')
ax1.set_title(r'2D Map of Averaged ${\cal F}_n at 11:00$')

# ax1.set_xscale('log')
print(np.max(theta))
fname = 'fnplotavstheta.png'
figname = outputdir + fname
plt.savefig(fname, dpi=500)

cut=(a<2)*(a>0.5)

fn=fn[cut]
print(np.median(fn), np.mean(fn))

# xbin, ybin, zbin, count = bin_xy_avg_z(LF, a, fn, nbins, nbins)
# xbin = np.repeat(xbin, nbins)
# ybin = np.tile(ybin, nbins)
# cleanz=zbin[~np.isnan(zbin)]
# print(np.mean(cleanz)+np.std(cleanz))
# colormax=np.mean(cleanz)+np.std(cleanz)
# extent = [xbin[0], xbin[-1], ybin[0], ybin[-1]]
# print("Color map a vs LF")

# fig, ax1 = plt.subplots(1)
# im = ax1.imshow(zbin.T, origin='lower', cmap='magma', extent=extent, aspect='auto', vmin=0, vmax=colormax)
# cbar = fig.colorbar(im, ax=ax1)
# cbar.set_label(r'Average ${\cal F}_n$')

# ax1.set_xlabel(r'$LF$')
# ax1.set_ylabel(r'$a$')
# ax1.set_yscale('log')
# ax1.set_title(r'2D Map of Averaged ${\cal F}_n$')

# # ax1.set_xscale('log')
# fname = 'logplotavsLF.png'
# figname = outputdir + fname
# plt.savefig(fname, dpi=500)


### cut
# maskg, maskl, fmaskg, fmaskl=filtering(LF, LF)
# fng=fn[maskg]; zlg=zl[maskg]; ag=a[maskg]

# xbin, ybin, zbin, count = bin_xy_avg_z(zlg, ag, fng, nbins, nbins)
# xbin = np.repeat(xbin, nbins)
# ybin = np.tile(ybin, nbins)

# extent = [xbin[0], xbin[-1], ybin[0], ybin[-1]]
# print("Color map ag vs zlg")
# cleanz=zbin[~np.isnan(zbin)]
# colormax=np.mean(cleanz)+np.std(cleanz)
# fig, ax1 = plt.subplots(1)
# im = ax1.imshow(zbin.T, origin='lower', cmap='magma', extent=extent, aspect='auto', vmin=0, vmax=colormax)
# cbar = fig.colorbar(im, ax=ax1)
# cbar.set_label(r'Average ${\cal F}_n$')

# ax1.set_xlabel(r'$z_{lg}$')
# ax1.set_ylabel(r'$a_{g}$')
# ax1.set_yscale('log')
# ax1.set_title("2D Map of Averaged Fn")

# # ax1.set_xscale('log')
# fname = 'cutgzlvsa.png'
# figname = outputdir + fname
# plt.savefig(fname, dpi=500)