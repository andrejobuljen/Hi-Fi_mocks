# import all that's needed
import sys
sys.path.insert(0,'/home/aobulj/data/Hi-Fi_mocks')
sys.path.insert(0,'../')
from lib.tng_lib import *
import numpy as np
from nbodykit.lab import *
import time
from argparse import ArgumentParser
from scipy.interpolate import interp1d
start = time.time()

comm = CurrentMPIComm.get()    
print ('comm', comm, 'comm.rank', comm.rank, 'comm.size', comm.size)
rank = comm.rank

ap = ArgumentParser()

ap.add_argument('--nmesh',
                type=int,
                default=256,
                help="Number of grid cells per side (default 256)")
                
ap.add_argument('--boxsize',
                type=float,
                default=500.,
                help="Box size [Mpc/h] (default 500 Mpc/h)")
                
ap.add_argument('--output_redshift',
                type=float,
                default=0.5,
                help="Output redshift (default 0.5)")

ap.add_argument('--output_folder',
                type=str,
                default='./output_folder',
                help="name for output folder")

cmd_args = ap.parse_args()

Nmesh = cmd_args.nmesh
BoxSize = cmd_args.boxsize
zout = cmd_args.output_redshift
output_folder = cmd_args.output_folder + '/'

##########################
# ## General parameters ###
# #########################

zic  = 99 # Quijote initial redshift
kmin = np.pi/BoxSize # kmin used in Pk measurements [h/Mpc]
axis = 2 # coordinate axis along which RSDs have been applied; matching z-axis, los=[0,0,1]
Nmufid = 6 # number of mu bins for nbodykit FFTPower, this is actually 2x the number of (positive) mu bins

print ("Generating shifted fields in redshift-space at output redshift z=%.1f, in a BoxSize L=%.1f on a Nmesh=%i^3 grid..."\
       %(zout, BoxSize, Nmesh))

# Growth factors from Abacus GrowthTable (abailable in header of IC files)
# Values below taken from this file: AbacusSummit_small_c000_ph3000_ic/ic_dens_N576.asdf
# Assuming zout = 0.5!!!
Dic  = 0.012842674496688728
Dout = 0.769423706351294
prefactor = Dout/Dic
print ("prefactor = %.4f"%prefactor)

# Parameter paths/options for saving/plotting
save_outputs = True
plot = True
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#################
### Main part ###
#################

# Path to IC & galaxy catalog files
# Change this!
fpath = '/home/aobulj/scratch/'

# Load linear overdensity field at zic 
print ('Loading initial density field... ')
dlin = np.load(fpath + 'AbacusSummit_small_c000_ph3545_ics.npy') 
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# resample to input Nmesh
print ('Resampling to input density field... ')
dlin = ArrayMesh(dlin, BoxSize).paint(mode='real', Nmesh=Nmesh).r2c()
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Compute shifted fields
print ('Computing shifted fields... ')
dz, d1, d2, dG2, dG2par, d3 = generate_fields_rsd_new(dlin, c, zic, zout, comm=comm)
p1 = FFTPower(d1, mode='2d', kmin=kmin, Nmu=Nmufid, poles=[0,2])
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Orthogonalize shifted fields
print ('Orthogonalizing shifted fields... ')
d2, dG2, d3 = orthogonalize_rsd(d1, d2, dG2, d3, Nmufid)
# d2, dG2, d3 = orthogonalize_rsd_loc(d1, d2, dG2, d3, Nmu=Nmufid, kmin=kmin)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

#################
### Load data ###
#################

print ('Loading halo catalog... ')

# #################
# RSD-part
# #################
# Here you need to put the redshift-space positions of halos, where distortions have been applied along axis axis (default along z; variable name from above)
# following this from before:
# print ('Loading galaxy catalog... ')
# gal_cat = np.load(fpath + 'AbacusSummit_small_c000_ph3545_pos.npy')
# dtype = np.dtype([('Position', ('f8', 3)),])
# cat = np.empty((gal_cat.shape[0],), dtype=dtype)
# cat['Position']  = gal_cat[:]
# #################

# make a catalog with halo positions in redshift-space
cat = ArrayCatalog(cat, BoxSize=BoxSize * np.ones(3), Nmesh=Nmesh)
cat = cat.to_mesh(compensated=True).paint() - 1.0
print('cat.mean', cat.cmean())
nbar = gal_cat.shape[0]/BoxSize**3    
print('Nh = %.2f, 1/nbar = %.2f'%(cat.size, 1/nbar))

dtype = np.dtype([('Position', ('f8', 3)),])
cat = np.empty((pos_h.shape[0],), dtype=dtype)
cat['Position'] = pos_h
cat = ArrayCatalog(cat, BoxSize=BoxSize * np.ones(3), Nmesh=Nmesh)
cat = cat.to_mesh(compensated=True).paint() - 1.0
print('cat.mean', cat.cmean())
print('Nh = %.2f, 1/nbar = %.2f'%(valid[valid].size, 1/nbar))

delta_h = ArrayMesh(cat, BoxSize)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Compute various Pk and cross-Pk
print ('computing transfer functions... ')
ph_fin  = FFTPower(delta_h, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)
pz = FFTPower(dz, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)
kk = pz.power.coords['k']
Nmu0 = int(pz.attrs['Nmu']/2)
mus = pz.power.coords['mu'][Nmu0:]
pz.power = pz.power[:,Nmu0:]

pd1ort = FFTPower(d1, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)
pd2ort = FFTPower(d2, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)
pG2ort = FFTPower(dG2,mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)
pd3ort = FFTPower(d3, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)

rsd_minus_ck = (delta_h.to_field(mode='complex') - (dz-3./7.*fout*dG2par)).c2r()

ph_d1ort = FFTPower(rsd_minus_ck, mode='2d', second=d1, kmin=kmin, poles=[0,2], Nmu=Nmufid)
ph_d2ort = FFTPower(rsd_minus_ck, mode='2d', second=d2, kmin=kmin, poles=[0,2], Nmu=Nmufid)
ph_dG2ort= FFTPower(rsd_minus_ck, mode='2d', second=dG2,kmin=kmin, poles=[0,2], Nmu=Nmufid)
ph_d3ort = FFTPower(rsd_minus_ck, mode='2d', second=d3, kmin=kmin, poles=[0,2], Nmu=Nmufid)

# Transfer functions
beta1 = ph_d1ort.power['power'].real[:,Nmu0:]/pd1ort.power['power'].real[:,Nmu0:]
beta2 = ph_d2ort.power['power'].real[:,Nmu0:]/pd2ort.power['power'].real[:,Nmu0:]
betaG2 = ph_dG2ort.power['power'].real[:,Nmu0:]/pG2ort.power['power'].real[:,Nmu0:]
beta3 = ph_d3ort.power['power'].real[:,Nmu0:]/pd3ort.power['power'].real[:,Nmu0:]

beta1_interkmu = interp1d_manual_k_binning(pz.power['k'], beta1, fill_value=[beta1[0][0],beta1[-1][0]], Ngrid=pz.attrs['Nmesh'], L = pz.attrs['BoxSize'][0], Pkref=ph_fin, kind='manual_Pk_k_mu_bins')
beta2_interkmu = interp1d_manual_k_binning(pz.power['k'], beta2, fill_value=[beta2[0][0],beta2[-1][0]], Ngrid=pz.attrs['Nmesh'], L = pz.attrs['BoxSize'][0], Pkref=ph_fin, kind='manual_Pk_k_mu_bins')
betaG2_interkmu = interp1d_manual_k_binning(pz.power['k'], betaG2, fill_value=[betaG2[0][0],betaG2[-1][0]], Ngrid=pz.attrs['Nmesh'], L = pz.attrs['BoxSize'][0], Pkref=ph_fin, kind='manual_Pk_k_mu_bins')
beta3_interkmu = interp1d_manual_k_binning(pz.power['k'], beta3, fill_value=[beta3[0][0],beta3[-1][0]], Ngrid=pz.attrs['Nmesh'], L = pz.attrs['BoxSize'][0], Pkref=ph_fin, kind='manual_Pk_k_mu_bins')
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

def rsd_filter_beta1(k3vec, val):
    absk = (sum(ki**2 for ki in k3vec))**0.5  # absk on the mesh
    # Dont use absk[absk==0]=1 b/c interp does not allow k=1.
    with np.errstate(invalid='ignore',
                     divide='ignore'):
        mu = sum(k3vec[i] * ph_fin.attrs['los'][i] for i in range(3)) / absk
    return beta1_interkmu(absk, mu) * val

def rsd_filter_beta2(k3vec, val):
    absk = (sum(ki**2 for ki in k3vec))**0.5  # absk on the mesh
    # Dont use absk[absk==0]=1 b/c interp does not allow k=1.
    with np.errstate(invalid='ignore',
                     divide='ignore'):
        mu = sum(k3vec[i] * ph_fin.attrs['los'][i] for i in range(3)) / absk
    return beta2_interkmu(absk, mu) * val

def rsd_filter_betaG2(k3vec, val):
    absk = (sum(ki**2 for ki in k3vec))**0.5  # absk on the mesh
    # Dont use absk[absk==0]=1 b/c interp does not allow k=1.
    with np.errstate(invalid='ignore',
                     divide='ignore'):
        mu = sum(k3vec[i] * ph_fin.attrs['los'][i] for i in range(3)) / absk
    return betaG2_interkmu(absk, mu) * val

def rsd_filter_beta3(k3vec, val):
    absk = (sum(ki**2 for ki in k3vec))**0.5  # absk on the mesh
    # Dont use absk[absk==0]=1 b/c interp does not allow k=1.
    with np.errstate(invalid='ignore',
                     divide='ignore'):
        mu = sum(k3vec[i] * ph_fin.attrs['los'][i] for i in range(3)) / absk
    return beta3_interkmu(absk, mu) * val

print ('making best-fit field... ')
beta1d1ort = d1.apply(rsd_filter_beta1, kind='wavenumber')
beta1d1ort[np.isnan(beta1d1ort)]=0+0j

beta2d2ort = d2.apply(rsd_filter_beta2, kind='wavenumber')
beta2d2ort[np.isnan(beta2d2ort)]=0+0j

betaG2dG2ort = dG2.apply(rsd_filter_betaG2, kind='wavenumber')
betaG2dG2ort[np.isnan(betaG2dG2ort)]=0+0j

beta3d3ort = d3.apply(rsd_filter_beta3, kind='wavenumber')
beta3d3ort[np.isnan(beta3d3ort)]=0+0j

# Best-fit fields including different bias operators
final_field = dz-3./7.*fout*dG2par + beta1d1ort + beta2d2ort + betaG2dG2ort + beta3d3ort
# final_field_quad = dz-3./7.*fout*dG2par + beta1d1ort + beta2d2ort + betaG2dG2ort
# final_field_lin = dz-3./7.*fout*dG2par + beta1d1ort

diff = (delta_h.to_field(mode='complex') - final_field).c2r()
# diff_quad = (delta_h.to_field(mode='complex') - final_field_quad).c2r()
# diff_lin = (delta_h.to_field(mode='complex') - final_field_lin).c2r()


# compute Pk of the residual
perr = FFTPower(diff, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)
# perr_quad = FFTPower(diff_quad, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)
# perr_lin = FFTPower(diff_lin, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)

pbf = FFTPower(final_field, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)
# pbf_quad = FFTPower(final_field_quad, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)
# pbf_lin = FFTPower(final_field_lin, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid)

phbf = FFTPower(delta_h, mode='2d', second=final_field, kmin=kmin, poles=[0,2], Nmu=Nmufid)
rh_bf = phbf.power['power'][:,Nmu0:].real/(ph_fin.power['power'][:,Nmu0:].real*pbf.power['power'][:,Nmu0:].real)**0.5

path_to_save = output_folder + "RSD_Abacus_zout_%.1f_grid_Nmesh_%i"%(zout, Nmesh)
results = kk, mus, perr.power['power'][:,Nmu0:].real, ph_fin.power['power'][:,Nmu0:].real, pbf.power['power'][:,Nmu0:].real, \
            rh_bf, beta1, beta2, betaG2, beta3
if not os.path.exists(path_to_save):
    np.save(path_to_save, results)

print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# plot halo field, best-fit field & difference
nslice = 25
titles = np.array(['truth', 'bestfit', 'diff'])
imshow_kw = dict(interpolation='none', cmap='RdBu_r', vmin=-1, vmax=10, extent=(0,BoxSize,0,BoxSize), origin='lower')
plt.figure(figsize=(15,4))
plt.subplot(131)
cax = plt.imshow(delta_h.apply(Gaussian(1)).paint()[:int(nslice*Nmesh/256),:,:].mean(axis=0).T, **imshow_kw)
plt.colorbar(cax)
plt.title('$\\delta_\\mathrm{h}^\\mathrm{truth}$')
plt.xlabel("$y\,[h^{-1}\,\\mathrm{Mpc}]$")
plt.ylabel("$z\,[h^{-1}\,\\mathrm{Mpc}]$")
plt.subplot(132)
cax = plt.imshow(ArrayMesh(final_field, BoxSize).apply(Gaussian(1)).paint()[:int(nslice*Nmesh/256),:,:].mean(axis=0).T, **imshow_kw)
plt.colorbar(cax)
plt.title('$\\delta_\\mathrm{h}^\\mathrm{best-fit}$')
plt.xlabel("$y\,[h^{-1}\,\\mathrm{Mpc}]$")
plt.ylabel("$z\,[h^{-1}\,\\mathrm{Mpc}]$")
plt.subplot(133)
cax = plt.imshow(ArrayMesh(diff, BoxSize).apply(Gaussian(1)).paint()[:int(nslice*Nmesh/256),:,:].mean(axis=0).T, **imshow_kw)
plt.colorbar(cax)
plt.title('$\\delta_\\mathrm{h}^\\mathrm{truth} - \\delta_\\mathrm{h}^\\mathrm{best-fit}$')
plt.xlabel("$y\,[h^{-1}\,\\mathrm{Mpc}]$")
plt.ylabel("$z\,[h^{-1}\,\\mathrm{Mpc}]$")
plt.savefig(output_folder + 'truth_rsd_bestfitcubic_diff_z=%.1f_yz_Nmesh_%i_sim_%i.pdf'%(zout, Nmesh, sim), bbox_inches='tight')
plt.close()

# ### Plus ###

fig, ax = plt.subplots(1,4, sharex=True, figsize=(16,3))

for i in range(Nmu0):
    ax[0].semilogx(kk, beta1[:,i], 'C0', alpha=(i+1)/Nmu0)
    ax[1].semilogx(kk, beta2[:,i], 'C1', alpha=(i+1)/Nmu0)
    ax[2].semilogx(kk, betaG2[:,i], 'C2', alpha=(i+1)/Nmu0)
    ax[3].semilogx(kk, beta3[:,i], 'C3', alpha=(i+1)/Nmu0)
ax[0].set_ylabel('$\\beta_1$', fontsize=12)
ax[1].set_ylabel('$\\beta_2$')
ax[2].set_ylabel('$\\beta_{\\mathcal{G}_2}$')
ax[3].set_ylabel('$\\beta_3$')
plt.subplots_adjust(wspace=0.5, hspace=0)
ax[0].plot([],[], 'k', alpha=((0+1)/Nmu0), label='$\\mu=%.2f$'%mus[0])
ax[0].plot([],[], 'k', alpha=((1+1)/Nmu0), label='$\\mu=%.2f$'%mus[1])
ax[0].plot([],[], 'k', alpha=((2+1)/Nmu0), label='$\\mu=%.2f$'%mus[2])
# ax[0].axhline(b1-1)
for axi in ax.reshape(-1): 
    axi.set_xlim(right=1)
    axi.set_xlabel("$k\,[h\,\mathrm{Mpc}^{-1}]$")
ax[0].legend(loc=0)
fig.suptitle("Abacus, $z=%.1f$, $10^{%.2f}<M_h[M_\\odot/h]<10^{%.2f}$"%(zout, np.log10(Mh_bins[0]), np.log10(Mh_bins[1])))
plt.savefig(output_folder + 'betas_rsd_z=%.1f_yz_Nmesh_%i_sim_%i.pdf'%(zout, Nmesh, sim), bbox_inches='tight')
plt.close()


plt.figure(figsize=(8,5))
for i in range(Nmu0):
    plt.loglog(kk, ph_fin.power['power'][:,Nmu0:].real[:,i], 'C0', alpha=((i+1)/Nmu0))
    plt.loglog(kk, pbf.power['power'][:,Nmu0:].real[:,i], 'C0:', alpha=((i+1)/Nmu0))
    plt.loglog(kk, perr.power['power'][:,Nmu0:].real[:,i], 'C1', alpha=((i+1)/Nmu0))

plt.plot([],[], 'C0', lw=2, label='HI truth')
plt.plot([],[], 'C0:', label='HI model (cubic bias)')
plt.plot([],[], 'C1', lw=1, label='$P_\\mathrm{err}$')

plt.plot([],[], 'k', alpha=((0+1)/Nmu0), label='$\\mu=%.2f$'%mus[0])
plt.plot([],[], 'k', alpha=((1+1)/Nmu0), label='$\\mu=%.2f$'%mus[1])
plt.plot([],[], 'k', alpha=((2+1)/Nmu0), label='$\\mu=%.2f$'%mus[2])
plt.xlim(right=1)
plt.title("Abacus, $z=%.1f$, $10^{%.2f}<M_h[M_\\odot/h]<10^{%.2f}$"%(zout, np.log10(Mh_bins[0]), np.log10(Mh_bins[1])))
plt.xlabel("$k\,[h\,\mathrm{Mpc}^{-1}]$", fontsize=14)
plt.ylabel("$P\,[h^{-3}\mathrm{Mpc}^3]$", fontsize=14)
plt.tick_params(labelsize=12)
plt.legend(loc=0, ncol=2, frameon=True)
plt.axhline(1/nbar, c='gray')
plt.savefig(output_folder + 'Abacus_RSD_Pk_z=%.1f_yz_Nmesh_%i_sim_%i.pdf'%(zout, Nmesh, sim), bbox_inches='tight')
plt.close()