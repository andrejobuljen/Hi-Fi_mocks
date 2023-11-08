# import all that's needed
import sys
sys.path.insert(0,'/home/aobulj/data/Hi-Fi_mocks')
sys.path.insert(0,'/home/aobulj/git/Pylians3/library')
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
ap.add_argument('--seed',
                type=int,
                default=5,
                help='IC seed number (default Quijote).')
                
ap.add_argument('--nmesh',
                type=int,
                default=256,
                help="Number of grid cells per side (default 256)")
                
ap.add_argument('--boxsize',
                type=float,
                default=1000.,
                help="Box size [Mpc/h] (default 1000 Mpc/h)")
                
ap.add_argument('--output_redshift',
                type=float,
                default=1,
                help="Output redshift z=0-2 (default 1)")

ap.add_argument('--output_folder',
                type=str,
                default='./output_folder',
                help="name for output folder")

cmd_args = ap.parse_args()

seed = cmd_args.seed
Nmesh = cmd_args.nmesh
BoxSize = cmd_args.boxsize
zout = cmd_args.output_redshift
output_folder = cmd_args.output_folder + '/'

##########################
# ## General parameters ###
# #########################

zic  = 127 # Quijote initial redshift
kmin = 2*np.pi/BoxSize/2 # kmin used in Pk measurements [h/Mpc]

print ("Generating shifted fields in real-space at output redshift z=%.1f, in a BoxSize L=%.1f on a Nmesh=%i^3 grid with IC seed %i..."\
       %(zout, BoxSize, Nmesh, seed))

# Quijote cosmology:
c = cosmology.Cosmology(h=0.6711, Omega0_b=0.049, Omega0_cdm=0.3175 - 0.049, n_s=0.9624, m_ncdm=[]).match(sigma8=0.834)

Plin_zout = cosmology.LinearPower(c, zout)
Plin_z0 = cosmology.LinearPower(c, 0)
Dic  = c.scale_independent_growth_factor(zic)
Dout = c.scale_independent_growth_factor(zout)

# Parameter paths/options for saving/plotting
# params_path = "./data/r_space_bestfit_params/"
save_outputs = True
plot = True
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#################
### Main part ###
#################
    
# Generate linear overdensity field at zic
print ('Generating initial density field... ')
dlin = get_dlin(seed, Nmesh, BoxSize, Plin_z0, comm)
dlin *= Dic
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Compute shifted fields
print ('Computing shifted fields... ')
d1, d2, dG2, d3 = generate_fields_new(dlin, c, zic, zout, comm=comm)
p1 = FFTPower(d1, mode='1d', kmin=kmin)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Orthogonalize shifted fields
print ('Orthogonalizing shifted fields... ')
d2, dG2, d3 = orthogonalize(d1, d2, dG2, d3)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

#################
### Load data ###
#################

Mh_bins = np.array([10**12.2, 10**12.8])#, 10**13.8, 10**15.2])

sim = 0

if zout == 3:
    snapnum = 0 
elif zout==2:
    snapnum = 1
elif zout==1:
    snapnum = 2
elif zout==0.5:
    snapnum = 3
elif zout==0:
    snapnum = 4
    
# where the Quijote FoF snapdir is 
snapdir = '/home/aobulj/data/TNG_fields/Quijote/HR/%i'%sim\

import readfof

# read FoF catalog
FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False,
                      swap=False, SFR=False, read_IDs=False)

# get the properties of the halos
pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h
mass  = FoF.GroupMass*1e10          #Halo masses in Msun/h
print ('log 10 Mh_bins = ', np.log10(Mh_bins))

# define the halo mass cut
valid = (mass>Mh_bins[0]) & (mass<Mh_bins[1])
nbar = valid[valid].size/BoxSize**3    

# make a catalog with halo positions 
dtype = np.dtype([('Position', ('f8', 3)),])
cat = np.empty((pos_h[valid].shape[0],), dtype=dtype)
cat['Position'] = pos_h[valid]
cat = ArrayCatalog(cat, BoxSize=BoxSize * np.ones(3), Nmesh=Nmesh)
cat = cat.to_mesh(compensated=True).paint() - 1.0
print('cat.mean', cat.cmean())
print('Nh = %.2f, 1/nbar = %.2f'%(valid[valid].size, 1/nbar))

delta_h = ArrayMesh(cat, BoxSize)

# Compute various Pk and cross-Pk
ph_fin  = FFTPower(delta_h, mode='1d', kmin=kmin)
kk = p1.power.coords['k']

ph_d1ort  = FFTPower(delta_h, mode='1d', second=d1, kmin=kmin)
ph_d2ort  = FFTPower(delta_h, mode='1d', second=d2, kmin=kmin)
ph_dG2ort = FFTPower(delta_h, mode='1d', second=dG2, kmin=kmin)
ph_d3ort  = FFTPower(delta_h, mode='1d', second=d3, kmin=kmin)

pd1     = FFTPower(d1, mode='1d', kmin=kmin)
pd2ort  = FFTPower(d2, mode='1d', kmin=kmin)
pdG2ort = FFTPower(dG2, mode='1d', kmin=kmin)
pd3ort  = FFTPower(d3, mode='1d', kmin=kmin)

# Transfer functions
beta1 = ph_d1ort.power['power'].real/pd1.power['power'].real
beta2 = ph_d2ort.power['power'].real/pd2ort.power['power'].real
betaG2 = ph_dG2ort.power['power'].real/pdG2ort.power['power'].real
beta3 = ph_d3ort.power['power'].real/pd3ort.power['power'].real

beta1inter  = interp1d(kk, beta1, kind='linear', fill_value=(beta1[0],beta1[-1]), bounds_error=False)
beta2inter  = interp1d(kk, beta2, kind='linear', fill_value=(beta2[0],beta2[-1]), bounds_error=False)
betaG2inter = interp1d(kk, betaG2, kind='linear', fill_value=(betaG2[0],betaG2[-1]), bounds_error=False)
beta3inter  = interp1d(kk, beta3, kind='linear', fill_value=(beta3[0],beta3[-1]), bounds_error=False)

beta11d1ort  =  d1.apply(lambda k, v: beta1inter( sum(ki ** 2 for ki in k)**0.5) * v)
beta22d2ort  =  d2.apply(lambda k, v: beta2inter( sum(ki ** 2 for ki in k)**0.5) * v)
betaG2dG2ort = dG2.apply(lambda k, v: betaG2inter(sum(ki ** 2 for ki in k)**0.5) * v)
beta33d3ort  =  d3.apply(lambda k, v: beta3inter(sum(ki ** 2 for ki in k)**0.5) * v)

# Best-fit fields including different bias operators
final_field_cub  = beta11d1ort + beta22d2ort + betaG2dG2ort + beta33d3ort
final_field_quad = beta11d1ort + beta22d2ort + betaG2dG2ort
final_field_lin  = beta11d1ort

diff_cub  = delta_h.to_field(mode='complex') - final_field_cub
diff_quad = delta_h.to_field(mode='complex') - final_field_quad
diff_lin  = delta_h.to_field(mode='complex') - final_field_lin

final_field  =  d1.apply(lambda k, v:  beta1inter(sum(ki ** 2 for ki in k)**0.5) * v)
final_field +=  d2.apply(lambda k, v:  beta2inter(sum(ki ** 2 for ki in k)**0.5) * v)
final_field += dG2.apply(lambda k, v: betaG2inter(sum(ki ** 2 for ki in k)**0.5) * v)
final_field +=  d3.apply(lambda k, v:  beta3inter(sum(ki ** 2 for ki in k)**0.5) * v)

diff = delta_h.to_field(mode='complex') - final_field

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
plt.savefig(output_folder + 'truth_bestfitcubic_diff_z=%.1f_yz_Nmesh_%i_sim_%i.pdf'%(zout, Nmesh, sim), bbox_inches='tight')
plt.close()

# compute Pk of the residual
perr = FFTPower(diff, mode='1d', kmin=kmin)
phbestfit = FFTPower(delta_h, mode='1d', second=final_field, kmin=kmin)
pbestfit = FFTPower(final_field, mode='1d', kmin=kmin)
rhbestfit = phbestfit.power['power'].real/(ph_fin.power['power'].real*pbestfit.power['power'].real)**0.5

path_to_save = output_folder + "Quijote_zout_%.1f_grid_Nmesh_%i_sim_%i"%(zout, Nmesh, sim)
results = kk, perr.power['power'].real, ph_fin.power['power'].real, phbestfit.power['power'].real, rhbestfit, beta1, beta2, betaG2, beta3
if not os.path.exists(path_to_save):
    np.save(path_to_save, results)

final_field = final_field.c2r()    
np.save(output_folder+"best_fit_field_zout_%.1f_grid_Nmesh_%i_sim_%i"%(zout, Nmesh, sim), final_field.value)

# ### Plus ###

perr_cub = FFTPower(diff_cub, mode='1d', kmin=kmin)
perr_quad = FFTPower(diff_quad, mode='1d', kmin=kmin)
perr_lin = FFTPower(diff_lin, mode='1d', kmin=kmin)

plt.figure(figsize=(8,5))
plt.semilogx(kk, beta1, label = '$\\beta_1$')
plt.semilogx(kk, beta2, label = '$\\beta_2$')
plt.semilogx(kk, betaG2, label = '$\\beta_{\\mathcal{G}_2}$')
plt.semilogx(kk, beta3, label = '$\\beta_3$')
plt.axhline(0)
plt.axvline(np.pi/(BoxSize/Nmesh))
plt.xlabel("$k\,[h\,\mathrm{Mpc}^{-1}]$", fontsize=12)
plt.ylabel("$\\beta_i(k)$", fontsize=12)
plt.legend(loc=0, ncol=2, fontsize=12, frameon=False)
plt.savefig(output_folder + 'betas_z=%.1f_yz_Nmesh_%i_sim_%i.pdf'%(zout, Nmesh, sim), bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,5))
plt.loglog(kk, Plin_zout(kk), 'k', label = '$P_{lin}$')
plt.loglog(kk, ph_fin.power['power'].real, label = 'truth')
plt.loglog(kk, phbestfit.power['power'].real, 'C0--', label = 'model (cubic bias)')
plt.loglog(kk, perr.power['power'].real, 'C1-', label = '$P_\\mathrm{err}$ (Cubic bias)')
plt.loglog(kk, perr_quad.power['power'].real, 'C1--', label = '$P_{\\rm err}\ \\mathrm{(quad)}$')
plt.loglog(kk, perr_lin.power['power'].real, 'C1:', label = '$P_{\\rm err}\ \\mathrm{(lin)}$')
plt.axhline(1/nbar, ls='--', c='gray')
plt.legend(loc=0, ncol=1, frameon=False)
plt.title("$z=%.1f$"%zout)
plt.xlabel("$k\,[h\,\mathrm{Mpc}^{-1}]$", fontsize=12)
plt.ylabel("$P\,[h^{-3}\mathrm{Mpc}^3]$")
plt.savefig(output_folder + 'Pk_z=%.1f_yz_Nmesh_%i_sim_%i.pdf'%(zout, Nmesh, sim), bbox_inches='tight')
plt.close()
