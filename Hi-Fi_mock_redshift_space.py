# import all that's needed
from lib.tng_lib import *
import numpy as np
from nbodykit.lab import *
import time
from argparse import ArgumentParser
from scipy import interpolate
start = time.time()

comm = CurrentMPIComm.get()    
print ('comm', comm, 'comm.rank', comm.rank, 'comm.size', comm.size)
rank = comm.rank

ap = ArgumentParser()
ap.add_argument('--seed',
                type=int,
                default=2695896,
                help='IC seed number (default TNG300-1).')
                
ap.add_argument('--nmesh',
                type=int,
                default=256,
                help="Number of grid cells per side (default 256)")
                
ap.add_argument('--boxsize',
                type=float,
                default=205.,
                help="Box size [Mpc/h] (default 205)")
                
ap.add_argument('--output_redshift',
                type=float,
                default=1,
                help="Output redshift z=0-5 (default 1)")

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
### General parameters ###
##########################

zic = 127 # TNG initial redshift
kmin=2*np.pi/BoxSize/2 # kmin used in Pk measurements [h/Mpc]
axis = 2 # coordinate axis along which RSDs have been applied; matching z-axis, los=[0,0,1]
Nmufid = 6 # number of mu bins for nbodykit FFTPower, this is actually 2x the number of (positive) mu bins

print ("Generating HI mock in redshift-space at output redshift z=%.1f, in a BoxSize L=%.1f on a Nmesh=%i^3 grid with IC seed %i..."\
       %(zout, BoxSize, Nmesh, seed))

# Cosmology and parameters
c = cosmology.Cosmology(h=0.6774, T0_cmb=2.725, Omega0_b=0.0486, Omega0_cdm=0.2603, m_ncdm=[], n_s=0.9667, k_pivot=0.05, A_s=2.055e-9, YHe=0.24)
Plin_zout = cosmology.LinearPower(c, zout)
Plin_z0 = cosmology.LinearPower(c, 0)
Dic  = c.scale_independent_growth_factor(zic)
Dout = c.scale_independent_growth_factor(zout)
fout = c.scale_independent_growth_rate(zout)

# Parameter paths/options for saving/plotting
params_path = "./data/z_space_bestfit_params/"
save_outputs = True
plot = True
# output_folder = './output_folder/'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

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
dz, d1, d2, dG2, dG2par, d3 = generate_fields_rsd_new(dlin, c, zic, zout, comm=comm)
p1 = FFTPower(d1, mode='2d', kmin=kmin, Nmu=Nmufid, poles=[0,2])
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Orthogonalize shifted fields
print ('Orthogonalizing shifted fields... ')
d2, dG2, d3 = orthogonalize_rsd(d1, d2, dG2, d3, Nmufid)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Generate 3D HI fields
print ('Generating polynomial field... ')
HI_field_poly = rsd_polynomial_field_zout(dz, d1, d2, dG2, dG2par, d3, params_path, zout, p1, fout)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Add Perr(k,mu) noise:
print ('Adding noise... ')
HI_field_poly += noise_kmu_zout(zout, Nmesh, BoxSize, axis, fout, params_path)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

print ('Saving HI field... ')
out_fname = output_folder+"rsd_HI_field_poly_L_%.1f_Nmesh_%.1f_zout_%.1f_seed_%i"%(BoxSize,Nmesh,zout,seed)
if save_outputs: FieldMesh(HI_field_poly).save(out_fname)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# compute Pks
pHI = FFTPower(HI_field_poly, mode='2d', kmin=kmin, Nmu=Nmufid, poles=[0,2])
pHI_fname = output_folder+"pHIrsd_L_%.1f_Nmesh_%i_zout_%.1f_seed_%i"%(BoxSize,Nmesh,zout,seed)
pHI.save(pHI_fname)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

################
### Plotting ###
################

if plot:
	if comm.size==1:
		print ('Plotting ... ')
		# load
	# 	HI_field_poly = BigFileMesh(output_folder+"rsd_HI_field_poly_TNG300_L_%.1f_nbar_%.1f_zout_%.1f_seed_%i"%(BoxSize,nbar,zout,seed), dataset='Field')

		# plot smoothed fields
		R_gaussian = 2 # Mpc/h

		imshow_kw = dict(interpolation='none', cmap='RdBu_r', vmin=-1, vmax=10, extent=(0,BoxSize,0,BoxSize), origin='lower')
		cax = plt.imshow(FieldMesh(HI_field_poly).apply(Gaussian(R_gaussian)).paint()[0,:,:].T, **imshow_kw)
		plt.colorbar(cax)
		plt.title('$\\delta_\\mathrm{HI}^s$ from polynomial fits, $z=%.1f$'%zout)
		plt.xlabel("$y\,[h^{-1}\,\\mathrm{Mpc}]$")
		plt.ylabel("$z\,[h^{-1}\,\\mathrm{Mpc}]$")
		if save_outputs:
			plt.savefig(output_folder + "rsd_slices_L_%.1f_Nmesh_%.1f_zout_%.1f_seed_%i.pdf"%(BoxSize,Nmesh,zout,seed))
		plt.close()

	pHI = FFTPower.load(pHI_fname)
	Nmu0 = int(pHI.attrs['Nmu']/2)
	
	# load TNG300-1 pHI measurements and interpolate to zout in order to overplot
	path_to_pHI = './data/measurements/'
	kkload, _, _, _ = np.loadtxt(path_to_pHI + "pHI_rsd_zout_0.0.txt", unpack=True)
	z_arr = np.array([0,0.5,1,1.5,2,3,5])
	pHI_matrix = np.zeros((z_arr.size, kkload.size, 3))
	for iz, zi in enumerate(z_arr):
	    kkload, pHI_matrix[iz, :, 0], pHI_matrix[iz,:, 1], pHI_matrix[iz,:,2] = np.loadtxt(path_to_pHI + "pHI_rsd_zout_%.1f.txt"%zi, unpack=True)
	phitrue_int = interp.interp1d(z_arr, pHI_matrix, axis=0)(zout)

	plt.figure(figsize=(8,5))
	plt.loglog(pHI.poles['k'], Plin_zout(pHI.poles['k']), 'k', label = '$P_{lin}$')
	for i in range(Nmu0):
	    plt.loglog(pHI.poles['k'], pHI.power['power'][:,Nmu0+i].real, 'C0-', alpha=(i+1)/Nmu0, label='$\\mu=%.2f$'%pHI.power.coords['mu'][Nmu0+i])
	for i in range(3):
	    plt.loglog(kkload, phitrue_int[:,i], color='gray', alpha=(i+1)/Nmu0, label = 'HI from TNG300-1 (for reference)')
	plt.title("HI at $z=%.1f$"%zout)
	plt.xlabel("$k\,[h\,\mathrm{Mpc}^{-1}]$", fontsize=12)
	plt.ylabel("$P\,[h^{-3}\mathrm{Mpc}^3]$")
	plt.xlim(right=1)
	plt.ylim(1e1,5e4)
	plt.legend(loc=0)
	if save_outputs:
		plt.savefig(output_folder + "rsd_Pks_L_%.1f_Nmesh_%.1f_zout_%.1f_seed_%i.pdf"%(BoxSize,Nmesh,zout,seed))
	plt.close()
	print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

print ("Total time taken: %.1f sec."%(time.time()-start))

