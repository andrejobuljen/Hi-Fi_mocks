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

print ("Generating HI mock in real-space at output redshift z=%.1f, in a BoxSize L=%.1f on a Nmesh=%i^3 grid with IC seed %i..."\
       %(zout, BoxSize, Nmesh, seed))

# Cosmology
c = cosmology.Cosmology(h=0.6774, T0_cmb=2.725, Omega0_b=0.0486, Omega0_cdm=0.2603, m_ncdm=[], n_s=0.9667, k_pivot=0.05, A_s=2.055e-9, YHe=0.24)
Plin_zout = cosmology.LinearPower(c, zout)
Plin_z0 = cosmology.LinearPower(c, 0)
Dic  = c.scale_independent_growth_factor(zic)
Dout = c.scale_independent_growth_factor(zout)

# Parameter paths/options for saving/plotting
params_path = "./data/r_space_bestfit_params/"
save_outputs = True
plot = True
output_folder = './output_folder/'
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

# Generate 3D HI fields and save outputs
print ('Generating polynomial field... ')
HI_field_poly = polynomial_field_zout(d1, d2, dG2, d3, params_path, zout, p1)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Add isotropic stochastic noise:
print ('Adding noise... ')
HI_field_poly += noise_zout(zout, Nmesh, BoxSize, params_path)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

print ('Saving HI field... ')
out_fname = output_folder+"HI_field_poly_L_%.1f_Nmesh_%.1f_zout_%.1f_seed_%i"%(BoxSize,Nmesh,zout,seed)
if save_outputs: FieldMesh(HI_field_poly).save(out_fname)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# load and measure Pk
HI_field_poly = BigFileMesh(out_fname, dataset='Field')
print ('Computing Pk of HI field... ')
# compute Pks
pHI = FFTPower(HI_field_poly, mode='1d', kmin=kmin)
pHI_fname = output_folder+"pHI_L_%.1f_Nmesh_%i_zout_%.1f_seed_%i"%(BoxSize,Nmesh,zout,seed)
pHI.save(pHI_fname)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

################
### Plotting ###
################

if plot:
	print ('Plotting and computing power spectra ... ')
	
	if comm.size==1:
		# plot smoothed fields
		R_gaussian = 2 # Mpc/h

		imshow_kw = dict(interpolation='none', cmap='RdBu_r', vmin=-1, vmax=10, extent=(0,BoxSize,0,BoxSize), origin='lower')
		cax = plt.imshow(HI_field_poly.apply(Gaussian(R_gaussian)).paint()[0,:,:].T, **imshow_kw)
		plt.colorbar(cax)
		plt.title('$\\delta_\\mathrm{HI}^r$ from polynomial fits, $z=%.0f$'%zout)
		plt.xlabel("$y\,[h^{-1}\,\\mathrm{Mpc}]$")
		plt.ylabel("$z\,[h^{-1}\,\\mathrm{Mpc}]$")
		if save_outputs:
			plt.savefig(output_folder + "real_slices_L_%.1f_Nmesh_%.1f_zout_%.1f_seed_%i.pdf"%(BoxSize,Nmesh,zout,seed))
		plt.close()

	# load Pk
    	pHI = FFTPower.load(pHI_fname)

	# load pHI measurements and interpolate to zout in order to overplot
	path_to_pHI = './data/measurements/'
	kkload, pHIload = np.loadtxt(path_to_pHI + "pHI_zout_0.0.txt", unpack=True)
	z_arr = np.array([0,0.5,1,1.5,2,3,5])
	pHI_matrix = np.zeros((z_arr.size, kkload.size))
	for iz, zi in enumerate(z_arr):
		kkload, pHI_matrix[iz, :] = np.loadtxt(path_to_pHI + "pHI_zout_%.1f.txt"%zi, unpack=True)
	phitrue_int = interpolate.interp1d(z_arr, pHI_matrix, axis=0)(zout)

	plt.figure(figsize=(8,5))
	plt.loglog(pHI.power.coords['k'], Plin_zout(pHI.power.coords['k']), 'k', label = '$P_{lin}$')
	plt.loglog(pHI.power.coords['k'], pHI.power['power'].real, label = 'HI mock')
	plt.loglog(kkload, phitrue_int,  '--', label = 'HI from TNG300-1 (for reference)')
	plt.title("HI at $z=%.1f$"%zout)
	plt.xlabel("$k\,[h\,\mathrm{Mpc}^{-1}]$", fontsize=12)
	plt.ylabel("$P\,[h^{-3}\mathrm{Mpc}^3]$")
	plt.xlim(right=1)
	plt.ylim(1e1,1e4)
	plt.legend(loc=0)
	if save_outputs:
        	plt.savefig(output_folder + "Pks_L_%.1f_Nmesh_%.1f_zout_%.1f_seed_%i.pdf"%(BoxSize,Nmesh,zout,seed))
	plt.close()
	print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

print ("Total time taken: %.1f sec."%(time.time()-start))

