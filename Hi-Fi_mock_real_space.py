# import all that's needed
from lib.tng_lib import *
import numpy as np
from nbodykit.lab import *
import time
start = time.time()

##########################
### General parameters ###
##########################

zic = 127 # TNG initial redshift
BoxSize=205. # TNG300-1 size [Mpc/h]
kmin=2*np.pi/BoxSize/2 # kmin used in Pk measurements [h/Mpc]

seed = 2695896 # IC random seed (2695896)

Nmesh=256
nbar = 1 # Npart/BoxSize^3 [(Mpc/h)^-3]
zout = 1 # output redshift

# Cosmology
c = cosmology.Planck15
c = c.match(sigma8=0.8159)
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
dlin = get_dlin(seed, Nmesh, BoxSize, Plin_z0)
dlin *= Dic

# Compute shifted fields
d1, d2, dG2, d3 = generate_fields(dlin, c, nbar, zic, zout)
p1 = FFTPower(d1, mode='1d', kmin=kmin)

# Orthogonalize shifted fields
d2, dG2, d3 = orthogonalize(d1, d2, dG2, d3)

# Generate 3D HI fields and save outputs
HI_field_poly = polynomial_field(d1, d2, dG2, d3, params_path, zout, p1)

# Add isotropic stochastic noise:
HI_field_poly += noise(zout, Nmesh, BoxSize)

if save_outputs: ArrayMesh(HI_field_poly, BoxSize).save(output_folder+"HI_field_poly_TNG300_L_%.1f_nbar_%.1f_zout_%.1f_seed_%i"%(BoxSize,nbar,zout,seed))

print ("time taken %.1f sec."%(time.time()-start))

################
### Plotting ###
################

if plot:
	# load
	HI_field_poly = BigFileMesh(output_folder+"HI_field_poly_TNG300_L_%.1f_nbar_%.1f_zout_%.1f_seed_%i"%(BoxSize,nbar,zout,seed), dataset='Field')

	# plot smoothed fields
	R_gaussian = 2 # Mpc/h

	imshow_kw = dict(interpolation='none', cmap='RdBu_r', vmin=-1, vmax=10, extent=(0,BoxSize,0,BoxSize), origin='lower')
	cax = plt.imshow(HI_field_poly.apply(Gaussian(R_gaussian)).paint()[0,:,:].T, **imshow_kw)
	plt.colorbar(cax)
	plt.title('$\\delta_\\mathrm{HI}^r$ from polynomial fits, $z=%.0f$'%zout)
	plt.xlabel("$y\,[h^{-1}\,\\mathrm{Mpc}]$")
	plt.ylabel("$z\,[h^{-1}\,\\mathrm{Mpc}]$")
	if save_outputs:
		plt.savefig(output_folder + "real_slices_L_%.1f_nbar_%.1f_zout_%.1f_seed_%i.pdf"%(BoxSize,nbar,zout,seed))
	plt.close()

	# compute Pks
	pk_field_poly = FFTPower(HI_field_poly, mode='1d', kmin=kmin)

	# load true pHI to overplot
	phitrue = FFTPower.load("./data/measurements/pHI_zout_%.1f"%zout)

	plt.figure(figsize=(8,5))
	plt.loglog(pk_field_poly.power['k'], Plin_zout(pk_field_poly.power['k']), 'k', label = '$P_{lin}$')
	plt.loglog(phitrue.power['k'], phitrue.power['power'].real, label = 'HI true')
	plt.loglog(pk_field_poly.power['k'], pk_field_poly.power['power'].real, '--', label = 'HI polyfit')
	plt.title("TNG300-1, $z=%.1f$"%zout)
	plt.xlabel("$k\,[h\,\mathrm{Mpc}^{-1}]$", fontsize=12)
	plt.ylabel("$P\,[h^{-3}\mathrm{Mpc}^3]$")
	plt.xlim(right=1)
	plt.ylim(1e1,1e4)
	plt.legend(loc=0)
	if save_outputs:
		plt.savefig(output_folder + "Pks_L_%.1f_nbar_%.1f_zout_%.1f_seed_%i.pdf"%(BoxSize,nbar,zout,seed))
	plt.close()