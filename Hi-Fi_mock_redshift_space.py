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

axis = 2 # coordinate axis along which RSDs have been applied; matching z-axis, los=[0,0,1]
Nmufid = 6 # number of mu bins for nbodykit FFTPower, this is actually 2x the number of (positive) mu bins

print ("Generating HI mock in redshift-space at output redshift z=%.0f, in a BoxSize L=%.1f using nbar=%.2f (%i particles) on a Nmesh=%i^3 grid with IC seed %i..."\
		%(zout, BoxSize, nbar, int(nbar*BoxSize**3), Nmesh, seed))

# Cosmology and parameters
c = cosmology.Planck15
c = c.match(sigma8=0.8159)
Plin_zout = cosmology.LinearPower(c, zout)
Plin_z0 = cosmology.LinearPower(c, 0)
Dic  = c.scale_independent_growth_factor(zic)
Dout = c.scale_independent_growth_factor(zout)
fout = c.scale_independent_growth_rate(zout)

# Parameter paths/options for saving/plotting
params_path = "./data/z_space_bestfit_params/"
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
dlin = get_dlin(seed, Nmesh, BoxSize, Plin_z0)
dlin *= Dic
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Compute shifted fields
print ('Computing shifted fields... ')
dz, d1, d2, dG2, dG2par, d3 = generate_fields_rsd(dlin, c, nbar, zic, zout, fout)
p1 = FFTPower(d1, mode='2d', kmin=kmin, Nmu=Nmufid, poles=[0,2])
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Orthogonalize shifted fields
print ('Orthogonalizing shifted fields... ')
d2, dG2, d3 = orthogonalize_rsd(d1, d2, dG2, d3, Nmufid)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Generate 3D HI fields
print ('Generating polynomial field... ')
HI_field_poly = rsd_polynomial_field(dz, d1, d2, dG2, dG2par, d3, params_path, zout, p1, fout)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Add Perr(k,mu) noise:
print ('Adding noise... ')
HI_field_poly += noise_kmu(zout, Nmesh, BoxSize, axis, fout, params_path)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

print ('Saving HI field... ')
if save_outputs: ArrayMesh(HI_field_poly, BoxSize).save(output_folder+"rsd_HI_field_poly_TNG300_L_%.1f_nbar_%.1f_zout_%.1f_seed_%i"%(BoxSize,nbar,zout,seed))
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

################
### Plotting ###
################

if plot:
	print ('Plotting and computing power spectra ... ')
	# load
	HI_field_poly = BigFileMesh(output_folder+"rsd_HI_field_poly_TNG300_L_%.1f_nbar_%.1f_zout_%.1f_seed_%i"%(BoxSize,nbar,zout,seed), dataset='Field')

	# plot smoothed fields
	R_gaussian = 2 # Mpc/h

	imshow_kw = dict(interpolation='none', cmap='RdBu_r', vmin=-1, vmax=10, extent=(0,BoxSize,0,BoxSize), origin='lower')
	cax = plt.imshow(HI_field_poly.apply(Gaussian(R_gaussian)).paint()[0,:,:].T, **imshow_kw)
	plt.colorbar(cax)
	plt.title('$\\delta_\\mathrm{HI}^s$ from polynomial fits, $z=%.0f$'%zout)
	plt.xlabel("$y\,[h^{-1}\,\\mathrm{Mpc}]$")
	plt.ylabel("$z\,[h^{-1}\,\\mathrm{Mpc}]$")
	if save_outputs:
		plt.savefig(output_folder + "rsd_slices_L_%.1f_nbar_%.1f_zout_%.1f_seed_%i.pdf"%(BoxSize,nbar,zout,seed))
	plt.close()

	# compute Pks
	pk = FFTPower(HI_field_poly, mode='2d', kmin=kmin, Nmu=Nmufid, poles=[0,2])
	Nmu0 = int(pk.attrs['Nmu']/2)

	phitrue = FFTPower.load("./data/measurements/pHI_rsd_zout_%.1f"%zout)

	plt.figure(figsize=(8,5))
	plt.loglog(pk.poles['k'], Plin_zout(pk.poles['k']), 'k', label = '$P_{lin}$')
	for i in range(Nmu0):
	    plt.loglog(pk.poles['k'], pk.power['power'][:,Nmu0+i].real, 'C0-', alpha=(i+1)/Nmu0, label='$\\mu=%.2f$'%pk.power.coords['mu'][Nmu0+i])
	    plt.loglog(phitrue.poles['k'], phitrue.power['power'][:,Nmu0+i].real, color='gray', alpha=(i+1)/Nmu0)
	plt.plot([],[],'gray', label = 'HI true (TNG)')
	plt.xlabel("$k\,[h\,\mathrm{Mpc}^{-1}]$", fontsize=12)
	plt.ylabel("$P\,[h^{-3}\mathrm{Mpc}^3]$")
	plt.xlim(right=1)
	plt.ylim(1e1,2e4)
	plt.legend(loc=0)
	if save_outputs:
		plt.savefig(output_folder + "rsd_Pks_L_%.1f_nbar_%.1f_zout_%.1f_seed_%i.pdf"%(BoxSize,nbar,zout,seed))
	plt.close()
	print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

print ("Total time taken: %.1f sec."%(time.time()-start))

