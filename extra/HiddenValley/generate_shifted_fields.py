# import all that's needed
import sys
sys.path.insert(0,'/home/aobulj/data/Hi-Fi_mocks')
sys.path.insert(0,'../')
from lib.tng_lib import *
import numpy as np
from nbodykit.lab import *
import time
from argparse import ArgumentParser

comm = CurrentMPIComm.get()    
print ('comm', comm, 'comm.rank', comm.rank, 'comm.size', comm.size)
rank = comm.rank

parser = ArgumentParser()
parser.add_argument('--Nmesh', type=int, default=256, help='Mesh size per side (default 256)')
parser.add_argument('--zout', type=float, default=0.5, help='output redshift (default 0.5)')
args = parser.parse_args()
Nmesh = args.Nmesh
zout = args.zout

output_folder = '/shares/stadel.ics.mnf.uzh/aobulj/HiddenValley/shifted_fields/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

##########################
### General parameters ###
##########################

BoxSize = 1024
zic  = 99 # Abacus initial redshift
kmin = np.pi/BoxSize # kmin used in Pk measurements [h/Mpc]
seed = 9100 

# HV cosmology:
c = cosmology.Cosmology(h=0.6770, Omega0_b=0.049, Omega0_cdm=0.26014, n_s=0.96824, m_ncdm=[], A_s=2.10732e-9)

Plin_zout = cosmology.LinearPower(c, zout)
Plin_z0 = cosmology.LinearPower(c, 0)
Dic  = c.scale_independent_growth_factor(zic)
Dout = c.scale_independent_growth_factor(zout)

#################
### Main part ###
#################
start = time.time()

# Generate linear overdensity field at zic
print ('Generating initial density field... ')
dlin = get_dlin(seed, Nmesh, BoxSize, Plin_z0, comm)
dlin *= Dic
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

print ("Generating shifted fields in real space at output redshift z=%.1f, in a BoxSize L=%.1f on a Nmesh=%i^3 grid..."%(zout, BoxSize, Nmesh))

# Compute shifted fields
print ('Computing shifted fields...')
d1, d2, dG2, d3 = generate_fields_new(dlin, c, zic, zout, comm=comm)
p1 = FFTPower(d1, mode='1d', kmin=kmin)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# Orthogonalize shifted fields
print ('Orthogonalizing shifted fields...')
d2, dG2, d3 = orthogonalize(d1, d2, dG2, d3)
print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

# I'll save real-valued fields
d1 = d1.c2r()[:]
d2 = d2.c2r()[:]
dG2 = dG2.c2r()[:]
d3 = d3.c2r()[:]

np.save(output_folder + 'shifted_fields_real_N_%i_zout_%.1f'%(Nmesh, zout), [d1, d2, dG2, d3])

# print ("Generating shifted fields in redshift space at output redshift z=%.1f, in a BoxSize L=%.1f on a Nmesh=%i^3 grid, using Abacus %i cosmo..."%(zout, BoxSize, Nmesh, c_ind))

# for axis_rsd in range(3):
    
#     # Compute shifted fields
#     print ('Computing shifted fields... ')
#     dz, d1, d2, dG2, dG2par, d3 = generate_fields_rsd_new_growth(dlin, prefactor, fout, zic, zout, axis=axis_rsd, comm=comm)
#     p1 = FFTPower(d1, mode='2d', kmin=kmin, Nmu=Nmufid, poles=[0,2])
#     print ('done (elapsed time: %1.f sec.)'%(time.time()-start))
    
#     # Orthogonalize shifted fields
#     print ('Orthogonalizing shifted fields... ')
#     d2, dG2, d3 = orthogonalize_rsd(d1, d2, dG2, d3, Nmufid, axis=axis_rsd)
#     print ('done (elapsed time: %1.f sec.)'%(time.time()-start))

#     dz = dz.c2r()[:]
#     d1 = d1.c2r()[:]
#     d2 = d2.c2r()[:]
#     dG2 = dG2.c2r()[:]
#     dG2par = dG2par.c2r()[:]
#     d3 = d3.c2r()[:]

#     np.save(output_folder + 'shifted_fields_redshift_axisrsd_%i_N_%i_c_%i'%(axis_rsd, Nmesh, c_ind), [dz, d1, d2, dG2, dG2par, d3])
