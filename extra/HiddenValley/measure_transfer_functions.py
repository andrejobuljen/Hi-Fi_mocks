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
from argparse import ArgumentParser

comm = CurrentMPIComm.get()    
print ('comm', comm, 'comm.rank', comm.rank, 'comm.size', comm.size)
rank = comm.rank

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--batch_num', type=int, required=True)
args = parser.parse_args()

batch_size = args.batch_size
batch_num = args.batch_num

Nmesh = 256
BoxSize = 1024
zout = 0.5
path = '/shares/stadel.ics.mnf.uzh/aobulj/HiddenValley/'
output_folder = path + 'results/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

zic  = 99 # Abacus initial redshift
kmin = np.pi/BoxSize # kmin used in Pk measurements [h/Mpc]

# HV cosmology:
c = cosmology.Cosmology(h=0.6770, Omega0_b=0.049, Omega0_cdm=0.26014, n_s=0.96824, m_ncdm=[], A_s=2.10732e-9)

Plin_zout = cosmology.LinearPower(c, zout)
Plin_z0 = cosmology.LinearPower(c, 0)
Dic  = c.scale_independent_growth_factor(zic)
Dout = c.scale_independent_growth_factor(zout)

d1, d2, dG2, d3 = np.load(path + 'shifted_fields/shifted_fields_real_N_%i_zout_%.1f.npy'%(Nmesh, zout))
d1  = ArrayMesh(d1, BoxSize=BoxSize).to_field(mode='complex')
d2  = ArrayMesh(d2, BoxSize=BoxSize).to_field(mode='complex')
dG2 = ArrayMesh(dG2, BoxSize=BoxSize).to_field(mode='complex')
d3  = ArrayMesh(d3, BoxSize=BoxSize).to_field(mode='complex')

start = batch_num * batch_size
end = start + batch_size

for i in range(start, end):
    
    path_to_save = output_folder + "results_HV_z=%.4f_Nmesh_%i.npy"%(i, zout, Nmesh)
    
    if not os.path.exists(path_to_save):

        # load delta_g
        print ('Loading HI catalog... ')
        gal_cat = np.load(path + 'AbacusSmall_c=%i_ph=3000_hod=%i_z=%.4f.npy'%(c_ind, i, zout))
        gal_cat.shape
        
        dtype = np.dtype([('Position', ('f8', 3)),])
        cat = np.empty((gal_cat.shape[0],), dtype=dtype)
        cat['Position']  = gal_cat[:,:3]
        
        cat = ArrayCatalog(cat, BoxSize=BoxSize * np.ones(3), Nmesh=Nmesh)
        cat = cat.to_mesh(compensated=True).paint() - 1.0
        print('cat.mean', cat.cmean())
        # nbar = gal_cat.shape[0]/BoxSize**3    
        # print('Nh = %.2f, 1/nbar = %.2f'%(gal_cat.shape[0], 1/nbar))
        
        delta_h = ArrayMesh(cat, BoxSize)
        print ('done (elapsed time: %1.f sec.)'%(time.time()-start))
        
        # Compute various Pk and cross-Pk
        ph_fin  = FFTPower(delta_h, mode='1d', kmin=kmin)
        kk = ph_fin.power.coords['k']
        
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
        
        beta11d1ort  =  d1.apply(lambda k, v: beta1inter(sum(ki ** 2 for ki in k)**0.5) * v)
        beta22d2ort  =  d2.apply(lambda k, v: beta2inter(sum(ki ** 2 for ki in k)**0.5) * v)
        betaG2dG2ort = dG2.apply(lambda k, v: betaG2inter(sum(ki** 2 for ki in k)**0.5) * v)
        beta33d3ort  =  d3.apply(lambda k, v: beta3inter(sum(ki ** 2 for ki in k)**0.5) * v)
        
        # Best-fit fields including different bias operators
        final_field = beta11d1ort + beta22d2ort + betaG2dG2ort + beta33d3ort
    
        diff = delta_h.to_field(mode='complex') - final_field
        
        # compute Pk of the residual
        perr = FFTPower(diff, mode='1d', kmin=kmin)
        phbestfit = FFTPower(delta_h, mode='1d', second=final_field, kmin=kmin)
        pbestfit = FFTPower(final_field, mode='1d', kmin=kmin)
        rhbestfit = phbestfit.power['power'].real/(ph_fin.power['power'].real*pbestfit.power['power'].real)**0.5
        
        results = kk, perr.power['power'].real, ph_fin.power['power'].real, phbestfit.power['power'].real, rhbestfit, beta1, beta2, betaG2, beta3, nbar
        np.save(path_to_save, results)


# # redshift space

# for axis_rsd in range(3):
    
#     dz, d1, d2, dG2, dG2par, d3 = np.load(path + 'shifted_fields/shifted_fields_redshift_axisrsd_%i_N_%i_c_%i.npy'%(axis_rsd, Nmesh, c_ind))
    
#     dz  = ArrayMesh(dz, BoxSize=BoxSize).to_field(mode='complex')
#     d1  = ArrayMesh(d1, BoxSize=BoxSize).to_field(mode='complex')
#     d2  = ArrayMesh(d2, BoxSize=BoxSize).to_field(mode='complex')
#     dG2 = ArrayMesh(dG2, BoxSize=BoxSize).to_field(mode='complex')
#     dG2par = ArrayMesh(dG2par, BoxSize=BoxSize).to_field(mode='complex')
#     d3  = ArrayMesh(d3, BoxSize=BoxSize).to_field(mode='complex')

#     los = np.zeros(3, dtype='int')
#     los[axis_rsd] = 1
#     print (los)
    
#     for i in range(start, end):

#         path_to_save = output_folder + "results_RSD_AbacusSmall_c=%i_ph=3000_hod=%i_z=%.4f_Nmesh_%i_axisrsd_%i.npy"%(c_ind, i, zout, Nmesh, axis_rsd)
#         if not os.path.exists(path_to_save):

#             # load delta_g
#             print ('Loading galaxy catalog... ')
#             gal_cat = np.load(path + 'AbacusSmall_c=%i_ph=3000_hod=%i_z=%.4f.npy'%(c_ind, i, zout))
                
#             dtype = np.dtype([('Position', ('f8', 3)),])
#             cat = np.empty((gal_cat.shape[0],), dtype=dtype)
#             if axis_rsd==2:
#                 cat['Position']  = gal_cat[:,[0,1,5]]
#             elif axis_rsd==1:
#                 cat['Position']  = gal_cat[:,[0,4,2]]
#             elif axis_rsd==0:
#                 cat['Position']  = gal_cat[:,[3,1,2]]
            
#             cat = ArrayCatalog(cat, BoxSize=BoxSize * np.ones(3), Nmesh=Nmesh)
#             cat = cat.to_mesh(compensated=True).paint() - 1.0
#             print('cat.mean', cat.cmean())
#             nbar = gal_cat.shape[0]/BoxSize**3    
#             print('Nh = %.2f, 1/nbar = %.2f'%(gal_cat.shape[0], 1/nbar))
            
#             delta_h = ArrayMesh(cat, BoxSize)
#             print ('done (elapsed time: %1.f sec.)'%(time.time()-start))
            
#             # Compute various Pk and cross-Pk
#             print ('computing transfer functions... ')
#             ph_fin = FFTPower(delta_h, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
#             pz = FFTPower(dz, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
#             kk = pz.power.coords['k']
#             Nmu0 = int(pz.attrs['Nmu']/2)
#             mus = pz.power.coords['mu'][Nmu0:]
#             pz.power = pz.power[:,Nmu0:]
            
#             pd1ort = FFTPower(d1, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
#             pd2ort = FFTPower(d2, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
#             pG2ort = FFTPower(dG2,mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
#             pd3ort = FFTPower(d3, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
            
#             rsd_minus_ck = (delta_h.to_field(mode='complex') - (dz-3./7.*fout*dG2par)).c2r()
            
#             ph_d1ort = FFTPower(rsd_minus_ck, mode='2d', second=d1, kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
#             ph_d2ort = FFTPower(rsd_minus_ck, mode='2d', second=d2, kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
#             ph_dG2ort= FFTPower(rsd_minus_ck, mode='2d', second=dG2,kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
#             ph_d3ort = FFTPower(rsd_minus_ck, mode='2d', second=d3, kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
            
#             # Transfer functions
#             beta1 = ph_d1ort.power['power'].real[:,Nmu0:]/pd1ort.power['power'].real[:,Nmu0:]
#             beta2 = ph_d2ort.power['power'].real[:,Nmu0:]/pd2ort.power['power'].real[:,Nmu0:]
#             betaG2= ph_dG2ort.power['power'].real[:,Nmu0:]/pG2ort.power['power'].real[:,Nmu0:]
#             beta3 = ph_d3ort.power['power'].real[:,Nmu0:]/pd3ort.power['power'].real[:,Nmu0:]
    
#             beta1_interkmu = interp1d_manual_k_binning(pz.power['k'], beta1, fill_value=[beta1[0][0],beta1[-1][0]], Ngrid=pz.attrs['Nmesh'], L = pz.attrs['BoxSize'][0], Pkref=ph_fin, kind='manual_Pk_k_mu_bins')
#             beta2_interkmu = interp1d_manual_k_binning(pz.power['k'], beta2, fill_value=[beta2[0][0],beta2[-1][0]], Ngrid=pz.attrs['Nmesh'], L = pz.attrs['BoxSize'][0], Pkref=ph_fin, kind='manual_Pk_k_mu_bins')
#             betaG2_interkmu = interp1d_manual_k_binning(pz.power['k'], betaG2, fill_value=[betaG2[0][0],betaG2[-1][0]], Ngrid=pz.attrs['Nmesh'], L = pz.attrs['BoxSize'][0], Pkref=ph_fin, kind='manual_Pk_k_mu_bins')
#             beta3_interkmu = interp1d_manual_k_binning(pz.power['k'], beta3, fill_value=[beta3[0][0],beta3[-1][0]], Ngrid=pz.attrs['Nmesh'], L = pz.attrs['BoxSize'][0], Pkref=ph_fin, kind='manual_Pk_k_mu_bins')
#             print ('done (elapsed time: %1.f sec.)'%(time.time()-start))
            
    
#             def rsd_filter_beta1(k3vec, val):
#                 absk = (sum(ki**2 for ki in k3vec))**0.5  # absk on the mesh
#                 # Dont use absk[absk==0]=1 b/c interp does not allow k=1.
#                 with np.errstate(invalid='ignore',
#                                  divide='ignore'):
#                     mu = sum(k3vec[i] * ph_fin.attrs['los'][i] for i in range(3)) / absk
#                 return beta1_interkmu(absk, mu) * val
            
#             def rsd_filter_beta2(k3vec, val):
#                 absk = (sum(ki**2 for ki in k3vec))**0.5  # absk on the mesh
#                 # Dont use absk[absk==0]=1 b/c interp does not allow k=1.
#                 with np.errstate(invalid='ignore',
#                                  divide='ignore'):
#                     mu = sum(k3vec[i] * ph_fin.attrs['los'][i] for i in range(3)) / absk
#                 return beta2_interkmu(absk, mu) * val
            
#             def rsd_filter_betaG2(k3vec, val):
#                 absk = (sum(ki**2 for ki in k3vec))**0.5  # absk on the mesh
#                 # Dont use absk[absk==0]=1 b/c interp does not allow k=1.
#                 with np.errstate(invalid='ignore',
#                                  divide='ignore'):
#                     mu = sum(k3vec[i] * ph_fin.attrs['los'][i] for i in range(3)) / absk
#                 return betaG2_interkmu(absk, mu) * val
            
#             def rsd_filter_beta3(k3vec, val):
#                 absk = (sum(ki**2 for ki in k3vec))**0.5  # absk on the mesh
#                 # Dont use absk[absk==0]=1 b/c interp does not allow k=1.
#                 with np.errstate(invalid='ignore',
#                                  divide='ignore'):
#                     mu = sum(k3vec[i] * ph_fin.attrs['los'][i] for i in range(3)) / absk
#                 return beta3_interkmu(absk, mu) * val
            
#             print ('making best-fit field... ')
#             beta1d1ort = d1.apply(rsd_filter_beta1, kind='wavenumber')
#             beta1d1ort[np.isnan(beta1d1ort)]=0+0j
            
#             beta2d2ort = d2.apply(rsd_filter_beta2, kind='wavenumber')
#             beta2d2ort[np.isnan(beta2d2ort)]=0+0j
            
#             betaG2dG2ort = dG2.apply(rsd_filter_betaG2, kind='wavenumber')
#             betaG2dG2ort[np.isnan(betaG2dG2ort)]=0+0j
            
#             beta3d3ort = d3.apply(rsd_filter_beta3, kind='wavenumber')
#             beta3d3ort[np.isnan(beta3d3ort)]=0+0j
            
#             # Best-fit fields including different bias operators
#             final_field = dz-3./7.*fout*dG2par + beta1d1ort + beta2d2ort + betaG2dG2ort + beta3d3ort
        
#             diff = (delta_h.to_field(mode='complex') - final_field).c2r()
             
#             # compute Pk of the residual
#             perr = FFTPower(diff, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
#             pbf = FFTPower(final_field, mode='2d', kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los) 
#             phbf = FFTPower(delta_h, mode='2d', second=final_field, kmin=kmin, poles=[0,2], Nmu=Nmufid, los=los)
#             rh_bf = phbf.power['power'][:,Nmu0:].real/(ph_fin.power['power'][:,Nmu0:].real*pbf.power['power'][:,Nmu0:].real)**0.5
            
#             # path_to_save = output_folder + "results_RSD_AbacusSmall_c=%i_ph=3000_hod=%i_z=%.4f_Nmesh_%i_axisrsd_%i.npy"%(c_ind, i, zout, Nmesh, axis_rsd)
#             results = kk, mus, perr.power['power'][:,Nmu0:].real, ph_fin.power['power'][:,Nmu0:].real, pbf.power['power'][:,Nmu0:].real, \
#                         rh_bf, beta1, beta2, betaG2, beta3, nbar
#             np.save(path_to_save, results)

