from __future__ import print_function, division
import numpy as np 
import matplotlib.pyplot as plt
from nbodykit.lab import ArrayMesh, FieldMesh
from nbodykit.filters import Gaussian
from nbodykit.source.catalog import UniformCatalog, ArrayCatalog
from nbodykit import mockmaker
from nbodykit.algorithms import FFTPower
from pmesh.pm import ParticleMesh

from nbodykit.utils import GatherArray

# from collections import OrderedDict
# import numpy.core.numeric as NX
import os, sys, time
from scipy import interpolate as interp

def generate_fields(delta_ic, cosmo, nbar, zic, zout, plot=True, weight=True, Rsmooth=0, seed=1234, Rdelta=0, posgrid='uniform'):
    scale_factor = 1/(1+zout)
    Nmesh = delta_ic.Nmesh
    BoxSize = delta_ic.BoxSize[0]
    prefactor = cosmo.scale_independent_growth_factor(zout)/cosmo.scale_independent_growth_factor(zic)

    # smooth delta_ic
    disp_f = [get_displacement_from_density_rfield(delta_ic, component=i, Psi_type='Zeldovich', smoothing={'R': Rsmooth,}) for i in range(3)]
    Nptcles_per_dim = int((nbar*BoxSize**3)**(1/3))

    delta_ic = ArrayMesh(delta_ic, BoxSize)
    weightsG2 = tidal_G2(delta_ic)
    delta_ic = delta_ic.apply(Gaussian(Rdelta)).paint()
    
    if posgrid=='reg':
        pos = reg_grid(Nptcles_per_dim=Nptcles_per_dim)
    elif posgrid=='uniform':
        pos = UniformCatalog(nbar, BoxSize=BoxSize, seed=seed)
    pos = pos['Position'].compute()

    weights1 = delta_ic.readout(pos, resampler='cic')*prefactor
    weights1 -= np.mean(weights1)
    weights2 = weights1**2
    weights2 -= np.mean(weights2)
    weightsG2 = weightsG2.readout(pos, resampler='cic')*prefactor**2

    delta_ic = ArrayMesh(delta_ic, BoxSize).apply(lambda k, v: v*((k.normp()**0.5) <= 0.5)).paint()
    weights3 = delta_ic.readout(pos, resampler='cic')*prefactor
    weights3 = weights3**3

    N = pos.shape[0]
    displacement = np.zeros((N, 3))
    for i in range(3):
        displacement[:, i] = disp_f[i].readout(pos, resampler='cic')
    displacement *= prefactor
    pos[:] = (pos + displacement) % BoxSize
    del displacement, disp_f
     
    dtype = np.dtype([('Position', ('f8', 3)), ('delta_1', 'f8'), ('delta_2', 'f8'), ('delta_G2', 'f8'), ('delta_3', 'f8')])
    catalog = np.empty((pos.shape[0],), dtype=dtype)
    catalog['Position'] = pos
    catalog['delta_1'] = weights1
    catalog['delta_2'] = weights2
    catalog['delta_G2'] = weightsG2
    catalog['delta_3'] = weights3
    catalog = ArrayCatalog(catalog, BoxSize=BoxSize * np.ones(3), Nmesh=Nmesh)
    del pos

    d1 = catalog.to_mesh(value='delta_1', compensated=True).to_real_field()      
    d2 = catalog.to_mesh(value='delta_2', compensated=True).to_real_field()    
    dG2 = catalog.to_mesh(value='delta_G2', compensated=True).to_real_field()     
    d3 = catalog.to_mesh(value='delta_3', compensated=True).to_real_field()
    return d1, d2, dG2, d3

def generate_fields_new(dlin, cosmo, zic, zout, comm=None, compensate=True):
    delta_ic = dlin.copy()
    scale_factor = 1/(1+zout)
    Nmesh = delta_ic.Nmesh
    BoxSize = delta_ic.BoxSize[0]
    prefactor = cosmo.scale_independent_growth_factor(zout)/cosmo.scale_independent_growth_factor(zic)
    
    pos = delta_ic.pm.generate_uniform_particle_grid(shift=0)
    N = pos.shape[0]
    catalog = np.empty(N, dtype=[('Position', ('f8', 3)), ('delta_1', 'f8'), ('delta_2', 'f8'), ('delta_G2', 'f8'), ('delta_3', 'f8')])
    catalog['Position'][:] = pos[:]
    layout = delta_ic.pm.decompose(catalog['Position']) 
    del pos
    
    delta_1 = delta_ic.c2r().readout(catalog['Position'], layout=layout, resampler='cic')*prefactor
    delta_1 -= np.mean(delta_1)
    catalog['delta_1'][:] = delta_1[:]
    del delta_1
    
    catalog['delta_2'][:] = catalog['delta_1']**2
    catalog['delta_2'][:] -= np.mean(catalog['delta_2'])
    
    delta_G2 = tidal_G2(FieldMesh(delta_ic)).readout(catalog['Position'], layout=layout, resampler='cic')*prefactor**2
    catalog['delta_G2'][:] = delta_G2[:]
    del delta_G2
    
    delta_3 = d3_smooth(FieldMesh(delta_ic)).readout(catalog['Position'], layout=layout, resampler='cic')**3*prefactor**3
    catalog['delta_3'][:] = delta_3[:]
    del delta_3

    def potential_transfer_function(k, v):
        k2 = k.normp(zeromode=1)
        return v / (k2)
    pot_k = delta_ic.apply(potential_transfer_function, out=Ellipsis)

    displ_catalog = np.empty(N, dtype=[('displ', ('f8',3))])
    
    for d in range(3):
        def force_transfer_function(k, v, d=d):
            return k[d] * 1j * v
        force_d = pot_k.apply(force_transfer_function).c2r(out=Ellipsis)
        displ_catalog['displ'][:, d] = force_d.readout(catalog['Position'], layout=layout, resampler='cic')*prefactor
    
    catalog['Position'][:] = (catalog['Position'][:] + displ_catalog['displ'][:]) % BoxSize
    del displ_catalog, force_d, pot_k
    
    catalog = ArrayCatalog(catalog, BoxSize=BoxSize * np.ones(3), Nmesh=Nmesh, comm=comm)
    
    d1 = catalog.to_mesh(value='delta_1', compensated=compensate).paint().r2c()
    d2 = catalog.to_mesh(value='delta_2', compensated=compensate).paint().r2c()
    dG2 = catalog.to_mesh(value='delta_G2', compensated=compensate).paint().r2c()
    d3 = catalog.to_mesh(value='delta_3', compensated=compensate).paint().r2c()
    
    return d1, d2, dG2, d3

def generate_fields_rsd(delta_ic, cosmo, nbar, zic, zout, fout, plot=True, weight=True, Rsmooth=0, seed=1234, Rdelta=0, posgrid='uniform'):
    scale_factor = 1/(1+zout)
    Nmesh = delta_ic.Nmesh
    BoxSize = delta_ic.BoxSize[0]
    prefactor = cosmo.scale_independent_growth_factor(zout)/cosmo.scale_independent_growth_factor(zic)

    # smooth delta_ic
    disp_f = [get_displacement_from_density_rfield(delta_ic, component=i, Psi_type='Zeldovich', smoothing={'R': Rsmooth,}) for i in range(3)]
    Nptcles_per_dim = int((nbar*BoxSize**3)**(1/3))

    delta_ic = ArrayMesh(delta_ic, BoxSize)
    weightsG2 = tidal_G2(delta_ic)
    weightsG2par = tidal_G2_par(ArrayMesh(weightsG2, BoxSize))
    delta_ic = delta_ic.apply(Gaussian(Rdelta)).paint()
    
    if posgrid=='reg':
        pos = reg_grid(Nptcles_per_dim=Nptcles_per_dim)
    elif posgrid=='uniform':
        pos = UniformCatalog(nbar,BoxSize=BoxSize, seed=seed)
    pos = pos['Position'].compute()
    
    weights1 = delta_ic.readout(pos, resampler='cic')*prefactor
    weights1 -= np.mean(weights1)
    weights2 = weights1**2
    weights2 -= np.mean(weights2)
    weightsG2 = weightsG2.readout(pos, resampler='cic')*prefactor**2
    weightsG2par = weightsG2par.readout(pos, resampler='cic')*prefactor**2
    
    delta_ic = ArrayMesh(delta_ic, BoxSize).apply(lambda k, v: v*((k.normp()**0.5) <= 0.5)).paint()
    weights3 = delta_ic.readout(pos, resampler='cic')*prefactor
    weights3 = weights3**3

    N = pos.shape[0]
    displacement = np.zeros((N, 3))
    for i in range(3):
        displacement[:, i] = disp_f[i].readout(pos, resampler='cic')
    displacement *= prefactor
    pos[:] = (pos + displacement*[1,1,(1+fout)]) % BoxSize
    del displacement, disp_f
           
    dtype = np.dtype([('Position', ('f8', 3)), ('delta_1', 'f8'), \
                      ('delta_2', 'f8'), ('delta_G2', 'f8'), ('delta_G2_par', 'f8'), ('delta_3', 'f8')])
    catalog = np.empty((pos.shape[0],), dtype=dtype)
    catalog['Position'] = pos
    catalog['delta_1'] = weights1
    catalog['delta_2'] = weights2
    catalog['delta_G2'] = weightsG2
    catalog['delta_G2_par'] = weightsG2par
    catalog['delta_3'] = weights3
    catalog = ArrayCatalog(catalog, BoxSize=BoxSize * np.ones(3), Nmesh=Nmesh)
    del pos, weights1, weights2, weights3, weightsG2, weightsG2par

    dz = catalog.to_mesh(compensated=True).to_real_field()
    dz -= dz.cmean()
    d1 = catalog.to_mesh(value='delta_1', compensated=True).to_real_field()
    d2 = catalog.to_mesh(value='delta_2', compensated=True).to_real_field()
    dG2 = catalog.to_mesh(value='delta_G2', compensated=True).to_real_field()
    dG2par = catalog.to_mesh(value='delta_G2_par', compensated=True).to_real_field()
    d3 = catalog.to_mesh(value='delta_3', compensated=True).to_real_field()

    return dz, d1, d2, dG2, dG2par, d3

def generate_fields_rsd_new(dlin, cosmo, zic, zout, comm=None, compensate=True):
    delta_ic = dlin.copy()
    scale_factor = 1/(1+zout)
    Nmesh = delta_ic.Nmesh
    BoxSize = delta_ic.BoxSize[0]
    prefactor = cosmo.scale_independent_growth_factor(zout)/cosmo.scale_independent_growth_factor(zic)
    fout = cosmo.scale_independent_growth_rate(zout)

    pos = delta_ic.pm.generate_uniform_particle_grid(shift=0)
    N = pos.shape[0]
    catalog = np.empty(N, dtype=[('Position', ('f8', 3)), ('delta_1', 'f8'), ('delta_2', 'f8'), ('delta_G2', 'f8'), ('delta_G2_par', 'f8'), ('delta_3', 'f8')])
    catalog['Position'][:] = pos[:]
    del pos
    
    layout = delta_ic.pm.decompose(catalog['Position']) 

    delta_1 = delta_ic.c2r().readout(catalog['Position'], layout=layout, resampler='cic')*prefactor
    delta_1 -= np.mean(delta_1)
    catalog['delta_1'][:] = delta_1[:]
    del delta_1
    
    catalog['delta_2'][:] = catalog['delta_1']**2
    catalog['delta_2'][:] -= np.mean(catalog['delta_2'])
    
    delta_G2 = tidal_G2(FieldMesh(delta_ic))#
    catalog['delta_G2'][:] = delta_G2.readout(catalog['Position'], layout=layout, resampler='cic')[:]*prefactor**2
    
    delta_G2_par = tidal_G2_par(FieldMesh(delta_G2)).readout(catalog['Position'], layout=layout, resampler='cic')*prefactor**2
    catalog['delta_G2_par'][:] = delta_G2_par[:]
    del delta_G2, delta_G2_par
    
    delta_3 = d3_smooth(FieldMesh(delta_ic)).readout(catalog['Position'], layout=layout, resampler='cic')**3*prefactor**3
    catalog['delta_3'][:] = delta_3[:]
    del delta_3
    
    def potential_transfer_function(k, v):
        k2 = k.normp(zeromode=1)
        return v / (k2)
    pot_k = delta_ic.apply(potential_transfer_function, out=Ellipsis)

    displ_catalog = np.empty(N, dtype=[('displ', ('f8',3))])
    
    for d in range(3):
        def force_transfer_function(k, v, d=d):
            return k[d] * 1j * v
        force_d = pot_k.apply(force_transfer_function).c2r(out=Ellipsis)
        displ_catalog['displ'][:, d] = force_d.readout(catalog['Position'], layout=layout, resampler='cic')*prefactor
    
    catalog['Position'][:] = (catalog['Position'][:] + displ_catalog['displ'][:]*[1,1,(1+fout)]) % BoxSize
    del displ_catalog, force_d, pot_k
    
    catalog = ArrayCatalog(catalog, BoxSize=BoxSize * np.ones(3), Nmesh=Nmesh, comm=comm)
    
    dz = catalog.to_mesh(compensated=compensate).paint()
    dz -= dz.cmean()
    dz = dz.r2c()
    
    d1 = catalog.to_mesh(value='delta_1', compensated=compensate).paint().r2c()
    d2 = catalog.to_mesh(value='delta_2', compensated=compensate).paint().r2c()
    dG2 = catalog.to_mesh(value='delta_G2', compensated=compensate).paint().r2c()
    dG2par = catalog.to_mesh(value='delta_G2_par', compensated=compensate).paint().r2c()
    d3 = catalog.to_mesh(value='delta_3', compensated=compensate).paint().r2c()
    return dz, d1, d2, dG2, dG2par, d3

# this routine is taken from here: https://github.com/mschmittfull/lsstools/
def get_displacement_from_density_rfield(in_density_rfield,
                                         in_density_cfield=None,
                                         component=None,
                                         Psi_type=None,
                                         smoothing=None,
                                         smoothing_Psi3LPT=None,
                                         prefac_Psi_1storder=1.0,
                                         prefac_Psi_2ndorder=1.0,
                                         prefac_Psi_3rdorder=1.0,
                                         RSD=False,
                                         RSD_line_of_sight=None,
                                         RSD_f_log_growth=None):
    """
    Given density delta(x) in real space, compute Zeldovich displacemnt Psi_component(x)
    given by Psi_component(\vk) = k_component / k^2 * W(k) * delta(\vk),
    where W(k) is smoothing window.
    For Psi_type='Zeldovich' compute 1st order displacement.
    For Psi_type='2LPT' compute 1st plus 2nd order displacement.
    etc
    Supply either in_density_rfield or in_density_cfield.
    Multiply 1st order displacement by prefac_Psi_1storder, 2nd order by 
    prefac_Psi_2ndorder, etc. Use this for getting time derivative of Psi.
    Follow http://rainwoodman.github.io/pmesh/intro.html.
    Parameters
    ----------
    RSD : boolean
        If True, include RSD by displacing by \vecPsi(q)+f (\e_LOS.\vecPsi(q)) \e_LOS, 
        where \ve_LOS is unit vector in line of sight direction.
    RSD_line_of_sight : array_like, (3,)
        Line of sight direction, e.g. [0,0,1] for z axis.
    """
    assert (component in [0, 1, 2])
    assert Psi_type in ['Zeldovich', '2LPT', '-2LPT']

    
    # comm = CurrentMPIComm.get()

    if in_density_cfield is None:
        # copy so we don't do any in-place changes by accident
        density_rfield = in_density_rfield.copy()
        density_cfield = density_rfield.r2c()
    else:
        assert in_density_rfield is None
        density_cfield = in_density_cfield.copy()


    if Psi_type in ['Zeldovich', '2LPT', '-2LPT']:

        # get zeldovich displacement in direction given by component

        def potential_transfer_function(k, v):
            k2 = sum(ki**2 for ki in k)
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.where(k2 == 0.0, 0 * v, v / (k2))

        # get potential pot = delta/k^2
        #pot_k = density_rfield.r2c().apply(potential_transfer_function)
        pot_k = density_cfield.apply(potential_transfer_function)
        #print("pot_k head:\n", pot_k[:2,:2,:2])

        # apply smoothing
        if smoothing is not None:
            pot_k = smoothen_cfield(pot_k, **smoothing)

            #print("pot_k head2:\n", pot_k[:2,:2,:2])

        # get zeldovich displacement
        def force_transfer_function(k, v, d=component):
            # MS: not sure if we want a factor of -1 here.
            return k[d] * 1j * v

        Psi_component_rfield = pot_k.apply(force_transfer_function).c2r()

        Psi_component_rfield *= prefac_Psi_1storder

    return Psi_component_rfield

# this routine was taken from here: 
# https://github.com/mschmittfull/lsstools/
def smoothen_cfield(in_pm_cfield, mode='Gaussian', R=0.0, kmax=None):

    pm_cfield = in_pm_cfield.copy()

    # zero pad all k>=kmax
    if kmax is not None:

        def kmax_fcn(k, v, kmax=kmax):
            k2 = sum(ki**2 for ki in k)
            return np.where(k2 < kmax**2, v, 0.0 * v)

        pm_cfield = pm_cfield.apply(kmax_fcn, out=Ellipsis)

    # apply smoothing
    if mode == 'Gaussian':
        if R != 0.0:

            def smoothing_fcn(k, v, R=R):
                k2 = sum(ki**2 for ki in k)
                W = np.exp(-0.5 * k2 * R**2)
                #print("smoothing: k:", k)
                #print("smoothing: :", W)
                return v * W

            pm_cfield = pm_cfield.apply(smoothing_fcn, out=Ellipsis)
    elif mode == '1-Gaussian':
        if R == 0.0:
            # W=1 so 1-W=0 and (1-W)delta=0
            pm_cfield = 0 * pm_cfield
        else:

            def OneMinusW_smoothing_fcn(k, v, R=R):
                k2 = sum(ki**2 for ki in k)
                W = np.exp(-0.5 * k2 * R**2)
                #print("smoothing: k:", k)
                #print("smoothing: :", W)
                return v * (1.0 - W)

            pm_cfield = pm_cfield.apply(OneMinusW_smoothing_fcn, out=Ellipsis)

    else:
        raise Exception("Invalid smoothing mode %s" % str(mode))

    return pm_cfield



def plot_fields(fields, vmin=None, vmax=None, titles=None):
    nfields = len(fields)
    fig, axes = plt.subplots(nrows=1, ncols=nfields, figsize=(5*nfields, 5))
    for i in range(nfields):
        cax = axes[i].imshow(fields[i].preview(axes=[0,1]), origin='lower', vmax=vmax, vmin=vmin)
        plt.colorbar(cax, ax=axes[i])
        if not np.any(titles== None):
            axes[i].set_title(titles[i])

def plotpk(delta_field, plot=True, label=None, second=None, ls='-'):
    pk = FFTPower(delta_field, mode='1d', second=second, kmin=2*np.pi/BoxSize/2)
    if plot: plt.loglog(pk.power['k'], pk.power['power'].real, label=label, ls=ls)
    return pk

def tidal_G2(delta):
# Compute -delta^2(\vx)
    out_rfield = - delta.compute(mode='real')**2

    # Compute d_ij(x). It's symmetric in i<->j so only compute j>=i.
    # d_ij = k_ik_j/k^2*basefield(\vk).
    for idir in range(3):
        for jdir in range(idir, 3):

            def my_transfer_function(k3vec, val, idir=idir, jdir=jdir):
                kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
                kk[kk == 0] = 1
                return k3vec[idir] * k3vec[jdir] / kk * val

            dij_k = delta.apply(my_transfer_function,
                                          mode='complex',
                                          kind='wavenumber')
            del my_transfer_function
            # do fft and convert field_mesh to RealField object
            dij_x = dij_k.compute(mode='real')
            # if verbose:
                # rfield_print_info(dij_x, comm, 'd_%d%d: ' % (idir, jdir))
            # Add \sum_{i,j=0..2} d_ij(\vx)d_ij(\vx)
            #   = [d_00^2+d_11^2+d_22^2 + 2*(d_01^2+d_02^2+d_12^2)]
            if jdir == idir:
                fac = 1.0
            else:
                fac = 2.0
            out_rfield += fac * dij_x**2
            del dij_x, dij_k
    return out_rfield

def tidal_G2_par(g2field):
    # for now only z
    def par_transfer_function(k3vec, val):
        kk = sum(ki**2 for ki in k3vec)  # k^2 on the mesh
        kk[kk == 0] = 1
        return k3vec[2]**2 / kk * val
    g2par = g2field.apply(par_transfer_function, mode='complex', kind='wavenumber')
    return g2par.compute(mode='real')

def d3_smooth(delta):
    def smooth(k, v):
        kk = (k.normp()**0.5)
        return v*(kk <= 0.5)
    dk = delta.apply(smooth, mode='complex', kind='wavenumber').compute(mode='real')
    return dk

def reg_grid(Nptcles_per_dim = 20):
    pmesh = ParticleMesh(BoxSize=BoxSize,Nmesh=[
                                     Nptcles_per_dim,
                                     Nptcles_per_dim,
                                     Nptcles_per_dim
                                 ])
    ptcles = pmesh.generate_uniform_particle_grid(shift=0.0, dtype='f8')
    dtype = np.dtype([('Position', ('f8', 3))])
    uni_cat_array = np.empty((ptcles.shape[0],), dtype=dtype)
    uni_cat_array['Position'] = ptcles
    uni_cat = ArrayCatalog(uni_cat_array,
                           comm=None,
                           BoxSize=BoxSize * np.ones(3),
                           Nmesh=[
                               Nptcles_per_dim,
                               Nptcles_per_dim,
                               Nptcles_per_dim
                           ])
    del ptcles, uni_cat_array
    return uni_cat



def get_dlin(seed, Nmesh, BoxSize, Pk, comm):
    pm = ParticleMesh([Nmesh,Nmesh,Nmesh], BoxSize, comm=comm)
    wn = pm.generate_whitenoise(seed)
    dlin = wn.apply(lambda k, v: Pk(sum(ki ** 2 for ki in k)**0.5) ** 0.5 * v / v.BoxSize.prod() ** 0.5)
    return dlin

# this routine is based on th: 
# https://github.com/mschmittfull/lsstools/
def interp1d_manual_k_binning(kin,
                              Pin,
                              kind='manual_Pk_k_bins',
                              fill_value=None,
                              bounds_error=False,
                              Ngrid=None,
                              L=None,
                              k_bin_width=1.0,
                              verbose=False,
                              Pkref=None):
    """
    Interpolate following a fixed k binning scheme that's also used to measure power spectra
    in cy_power_estimator.pyx.

    Parameters
    ----------
    kind : string
        Use 'manual_Pk_k_bins' for 1d power, or 'manual_Pk_k_mu_bins' for 2d power.

    L : float
        boxsize in Mpc/h

    kin, Pin: numpy.ndarray, (Nk*Nmu,)
        These are interpolated. Defined at k,mu bin central values.

    Pkref : MeasuredPower1D or MeasuredPower2D.
        This is used to get options of the measured power spectrum corresponding to
        Pin, e.g. Nk, Nmu, los, etc. (Note that Pin is ndarray so can't infer from that.)
        Does not use Pkref.power.k, Pkref.power.power etc.
    """
    # check args
    if (fill_value is None) and (not bounds_error):
        raise Exception("Must provide fill_value if bounds_error=False")
    if Ngrid is None:
        raise Exception("Must provide Ngrid")
    if L is None:
        raise Exception("Must provide L")

    if kind == 'manual_Pk_k_bins':

        check_Pk_is_1d(Pkref)

        dk = 2.0 * np.pi / float(L)

        # check that kin has all k bins
        if k_bin_width == 1.:
            # 18 Jan 2019: somehow need 0.99 factor for nbodykit 0.3 to get last k bin right.
            #kin_expected = np.arange(1,np.max(kin)*0.99/dk+1)*dk
            # 16 Mar 2019: Fix expected k bins to match nbodykit for larger Ngrid
            kin_expected = np.arange(1, kin.shape[0] + 1) * dk

            if verbose:
                print("kin:", kin)
                print("kin_expected:", kin_expected)
                print("kin/kin_expected (should be between 0.5 and 1.5):\n",
                      kin / kin_expected)

            # bin center is computed by averaging k within bin, so it's not exactly dk*i.
            if not np.allclose(kin, kin_expected, rtol=0.35):
                print("kin:", kin)
                print("kin_expected:", kin_expected)
                print("kin/kin_expected (should be between 0.5 and 1.5):\n",
                      kin / kin_expected)
                raise Exception('Found issue with k bins when interpolating')

        else:
            raise Exception("k_bin_width=%s not implemented yet" %
                            str(k_bin_width))

        def interpolator(karg):
            """
            Function that interpolates Pin from kin to karg.
            """
            ibin = round_float2int_arr(karg / (dk * k_bin_width))
            # first bin is dropped
            ibin -= 1

            # k's between kmin and max
            max_ibin = Pin.shape[0] - 1
            Pout = np.where((ibin >= 0) & (ibin <= max_ibin),
                            Pin[ibin % (max_ibin + 1)],
                            np.zeros(ibin.shape) + np.nan)

            # k<kmin
            if np.where(ibin < 0)[0].shape[0] > 0:
                if bounds_error:
                    raise Exception(
                        "Bounds error: k<kmin in interpolation, k=%s" %
                        str(karg))
                else:
                    Pout = np.where(ibin < 0,
                                    np.zeros(Pout.shape) + fill_value[0], Pout)

            # k>kmax
            if np.where(ibin > max_ibin)[0].shape[0] > 0:
                if bounds_error:
                    raise Exception(
                        "Bounds error: k>kmax in interpolation, k=%s" %
                        str(karg))
                else:
                    Pout = np.where(ibin > max_ibin,
                                    np.zeros(Pout.shape) + fill_value[1], Pout)

            if verbose:
                print("kin:\n", kin)
                print("Pin:\n", Pin)
                print("karg:\n", karg)
                print("Pout:\n", Pout)
            return Pout

        if verbose:
            print("Test manual_Pk_k_bins interpolator")
            print("Pin-interpolator(kin):\n", Pin - interpolator(kin))
            print("isclose:\n",
                  np.isclose(Pin,
                             interpolator(kin),
                             rtol=0.05,
                             atol=0.05 *
                             np.mean(Pin[np.where(~np.isnan(Pin))[0]]**2)**0.5,
                             equal_nan=True))
        if False:
            # ok on 64^3 but sometimes crashes 512^3 runs b/c of nan differences at high k
            assert np.allclose(
                Pin,
                interpolator(kin),
                rtol=0.05,
                atol=0.05 * np.mean(Pin[np.where(~np.isnan(Pin))[0]]**2)**0.5,
                equal_nan=True)
        if verbose:
            print("OK")
            print("test interpolator:", interpolator(kin))

    elif kind == 'manual_Pk_k_mu_bins':

        check_Pk_is_2d(Pkref)

        # get los and other attrs
        los0 = Pkref.power.attrs['los']
        Nmu0 = int(Pkref.attrs['Nmu']/2)
        Nk0 = Pkref.power['k'].shape[0]

        edges = Pkref.power.edges
        # print('edges:', edges)

        # setup edges
        # see project_to_basis in https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/algorithms/fftpower.html#FFTPower
        kedges = edges['k']
        muedges = edges['mu'][Nmu0:]
        Nk = len(kedges) - 1
        Nmu = len(muedges) - 1

        assert Nk == Nk0
        assert Nmu == Nmu0
        
        
        # new nbodykit uses mu's [-1,1]...
        kin = kin[:,:].flatten()
        Pin = Pin[:,:].flatten()
        
        # For indexing to be correct, first mu bin has to start at 0.
        assert muedges[0] == 0.0
        assert muedges[-1] == 1.0
        assert kedges[0] > 0.0
        assert kedges[0] < 2.0 * np.pi / L  # will drop first bin b/c of this

        assert Pkref.power['k'][:,Nmu0:].flatten().shape == (Nk * Nmu,)

        # Check kin and Pin have right shape and indexing
        assert kin.flatten().shape == (Nk * Nmu0,)
        assert Pin.flatten().shape == (Nk * Nmu0,)
        ww = np.where(~np.isnan(kin))
        assert np.allclose(kin[ww], Pkref.power['k'][:,Nmu0:].flatten()[ww])

        def interpolator(karg, muarg):
            """
            Function that interpolates Pin(kin) to karg, muarg.
            Use same binning as what is used to get P(k,mu) in 2d FFTPower code.

            Parameters
            ----------
            karg : np.ndarray, (N,)
            muarg : np.ndarray, (N,)
            """
            k_indices = np.digitize(karg, kedges)
            mu_indices = np.digitize(np.abs(muarg), muedges)
            # print ('muedges', muedges, 'muindices', mu_indices)
            # print ('kedges', kedges, 'kindices', k_indices)


            # nbodykit uses power[1:-1] at the end to drop stuff <edges[0]
            # and >=edges[-1]. Similarly, digitize returns 0 if mu<edges[0] (never occurs)
            # and Nmu if mu>=edges[-1]. Subtract one so we get we get mu_indices=0..Nmu,
            # and assign mu=1 to mu_index=Nmu-1
            mu_indices -= 1
            # When mu==1, assign to last bin, so it is right-inclusive.
            # print ('is close', np.isclose(np.abs(muarg), 1.0))
            mu_indices[np.isclose(np.abs(muarg), 1.0)] = Nmu - 1
            # mu_indices[mu_indices>Nmu-1] = Nmu-1

            # Same applies to k:
            k_indices -= 1

            # mu>=mumin=0
            assert np.all(mu_indices[~np.isnan(muarg)] >= 0)
            # mu<=mumax=1
            if not np.all(mu_indices[~np.isnan(muarg)] < Nmu):
                print("Found mu>1: ", muarg[mu_indices > Nmu - 1])
                raise Exception('Too large mu')

            # take lowest k bin when karg=0
            #k_indices[karg==0] = 0

            ##print('k_indices:', k_indices)
            #print('mu_indices:', mu_indices)

            #print('edges:', edges)
            #raise Exception('tmp')

            # Want to get Pin at indices k_indices, mu_indices.
            # Problem: Pin is (Nk*Nmu,) array so need to convert 2d to 1d index.
            # Use numpy ravel
            #multi_index = np.ravel_multi_index([k_indices, mu_indices], (Nk,Nmu))
            # Do manually (same result as ravel when 0<=k_indices<=Nk-1 and 0<=mu_indices<=Nmu-1.)
            # Also take modulo max_multi_index to avoid errror when k_indices or mu_indices out of bounds,
            # will handle those cases explicitly later.
            max_multi_index = (Nk - 1) * Nmu + (Nmu - 1)
            multi_index = (k_indices * Nmu + mu_indices) % (max_multi_index + 1)

            Pout = Pin.flatten()[multi_index]

            # Handle out of bounds cases

            # k>kmax
            if not np.all(k_indices < Nk):
                if bounds_error:
                    print('too large k: ', karg[k_indices >= Nk])
                    raise Exception(
                        "Bounds error: k>kmax in interpolation, k=%s" %
                        str(karg))
                else:
                    Pout = np.where(k_indices < Nk, Pout, np.zeros(Pout.shape) + fill_value[1])

            # k<kmin
            if not np.all(k_indices >= 0):
                if bounds_error:
                    print('too small k: ', karg[k_indices < 0])
                    raise Exception(
                        "Bounds error: k<kmin in interpolation, k=%s" %
                        str(karg))
                else:
                    Pout = np.where(k_indices >= 0, Pout, np.zeros(Pout.shape) + fill_value[0])

            # handle nan input
            Pout = np.where(np.isnan(karg), np.zeros(Pout.shape) + np.nan, Pout)
            Pout = np.where(np.isnan(muarg),np.zeros(Pout.shape) + np.nan, Pout)
            return Pout

    else:
        raise Exception("invalid kind %s" % str(kind))

    return interpolator

# this routine is based on a routine from here: https://github.com/mschmittfull/lsstools/
def check_Pk_is_1d(Pkref):
    # check Pkref is 2d
#     assert type(Pkref) == MeasuredPower1D
    assert Pkref.power.attrs['mode'] == '1d'
    assert Pkref.power.shape == (Pkref.power['k'].shape[0],)

# this routine is based on a routine from here: https://github.com/mschmittfull/lsstools/
def check_Pk_is_2d(Pkref):
    # check Pkref is 2d
#     assert type(Pkref) == MeasuredPower2D
    assert Pkref.power.attrs['mode'] == '2d'
    assert Pkref.power.shape == (Pkref.power['k'].shape[0], Pkref.attrs['Nmu'])

# this routine is based on a routine from here: https://github.com/mschmittfull/lsstools/
def round_float2int_arr(x):
    """round float to nearest int"""
    return np.where(x >= 0.0, (x + 0.5).astype('int'), (x - 0.5).astype('int'))

def orthogonalize(d1, d2, dG2, d3):
    
    kmin = 2*np.pi/d1.BoxSize[0]/2
    
    p1  = FFTPower(d1, mode='1d', kmin=kmin)
    p2  = FFTPower(d2, mode='1d', kmin=kmin)
    pG2 = FFTPower(dG2, mode='1d', kmin=kmin)
    p3  = FFTPower(d3, mode='1d', kmin=kmin)

    p12 = FFTPower(d1, mode='1d', second=d2, kmin=kmin)
    p1G2 = FFTPower(d1, mode='1d', second=dG2, kmin=kmin)
    p13 = FFTPower(d1, mode='1d', second=d3, kmin=kmin)

    p2G2 = FFTPower(d2, mode='1d', second=dG2, kmin=kmin)
    p23 = FFTPower(d2, mode='1d', second=d3, kmin=kmin)

    pG23 = FFTPower(dG2, mode='1d', second=d3, kmin=kmin)
    
    C = np.zeros((p1.power['k'].size,4,4)) + np.nan

    C[:,0,0] = 1.
    C[:,1,1] = 1.
    C[:,2,2] = 1.
    C[:,3,3] = 1.

    C[:,0,1] = p12.power['power'].real /(p1.power['power'].real*p2.power['power'].real)**0.5
    C[:,0,2] = p1G2.power['power'].real/(p1.power['power'].real*pG2.power['power'].real)**0.5
    C[:,0,3] = p13.power['power'].real/(p1.power['power'].real*p3.power['power'].real)**0.5

    C[:,1,2] = p2G2.power['power'].real/(p2.power['power'].real*pG2.power['power'].real)**0.5
    C[:,1,3] = p23.power['power'].real/(p2.power['power'].real*p3.power['power'].real)**0.5

    C[:,2,3] = pG23.power['power'].real/(pG2.power['power'].real*p3.power['power'].real)**0.5

    C[:,1,0] = C[:,0,1]
    C[:,2,0] = C[:,0,2]
    C[:,3,0] = C[:,0,3]

    C[:,2,1] = C[:,1,2]
    C[:,3,1] = C[:,1,3]

    C[:,3,2] = C[:,2,3]

    L = np.linalg.cholesky(C)
    Linv = np.linalg.inv(L)

    ratio10 = np.sqrt(p2.power['power'].real/p1.power['power'].real)
    ratio20 = np.sqrt(pG2.power['power'].real/p1.power['power'].real)
    ratio30 = np.sqrt(p3.power['power'].real/p1.power['power'].real)
    ratio21 = np.sqrt(pG2.power['power'].real/p2.power['power'].real)
    ratio31 = np.sqrt(p3.power['power'].real/p2.power['power'].real)
    ratio32 = np.sqrt(p3.power['power'].real/pG2.power['power'].real)

    M10 = Linv[:,1,0]/Linv[:,1,1]*ratio10
    M20 = Linv[:,2,0]/Linv[:,2,2]*ratio20
    M30 = Linv[:,3,0]/Linv[:,3,3]*ratio30
    M21 = Linv[:,2,1]/Linv[:,2,2]*ratio21
    M31 = Linv[:,3,1]/Linv[:,3,3]*ratio31
    M32 = Linv[:,3,2]/Linv[:,3,3]*ratio32
    
    kk = p1.power.coords['k']
    
    interkmu_M10 = interp1d_manual_k_binning(kk, M10, fill_value=[M10[0],M10[-1]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    interkmu_M20 = interp1d_manual_k_binning(kk, M20, fill_value=[M20[0],M20[-1]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    interkmu_M30 = interp1d_manual_k_binning(kk, M30, fill_value=[M30[0],M30[-1]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    interkmu_M21 = interp1d_manual_k_binning(kk, M21, fill_value=[M21[0],M21[-1]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    interkmu_M31 = interp1d_manual_k_binning(kk, M31, fill_value=[M31[0],M31[-1]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    interkmu_M32 = interp1d_manual_k_binning(kk, M32, fill_value=[M32[0],M32[-1]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    
    test = d1.apply(lambda k, v: interkmu_M10(sum(ki ** 2 for ki in k)**0.5) * v)
    d2ort = d2+test

    test = d1.apply(lambda k, v: interkmu_M20(sum(ki ** 2 for ki in k)**0.5) * v)
    test2 = d2.apply(lambda k, v: interkmu_M21(sum(ki ** 2 for ki in k)**0.5) * v)
    dG2ort = dG2+test+test2

    test = d1.apply(lambda k, v: interkmu_M30(sum(ki ** 2 for ki in k)**0.5) * v)
    test2 = d2.apply(lambda k, v: interkmu_M31(sum(ki ** 2 for ki in k)**0.5) * v)
    testG2 = dG2.apply(lambda k, v: interkmu_M32(sum(ki ** 2 for ki in k)**0.5) * v)

    d3ort = d3+testG2+test+test2
    del test, test2, testG2
    
    return d2ort, dG2ort, d3ort

def orthogonalize_rsd(d1, d2, dG2, d3, Nmu):
    
    kmin = 2*np.pi/d1.BoxSize[0]/2

    p1_ref = FFTPower(d1, mode='2d', Nmu=Nmu, kmin=kmin, poles=[0,2])
    kk = p1_ref.poles['k']

    p1 = FFTPower(d1, mode='2d', Nmu=Nmu, kmin=kmin)
    p1 = FFTPower(d1, mode='2d', Nmu=Nmu, kmin=kmin)
    p2 = FFTPower(d2, mode='2d', Nmu=Nmu, kmin=kmin)
    pG2 = FFTPower(dG2, mode='2d', Nmu=Nmu, kmin=kmin)
    p3 = FFTPower(d3, mode='2d', Nmu=Nmu, kmin=kmin)

    p12 = FFTPower(d1, second=d2, mode='2d', Nmu=Nmu, kmin=kmin)
    p1G2 = FFTPower(d1, second=dG2, mode='2d', Nmu=Nmu, kmin=kmin)
    p13 = FFTPower(d1, second=d3, mode='2d', Nmu=Nmu, kmin=kmin)

    p2G2 = FFTPower(d2, second=dG2, mode='2d', Nmu=Nmu, kmin=kmin)
    p23 = FFTPower(d2, second=d3, mode='2d', Nmu=Nmu, kmin=kmin)

    pG23 = FFTPower(dG2, second=d3, mode='2d', Nmu=Nmu, kmin=kmin)

    Nmu0 = int(p1.attrs['Nmu']/2) 

    p1.power = p1.power[:,Nmu0:]
    p2.power = p2.power[:,Nmu0:]
    pG2.power = pG2.power[:,Nmu0:]
    p3.power = p3.power[:,Nmu0:]

    p12.power = p12.power[:,Nmu0:]
    p1G2.power = p1G2.power[:,Nmu0:]
    p13.power = p13.power[:,Nmu0:]

    p2G2.power = p2G2.power[:,Nmu0:]
    p23.power = p23.power[:,Nmu0:]

    pG23.power = pG23.power[:,Nmu0:]

    C = np.zeros((p1.power['power'].shape[0],p1.power['power'].shape[1],4,4)) + np.nan

    C[...,0,0] = 1.
    C[...,1,1] = 1.
    C[...,2,2] = 1.
    C[...,3,3] = 1.

    C[...,0,1] = p12.power['power'].real /(p1.power['power'].real*p2.power['power'].real)**0.5
    C[...,0,2] = p1G2.power['power'].real/(p1.power['power'].real*pG2.power['power'].real)**0.5
    C[...,0,3] = p13.power['power'].real/(p1.power['power'].real*p3.power['power'].real)**0.5

    C[...,1,2] = p2G2.power['power'].real/(p2.power['power'].real*pG2.power['power'].real)**0.5
    C[...,1,3] = p23.power['power'].real/(p2.power['power'].real*p3.power['power'].real)**0.5
    C[...,2,3] = pG23.power['power'].real/(pG2.power['power'].real*p3.power['power'].real)**0.5

    C[...,1,0] = C[...,0,1]
    C[...,2,0] = C[...,0,2]
    C[...,3,0] = C[...,0,3]

    C[...,2,1] = C[...,1,2]
    C[...,3,1] = C[...,1,3]
    C[...,3,2] = C[...,2,3]


    C = np.where(np.isnan(C), 0, C)
    L = np.linalg.cholesky(C)
    Linv = np.linalg.inv(L)

    ratio10 = np.sqrt( p2.power['power'].real/p1.power['power'].real)
    ratio20 = np.sqrt(pG2.power['power'].real/p1.power['power'].real)
    ratio30 = np.sqrt(p3.power['power'].real/p1.power['power'].real)
    ratio21 = np.sqrt(pG2.power['power'].real/p2.power['power'].real)
    ratio31 = np.sqrt(p3.power['power'].real/p2.power['power'].real)
    ratio32 = np.sqrt(p3.power['power'].real/pG2.power['power'].real)

    ratio10 = np.where(np.isnan(ratio10), 0, ratio10)
    ratio20 = np.where(np.isnan(ratio20), 0, ratio20)
    ratio30 = np.where(np.isnan(ratio30), 0, ratio30)
    ratio21 = np.where(np.isnan(ratio21), 0, ratio21)
    ratio31 = np.where(np.isnan(ratio31), 0, ratio31)
    ratio32 = np.where(np.isnan(ratio32), 0, ratio32)

    M10 = Linv[...,1,0]/Linv[...,1,1]*ratio10
    M20 = Linv[...,2,0]/Linv[...,2,2]*ratio20
    M30 = Linv[...,3,0]/Linv[...,3,3]*ratio30
    M21 = Linv[...,2,1]/Linv[...,2,2]*ratio21
    M31 = Linv[...,3,1]/Linv[...,3,3]*ratio31
    M32 = Linv[...,3,2]/Linv[...,3,3]*ratio32

    interkmu_M10 = interp1d_manual_k_binning(p1.power['k'], M10, fill_value=[M10[0][0],M10[-1][0]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1_ref, kind='manual_Pk_k_mu_bins')
    interkmu_M20 = interp1d_manual_k_binning(p1.power['k'], M20, fill_value=[M20[0][0],M20[-1][0]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1_ref, kind='manual_Pk_k_mu_bins')
    interkmu_M30 = interp1d_manual_k_binning(p1.power['k'], M30, fill_value=[M30[0][0],M30[-1][0]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1_ref, kind='manual_Pk_k_mu_bins')
    interkmu_M21 = interp1d_manual_k_binning(p1.power['k'], M21, fill_value=[M21[0][0],M21[-1][0]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1_ref, kind='manual_Pk_k_mu_bins')
    interkmu_M31 = interp1d_manual_k_binning(p1.power['k'], M31, fill_value=[M31[0][0],M31[-1][0]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1_ref, kind='manual_Pk_k_mu_bins')
    interkmu_M32 = interp1d_manual_k_binning(p1.power['k'], M32, fill_value=[M32[0][0],M32[-1][0]], Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1_ref, kind='manual_Pk_k_mu_bins')

    def rsd_filter_M10(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5  # absk on the mesh
        # Dont use absk[absk==0]=1 b/c interp does not allow k=1.
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return interkmu_M10(absk, mu) * val

    def rsd_filter_M20(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return interkmu_M20(absk, mu) * val

    def rsd_filter_M21(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return interkmu_M21(absk, mu) * val

    def rsd_filter_M30(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return interkmu_M30(absk, mu) * val

    def rsd_filter_M31(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return interkmu_M31(absk, mu) * val

    def rsd_filter_M32(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return interkmu_M32(absk, mu) * val

    M10d1 = d1.apply(rsd_filter_M10, kind='wavenumber')
    M10d1[np.isnan(M10d1)]=0+0j

    M20d1 = d1.apply(rsd_filter_M20, kind='wavenumber')
    M20d1[np.isnan(M20d1)]=0+0j

    M21d2 = d2.apply(rsd_filter_M21, kind='wavenumber')
    M21d2[np.isnan(M21d2)]=0+0j

    M30d1 = d1.apply(rsd_filter_M30, kind='wavenumber')
    M30d1[np.isnan(M30d1)]=0+0j

    M31d2 = d2.apply(rsd_filter_M31, kind='wavenumber')
    M31d2[np.isnan(M31d2)]=0+0j

    M32dG2 = dG2.apply(rsd_filter_M32, kind='wavenumber')
    M32dG2[np.isnan(M32dG2)]=0+0j

    d2ort  = d2  + M10d1
    dG2ort = dG2 + M21d2 + M20d1
    d3ort  = d3  + M30d1 + M31d2 + M32dG2

    return d2ort, dG2ort, d3ort

def polynomial_field(d1, d2ort, dG2ort, d3ort, path, zout, p1):
    kk = p1.power['k']

    b1_params = np.loadtxt(path + 'b1_poly_zout_%.1f.txt'%zout, unpack=True)
    b2_params = np.loadtxt(path + 'b2_poly_zout_%.1f.txt'%zout, unpack=True)
    bG2_params = np.loadtxt(path + 'bG2_poly_zout_%.1f.txt'%zout, unpack=True)
    b3_params = np.loadtxt(path + 'b3_poly_zout_%.1f.txt'%zout, unpack=True)

    b1_poly = np.dot(np.array([kk*0+1, kk, kk**2, kk**4]).T, b1_params)
    b2_poly = np.dot(np.array([kk*0+1, kk**2, kk**4]).T, b2_params)
    bG2_poly= np.dot(np.array([kk*0+1, kk**2, kk**4]).T, bG2_params)
    b3_poly = np.dot(np.array([kk*0+1, kk**2, kk**4]).T, b3_params)

    b1_polyinter = interp1d_manual_k_binning(kk, b1_poly, fill_value=[b1_poly[0], b1_poly[-1]], \
                                             Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    poly_field   =  d1.r2c().apply(lambda k, v: b1_polyinter( sum(ki ** 2 for ki in k)**0.5) * v)
    
    b2_polyinter = interp1d_manual_k_binning(kk, b2_poly, fill_value=[b2_poly[0], b2_poly[-1]], \
                                             Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    poly_field +=  d2ort.r2c().apply(lambda k, v: b2_polyinter( sum(ki ** 2 for ki in k)**0.5) * v)
    
    bG2_polyinter = interp1d_manual_k_binning(kk, bG2_poly, fill_value=[bG2_poly[0], bG2_poly[-1]], \
                                             Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    poly_field  +=  dG2ort.r2c().apply(lambda k, v: bG2_polyinter( sum(ki ** 2 for ki in k)**0.5) * v)
    
    b3_polyinter = interp1d_manual_k_binning(kk, b3_poly, fill_value=[b3_poly[0], b3_poly[-1]], \
                                             Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    poly_field  +=  d3ort.r2c().apply(lambda k, v: b3_polyinter( sum(ki ** 2 for ki in k)**0.5) * v)
    
    return poly_field

def polynomial_field_zout(d1, d2ort, dG2ort, d3ort, path, zout, p1):
    kk = p1.power.coords['k']
#     kk = np.logspace(-3, 0, 1000)
    
    # available redshifts
    z_arr = np.array([0,0.5,1,1.5,2,3,5])
    b1_poly_z = np.zeros((z_arr.size, kk.size))
    b2_poly_z = np.zeros((z_arr.size, kk.size))
    bG2_poly_z = np.zeros((z_arr.size, kk.size))
    b3_poly_z = np.zeros((z_arr.size, kk.size))
    
    assert (zout>=0) and (zout<=5)

    for iz, zi in enumerate(z_arr):
        b1_params = np.loadtxt(path + 'b1_poly_zout_%.1f.txt'%zi, unpack=True)
        b2_params = np.loadtxt(path + 'b2_poly_zout_%.1f.txt'%zi, unpack=True)
        bG2_params = np.loadtxt(path + 'bG2_poly_zout_%.1f.txt'%zi, unpack=True)
        b3_params = np.loadtxt(path + 'b3_poly_zout_%.1f.txt'%zi, unpack=True)
        
        b1_poly_z[iz,:] = np.dot(np.array([kk*0+1, kk, kk**2, kk**4]).T, b1_params)
        b2_poly_z[iz,:] = np.dot(np.array([kk*0+1, kk**2, kk**4]).T, b2_params)
        bG2_poly_z[iz,:] = np.dot(np.array([kk*0+1, kk**2, kk**4]).T, bG2_params)
        b3_poly_z[iz,:] = np.dot(np.array([kk*0+1, kk**2, kk**4]).T, b3_params)

    # interpolate along redshifts and take the value at zout       
    b1_poly_zout = interp.interp1d(z_arr, b1_poly_z, axis=0)(zout)
    b2_poly_zout = interp.interp1d(z_arr, b2_poly_z, axis=0)(zout)
    bG2_poly_zout = interp.interp1d(z_arr, bG2_poly_z, axis=0)(zout)
    b3_poly_zout = interp.interp1d(z_arr, b3_poly_z, axis=0)(zout)
    
    # now make a function that interpolates at any k
    b1_poly = interp.interp1d(kk, b1_poly_zout, bounds_error=False, fill_value=(b1_poly_zout[0],b1_poly_zout[-1]))
    b2_poly = interp.interp1d(kk, b2_poly_zout, bounds_error=False, fill_value=(b2_poly_zout[0],b2_poly_zout[-1]))
    bG2_poly = interp.interp1d(kk, bG2_poly_zout, bounds_error=False, fill_value=(bG2_poly_zout[0],bG2_poly_zout[-1]))
    b3_poly = interp.interp1d(kk, b3_poly_zout, bounds_error=False, fill_value=(b3_poly_zout[0],b3_poly_zout[-1]))
    
    # b1_polyinter = interp1d_manual_k_binning(kk, b1_poly, fill_value=[b1_poly[0], b1_poly[-1]], \
                                             # Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    poly_field   =  d1.apply(lambda k, v: b1_poly( sum(ki ** 2 for ki in k)**0.5) * v)
    
    # b2_polyinter = interp1d_manual_k_binning(kk, b2_poly, fill_value=[b2_poly[0], b2_poly[-1]], \
                                             # Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    poly_field +=  d2ort.apply(lambda k, v: b2_poly( sum(ki ** 2 for ki in k)**0.5) * v)
    
    # bG2_polyinter = interp1d_manual_k_binning(kk, bG2_poly, fill_value=[bG2_poly[0], bG2_poly[-1]], \
                                             # Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    poly_field  +=  dG2ort.apply(lambda k, v: bG2_poly( sum(ki ** 2 for ki in k)**0.5) * v)
    
    # b3_polyinter = interp1d_manual_k_binning(kk, b3_poly, fill_value=[b3_poly[0], b3_poly[-1]], \
                                             # Ngrid=p1.attrs['Nmesh'], L = p1.attrs['BoxSize'][0], Pkref=p1)
    poly_field  +=  d3ort.apply(lambda k, v: b3_poly( sum(ki ** 2 for ki in k)**0.5) * v)
    
    return poly_field

def rsd_polynomial_field(dz, d1, d2ort, dG2ort, dG2par, d3ort, path, zout, p1, fout):

    b1_params = np.loadtxt(path + 'rsd_b1_poly_zout_%.1f.txt'%zout, unpack=True)
    b2_params = np.loadtxt(path + 'rsd_b2_poly_zout_%.1f.txt'%zout, unpack=True)
    bG2_params = np.loadtxt(path + 'rsd_bG2_poly_zout_%.1f.txt'%zout, unpack=True)
    b3_params = np.loadtxt(path + 'rsd_b3_poly_zout_%.1f.txt'%zout, unpack=True)

    def rsd_filter_beta1_poly(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return beta1_poly_interkmu(absk, mu) * val

    def rsd_filter_beta2_poly(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return beta2_poly_interkmu(absk, mu) * val

    def rsd_filter_betaG2_poly(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return betaG2_poly_interkmu(absk, mu) * val

    def rsd_filter_beta3_poly(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return beta3_poly_interkmu(absk, mu) * val

    # def beta1_poly_interkmu(k,mu):
    #     return np.dot(b1_params, np.array([np.ones_like(k*mu), k, k**2, k**4, (k*mu)**2, (k*mu)**4]))
    # def beta2_poly_interkmu(k,mu):
    #     return np.dot(b2_params, np.array([np.ones_like(k*mu), k**2, k**4, (k*mu)**2, (k*mu)**4]))
    # def betaG2_poly_interkmu(k,mu):
    #     return np.dot(bG2_params, np.array([np.ones_like(k*mu), k**2, k**4, (k*mu)**2, (k*mu)**4]))
    # def beta3_poly_interkmu(k,mu):
    #     return np.dot(b3_params, np.array([np.ones_like(k), k**2, k**4, (k*mu)**2, (k*mu)**4]))

    def beta1_poly_interkmu(k,mu):
        return np.dot(np.array([np.ones_like(k), k, k**2, k**4, (k*mu)**2, (k*mu)**4]).T, b1_params).T

    def beta2_poly_interkmu(k,mu):
        return np.dot(np.array([np.ones_like(k), k**2, k**4, (k*mu)**2, (k*mu)**4]).T, b2_params).T

    def betaG2_poly_interkmu(k,mu):
        return np.dot(np.array([np.ones_like(k), k**2, k**4, (k*mu)**2, (k*mu)**4]).T, bG2_params).T

    def beta3_poly_interkmu(k,mu):
        return np.dot(np.array([np.ones_like(k), k**2, k**4, (k*mu)**2, (k*mu)**4]).T, b3_params).T

    beta11_poly = d1.r2c().apply(rsd_filter_beta1_poly, kind='wavenumber')
    beta11_poly[np.isnan(beta11_poly)]=0+0j
    beta22_poly = d2ort.r2c().apply(rsd_filter_beta2_poly, kind='wavenumber')
    beta22_poly[np.isnan(beta22_poly)]=0+0j
    betaG2G2_poly = dG2ort.r2c().apply(rsd_filter_betaG2_poly, kind='wavenumber')
    betaG2G2_poly[np.isnan(betaG2G2_poly)]=0+0j
    beta33_poly = d3ort.r2c().apply(rsd_filter_beta3_poly, kind='wavenumber')
    beta33_poly[np.isnan(beta33_poly)]=0+0j

    final_field_poly = dz.r2c() - 3./7.*fout*dG2par.r2c() + beta11_poly + beta22_poly + betaG2G2_poly + beta33_poly

    return final_field_poly

def rsd_polynomial_field_zout(dz, d1, d2ort, dG2ort, dG2par, d3ort, path, zout, p1, fout):

    # available redshifts
    z_arr = np.array([0,0.5,1,1.5,2,3,5])
    
    rsd_b1_params_z = np.zeros((z_arr.size, 6))
    rsd_b2_params_z = np.zeros((z_arr.size, 5))
    rsd_bG2_params_z = np.zeros((z_arr.size, 5))
    rsd_b3_params_z = np.zeros((z_arr.size, 5))

    assert (zout>=0) and (zout<=5)
    
    for iz, zi in enumerate(z_arr):
        rsd_b1_params_z[iz,:] = np.loadtxt(path+'rsd_b1_poly_zout_%.1f.txt'%zi, unpack=True)
        rsd_b2_params_z[iz,:] = np.loadtxt(path+'rsd_b2_poly_zout_%.1f.txt'%zi, unpack=True)
        rsd_bG2_params_z[iz,:] = np.loadtxt(path+'rsd_bG2_poly_zout_%.1f.txt'%zi, unpack=True)
        rsd_b3_params_z[iz,:] = np.loadtxt(path+'rsd_b3_poly_zout_%.1f.txt'%zi, unpack=True)
        
    # interpolate along redshifts and take the value at zout       
    rsd_b1_params_zout = interp.interp1d(z_arr, rsd_b1_params_z, axis=0)(zout)
    rsd_b2_params_zout = interp.interp1d(z_arr, rsd_b2_params_z, axis=0)(zout)
    rsd_bG2_params_zout = interp.interp1d(z_arr, rsd_bG2_params_z, axis=0)(zout)
    rsd_b3_params_zout = interp.interp1d(z_arr, rsd_b3_params_z, axis=0)(zout)

    def rsd_filter_beta1_poly(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return beta1_poly_interkmu(absk, mu) * val

    def rsd_filter_beta2_poly(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return beta2_poly_interkmu(absk, mu) * val

    def rsd_filter_betaG2_poly(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return betaG2_poly_interkmu(absk, mu) * val

    def rsd_filter_beta3_poly(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = sum(k3vec[i] * p1.attrs['los'][i] for i in range(3)) / absk
        return beta3_poly_interkmu(absk, mu) * val

    def beta1_poly_interkmu(k,mu):
        return np.dot(np.array([np.ones_like(k), k, k**2, k**4, (k*mu)**2, (k*mu)**4]).T, rsd_b1_params_zout).T

    def beta2_poly_interkmu(k,mu):
        return np.dot(np.array([np.ones_like(k), k**2, k**4, (k*mu)**2, (k*mu)**4]).T, rsd_b2_params_zout).T

    def betaG2_poly_interkmu(k,mu):
        return np.dot(np.array([np.ones_like(k), k**2, k**4, (k*mu)**2, (k*mu)**4]).T, rsd_bG2_params_zout).T

    def beta3_poly_interkmu(k,mu):
        return np.dot(np.array([np.ones_like(k), k**2, k**4, (k*mu)**2, (k*mu)**4]).T, rsd_b3_params_zout).T

    beta11_poly = d1.apply(rsd_filter_beta1_poly, kind='wavenumber')
    beta11_poly[np.isnan(beta11_poly)]=0+0j
    beta22_poly = d2ort.apply(rsd_filter_beta2_poly, kind='wavenumber')
    beta22_poly[np.isnan(beta22_poly)]=0+0j
    betaG2G2_poly = dG2ort.apply(rsd_filter_betaG2_poly, kind='wavenumber')
    betaG2G2_poly[np.isnan(betaG2G2_poly)]=0+0j
    beta33_poly = d3ort.apply(rsd_filter_beta3_poly, kind='wavenumber')
    beta33_poly[np.isnan(beta33_poly)]=0+0j

    final_field_poly = dz - 3./7.*fout*dG2par + beta11_poly + beta22_poly + betaG2G2_poly + beta33_poly

    return final_field_poly

def noise(zout, Nmesh, BoxSize):

    noise_seed = np.random.randint(0,1000000)

    pm = ParticleMesh([Nmesh,Nmesh,Nmesh], BoxSize)
    wn = pm.generate_whitenoise(noise_seed)

    if zout==0:
        perr_level = 70.75
    elif zout==1:
        perr_level = 34.9

    def Perr_level(k):
        return np.ones_like(k)*perr_level

    return wn.apply(lambda k, val: Perr_level(sum(ki ** 2 for ki in k)**0.5) ** 0.5 * val / val.BoxSize.prod() ** 0.5)

def noise_zout(zout, Nmesh, BoxSize, path):

    noise_seed = np.random.randint(0,1000000)

    pm = ParticleMesh([Nmesh,Nmesh,Nmesh], BoxSize)
    wn = pm.generate_whitenoise(noise_seed)

    z, perrz = np.loadtxt(path + "z_Perr.txt", unpack=True)
    perr_zout = np.interp(zout, z, perrz)
    
    def Perr_level(k):
        return np.ones_like(k)*perr_zout

    return wn.apply(lambda k, val: Perr_level(sum(ki ** 2 for ki in k)**0.5) ** 0.5 * val / val.BoxSize.prod() ** 0.5)

def noise_kmu(zout, Nmesh, BoxSize, axis, fout, path):

    noise_seed = np.random.randint(0,1000000)

    pm = ParticleMesh([Nmesh,Nmesh,Nmesh], BoxSize)
    wn = pm.generate_whitenoise(noise_seed)

    a0, a2, a3, a4, a22, a33, a44 = np.loadtxt(path + 'Perr_polyfit_zout_%.1f.txt'%zout, unpack=True)
    los = np.zeros(3, dtype=int)
    los[axis]=1    
    # print ('los ', los)

    def Perr_kmu_model(k,mu):
        return a0 + a2*k**2 + a3*k**3 + a4*k**4 + a22*(k*mu)**2 + a33*(k*mu)**3 + a44*(k*mu)**4

    def Perr_kmu_function(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore',
                         divide='ignore'):
            mu = sum(k3vec[i] * los[i] for i in range(3)) / absk
        return Perr_kmu_model(absk, mu)**0.5 * val / val.BoxSize.prod() ** 0.5

    wn = wn.apply(Perr_kmu_function, kind='wavenumber')
    wn[np.isnan(wn)]=0+0j

    return wn

def noise_kmu_zout(zout, Nmesh, BoxSize, axis, fout, path):

    noise_seed = np.random.randint(0,1000000)

    pm = ParticleMesh([Nmesh,Nmesh,Nmesh], BoxSize)
    wn = pm.generate_whitenoise(noise_seed)
    
    # available redshifts
    z_arr = np.array([0,0.5,1,1.5,2,3,5])
    
    noise_params_z = np.zeros((z_arr.size, 7))

    assert (zout>=0) and (zout<=5)
    
    for iz, zi in enumerate(z_arr):
        noise_params_z[iz,:] = np.loadtxt(path+'Perr_polyfit_zout_%.1f.txt'%zi, unpack=True)

    # interpolate along redshifts and take the value at zout       
    noise_params_zout = interp.interp1d(z_arr, noise_params_z, axis=0)(zout)
    a0, a2, a3, a4, a22, a33, a44 = noise_params_zout

    los = np.zeros(3, dtype=int)
    los[axis]=1    

    def Perr_kmu_model(k,mu):
        return a0 + a2*k**2 + a3*k**3 + a4*k**4 + a22*(k*mu)**2 + a33*(k*mu)**3 + a44*(k*mu)**4

    def Perr_kmu_function(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore',
                         divide='ignore'):
            mu = sum(k3vec[i] * los[i] for i in range(3)) / absk
        return Perr_kmu_model(absk, mu)**0.5 * val / val.BoxSize.prod() ** 0.5

    wn = wn.apply(Perr_kmu_function, kind='wavenumber')
    wn[np.isnan(wn)]=0+0j

    return wn

def _noise_kmu_(zout, Nmesh, BoxSize, axis, fout, path):

    noise_seed = np.random.randint(0,1000000)

    pm = ParticleMesh([Nmesh,Nmesh,Nmesh], BoxSize)
    wn = pm.generate_whitenoise(noise_seed)

    c1, c2 = np.loadtxt(path + 'perr_kmu_fit_zout_%.1f.txt'%zout, unpack=True)
    los = np.zeros(3, dtype=int)
    los[axis]=1    
    # print ('los ', los)

    def Perr_kmu_model(k,mu):
        # c1, c2 = theta
        return c1 + fout*c2*(k*mu)**2

    def Perr_kmu_function(k3vec, val):
        absk = (sum(ki**2 for ki in k3vec))**0.5
        with np.errstate(invalid='ignore',
                         divide='ignore'):
            mu = sum(k3vec[i] * los[i] for i in range(3)) / absk
        return Perr_kmu_model(absk, mu)**0.5 * val / val.BoxSize.prod() ** 0.5

    wn = wn.apply(Perr_kmu_function, kind='wavenumber')
    wn[np.isnan(wn)]=0+0j

    return wn

def generate_fields_new_growth(dlin, prefactor, zic, zout, comm=None, compensate=True):
    delta_ic = dlin.copy()
    Nmesh = delta_ic.Nmesh
    BoxSize = delta_ic.BoxSize[0]
    
    pos = delta_ic.pm.generate_uniform_particle_grid(shift=0)
    N = pos.shape[0]
    catalog = np.empty(N, dtype=[('Position', ('f8', 3)), ('delta_1', 'f8'), ('delta_2', 'f8'), ('delta_G2', 'f8'), ('delta_3', 'f8')])
    catalog['Position'][:] = pos[:]
    layout = delta_ic.pm.decompose(catalog['Position']) 
    del pos
    
    delta_1 = delta_ic.c2r().readout(catalog['Position'], layout=layout, resampler='cic')*prefactor
    delta_1 -= np.mean(delta_1)
    catalog['delta_1'][:] = delta_1[:]
    del delta_1
    
    catalog['delta_2'][:] = catalog['delta_1']**2
    catalog['delta_2'][:] -= np.mean(catalog['delta_2'])
    
    delta_G2 = tidal_G2(FieldMesh(delta_ic)).readout(catalog['Position'], layout=layout, resampler='cic')*prefactor**2
    catalog['delta_G2'][:] = delta_G2[:]
    del delta_G2
    
    delta_3 = d3_smooth(FieldMesh(delta_ic)).readout(catalog['Position'], layout=layout, resampler='cic')**3*prefactor**3
    catalog['delta_3'][:] = delta_3[:]
    del delta_3

    def potential_transfer_function(k, v):
        k2 = k.normp(zeromode=1)
        return v / (k2)
    pot_k = delta_ic.apply(potential_transfer_function, out=Ellipsis)

    displ_catalog = np.empty(N, dtype=[('displ', ('f8',3))])
    
    for d in range(3):
        def force_transfer_function(k, v, d=d):
            return k[d] * 1j * v
        force_d = pot_k.apply(force_transfer_function).c2r(out=Ellipsis)
        displ_catalog['displ'][:, d] = force_d.readout(catalog['Position'], layout=layout, resampler='cic')*prefactor
    
    catalog['Position'][:] = (catalog['Position'][:] + displ_catalog['displ'][:]) % BoxSize
    del displ_catalog, force_d, pot_k
    
    catalog = ArrayCatalog(catalog, BoxSize=BoxSize * np.ones(3), Nmesh=Nmesh, comm=comm)
    
    d1 = catalog.to_mesh(value='delta_1', compensated=compensate).paint().r2c()
    d2 = catalog.to_mesh(value='delta_2', compensated=compensate).paint().r2c()
    dG2 = catalog.to_mesh(value='delta_G2', compensated=compensate).paint().r2c()
    d3 = catalog.to_mesh(value='delta_3', compensated=compensate).paint().r2c()
    
    return d1, d2, dG2, d3

def decic(field, n=2):
    """
    Computes CIC compensation
    Adapted from https://github.com/modichirag/21cm_cleaning/blob/1615fea4e2d617bb6ef00770a49698901227daa8/code/utils/features.py#L101
    
    """
    def tf(k):
        kny = [np.sinc(k[i]*field.attrs['BoxSize'][i]/(2*np.pi*field.attrs['Nmesh'][i])) for i in range(3)]
        wts = (kny[0]*kny[1]*kny[2])**(-1*n)
        return wts
        
    if field.dtype == 'complex128' or field.dtype == 'complex64':
        toret = field.apply(lambda k, v: tf(k)*v).c2r()
    elif field.dtype == 'float32' or field.dtype == 'float64':
        toret = field.to_real_field().r2c().apply(lambda k, v: tf(k)*v).c2r()
    return toret