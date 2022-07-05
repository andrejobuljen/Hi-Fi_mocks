# Hi-Fi mocks

Codes to generate fast HI field-level (Hi-Fi) mocks in real and redshift space.

## About

Codes to produce 3D neutral hydrogen (HI) over-density fields in real and redshift space tuned to HI clustering properties of Illustris TNG300-1 (L=205 Mpc/h). These codes allow quick and accurate production of HI mocks. This code is accompanying the paper: arXiv:XXXX. It is based on the perturbative approach from [Schmittfull+18](https://arxiv.org/abs/1811.10640) & [Schmittfull+19](https://arxiv.org/abs/2012.03334).

## Usage

`Hi-Fi mocks` are based on [nbodykit](https://github.com/bccp/nbodykit) package. To install `nbodykit` please follow these installation [instructions](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html).

To generate HI mock in real space run the following:

``python Hi-Fi_real_space.py``

To generate HI mock in redshift space run the following:

``python Hi-Fi_redshift_space.py``

The parameters for each run (`BoxSize`, `Nmesh`, initial condition `seed`, resolution) can be set within these two codes. Based on these given parameters these codes produce HI meshes in real & redshift space using best-fit polynomials for transfer functions. The fit is tuned to avaialable scales of TNG300 (k in 0.03-1 h/Mpc) and output redshifts 0 & 1. The codes also produce the overdensity slice and power spectra plot.

### Author
- [Andrej Obuljen](mailto:andrej.obuljen@uzh.ch) (ICS, Zurich)

Acknowledgement: Parts of our code uses scripts from [perr](https://github.com/mschmittfull/perr) and [lsstools](https://github.com/mschmittfull/lsstools).
