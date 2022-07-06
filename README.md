# Hi-Fi mocks

Codes to generate fast HI field-level (Hi-Fi) mocks in real and redshift space.

## About

Codes to produce 3D neutral hydrogen (HI) over-density fields in real and redshift space tuned to HI clustering properties of [Illustris](https://www.tng-project.org) TNG300-1 (L=205 Mpc/h). These codes allow quick and accurate production of HI mocks. This code is accompanying the paper: arXiv:XXXX. It is based on the perturbative approach from [Schmittfull+18](https://arxiv.org/abs/1811.10640) & [+19](https://arxiv.org/abs/2012.03334).

## Usage

`Hi-Fi mocks` are based on and require [nbodykit](https://github.com/bccp/nbodykit) package. To install `nbodykit` please follow these [instructions](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html) (preferably using `conda`). Once installed, also install `matplotlib` package by running: `conda install matplotlib` within the `nbodykit` conda environment.

After installing `nbodykit` package, download `Hi-Fi mocks` code. This can be done for example via terminal using:
`git clone https://github.com/andrejobuljen/Hi-Fi_mocks.git`. 

Enter the folder `Hi-fi_mocks`: `cd Hi-fi_mocks` and follow the next steps.

To generate HI mock in real space run the following:

``python Hi-Fi_mock_real_space.py``

To generate HI mock in redshift space run the following:

``python Hi-Fi_mock_redshift_space.py``

The parameters for each run: `BoxSize`, `Nmesh`, initial condition (IC) `seed`, resolution etc. can be set within the main codes. Based on these given parameters these codes produce HI meshes in real & redshift space using best-fit polynomials for transfer functions. The fits are tuned to scales of TNG300-1 (k range: 0.03-1 h/Mpc) and limited to output redshifts of z = 0 & 1.

For the fiducial parameters (`BoxSize=205 Mpc/h`, `Nmesh=256^3`, `nbar=1 (h/Mpc)^3`), each code finishes in ~2 minutes on a modern laptop. The codes output the final HI overdensity field, figure with smoothed overdensity slice and the measured power spectra directly to the `output_folder`.

### Author
- [Andrej Obuljen](mailto:andrej.obuljen@uzh.ch) (ICS, Zurich)

### Acknowledgement
Parts of our code uses scripts from [lsstools](https://github.com/mschmittfull/lsstools).
