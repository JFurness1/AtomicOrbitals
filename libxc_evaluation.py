from AtomicOrbitals import GridGenerator, AtomData, Atom
from math import pi
import numpy as np
import pylibxc
import os

# Check for exceptions?
if True:
    from ctypes import CDLL, c_int
    libm = CDLL('libm.so.6')
    #libm.feenableexcept(c_int(1)) # invalid
    #libm.feenableexcept(c_int(4)) # division by zero
    #libm.feenableexcept(c_int(8)) # overflow
    libm.feenableexcept(c_int(13)) # the three above
    #libm.feenableexcept(c_int(16)) # underflow
    #libm.feenableexcept(c_int(32)) # inexact
    #libm.feenableexcept(c_int(61)) # all exceptions

dens_thresh = None
sigma_thresh = None
actual, r, wt = GridGenerator.make_grid(2000)
grid = 4*pi*wt

print('Using a {} point grid'.format(grid.size))
print("Running with libxc version {}".format(pylibxc.util.xc_version_string()))
if dens_thresh is not None:
    print('USING OVERWRITTEN DENSITY THRESHOLD {}'.format(dens_thresh))
if sigma_thresh is not None:
    print('USING OVERWRITTEN SIGMA THRESHOLD {}'.format(sigma_thresh))

# Create directory
dir=pylibxc.util.xc_version_string()
if not os.path.exists(dir):
    os.mkdir(dir)

data = AtomData()

funcs=pylibxc.util.xc_available_functional_names()
x_funcs=[f for f in funcs if f.find("_x_")!=-1]
c_funcs=[f for f in funcs if f.find("_c_")!=-1]
k_funcs=[f for f in funcs if f.find("_k_")!=-1]
xc_funcs=[f for f in funcs if f.find("_xc_")!=-1]

for restricted in [True, False]:
    #for func in funcs:
    #    if func.startswith('lda_') or func.startswith('gga_'):
    #        continue
    for func in ['mgga_x_lak', 'mgga_c_lak']:
        feval = pylibxc.LibXCFunctional(func, "unpolarized" if restricted else "polarized")
        if not feval._have_exc:
            print('{} doesn\'t have exc'.format(func))
            continue
        # Skip badly behaving functionals
        #bad_funcs = ['mgga_c_b88', 'mgga_c_b94', 'gga_c_op_', 'mgga_x_br89_explicit', 'mgga_k_csk', 'mgga_x_edmgga', 'hyb_mgga_xc_edmggah', 'hyb_mgga_xc_b94_hyb']
        #    bad_funcs = ['mgga_k_csk', 'mgga_c_b88', 'mgga_c_b94', 'mgga_x_edmgga', 'hyb_mgga_xc_edmggah', 'mgga_x_br89_explicit', 'hyb_mgga_xc_b94_hyb']
        #    bad_funcs = ['mgga_x_edmgga', 'hyb_mgga_xc_edmggah']
        bad_funcs = []
        if func in bad_funcs:
            print('skipping ill-behaving {}'.format(func))
            continue

        # LDAs and GGAs are ok
        #if func.find('lda_')!=-1:
        #    print('skipping lda {}'.format(func))
        #    continue
        #if func.find('mgga_')==-1:
        #    print('skipping non-mgga {}'.format(func))
        #    continue

        print(f'{func} {restricted}')

        outf=open(f'{dir}/{func}_{restricted}','w')
        for a in list(data.ke_test.keys()):
            atom = Atom(a)

            d0, d1, _, _, _, _, _, _ = atom.get_densities(r)
            nel = np.dot(d0+d1, grid)

            nE, vrho, vsigma, vtau, vlapl = atom.libxc_eval(r, functional=func, restricted=restricted, density_threshold=dens_thresh, sigma_threshold=sigma_thresh, nan_check=True)
            Exc = (np.dot(nE, grid)).item()
            outf.write('{:3s} {: .10e}\n'.format(a, Exc))
        outf.close()
