import numpy as np
from math import pi, log, factorial
from literature_data.parser import parse
import os

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MAKE_MPL_COLOR_MAP = True
except ImportError:
    MAKE_MPL_COLOR_MAP = False

try:
    import pylibxc
    HAVE_LIBXC = True
except ImportError:
    HAVE_LIBXC = False

try:
    import xcfun
    HAVE_XCFUN = True
except ImportError:
    HAVE_XCFUN = False

try:
    import scipy
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

"""Code to calculate the electron density, its gradient and laplacian,
as well as the local kinetic energy density from analytic Slater-type
orbital expansions of atomic Hartree-Fock wave functions published in
the literature.

Although the best-known Hartree-Fock wave functions for atoms are the
ones published by Clementi and Roetti in 1974, their wave functions
have significant errors (up to millihartree), especially for the
heavier atoms. Instead, the wave functions included in this library
for H-Xe and their cations and anions are from

Koga et al, "Analytical Hartree–Fock wave functions subject to cusp
and asymptotic constraints: He to Xe, Li+ to Cs+, H− to I−",
Int. J. Quantum Chem. 71, 491 (1999),
doi: 10.1002/(SICI)1097-461X(1999)71:6<491::AID-QUA6>3.0.CO;2-T

which yield total eneregies that deviate from the exact numerical
solution only by microhartrees. Included are also non-relativistic
Hartree-Fock wave functions for heavier atoms from

Koga et al, "Analytical Hartree–Fock wave functions for the atoms Cs
to Lr", Theor. Chem. Acc. 104, 411 (2000), doi: 10.1007/s002140000150

which have an average error of 0.19 mEh compared to exact, fully
numerical calculations.

The core functionality of the library comes from the Atom() class
which implements the get_densities(r) function to return the
above-mentioned quantities at a given numpy array of radial points r.

The Hartree-Fock orbital data are stored in the static AtomData class,
which implements dictionaries of parameters for various elements that
are accessed by upper-case element symbols.

The GridGenerator gives a simple implementation of an efficient
Gauss-Legendre quadrature grid.

A color dictionary following colors used by Jmol (roughly CPK) by
element symbol.  Should be accessed using get_colors_for_elements()
which returns a list of RGB for an input list of string element labels
or Atom objects.

Module is Python 2 and 3 compatible.

Examples of use can be seen in test_densities() function.

"""


def test_densities():
    """
    Simple unit testing routine.

    All functions were validated against the reference FORTRAN code to good agreement.
    However, it is possible to test the gradient by finite difference, so lets do that.
    The Laplacian does not seem so well behaved (but agrees with the reference).
    This should not need re-testing as it is independent of input data, but it can't hurt.

    As long as reference total electron number and kinetic energies are included
    for the atom specification we can check the Slater orbital parameters that have been
    entered by integrating the density and kinetic energy density. Good agreement
    indicates parameters have been entered correctly.

    Whilst the threshold value on these integrations may seem a bit high,
    painstaking cross checking shows that even with correct values
    and converged grids, this is as close as we get. Most small changes seem to break it
    so we can have confidence.
    """

    actual, r, wt = GridGenerator.make_grid(400, 'ahlrichsm3')
    grid = 4*pi*wt

    data = AtomData()

    print("\nINTEGRATED DENSITY TEST")
    print("=======================")
    for a in list(data.nuclear_charge.keys()):
        atom = Atom(a)
        Nel = data.electron_count[a]
        d0, d1, g0, g1, t0, t1, l0, l1 = atom.get_densities(r)

        # Count electrons per spin channel
        s_occ = AtomData.s_occ.get(a, [0, 0])
        p_occ = AtomData.p_occ.get(a, [0, 0])
        d_occ = AtomData.d_occ.get(a, [0, 0])
        f_occ = AtomData.f_occ.get(a, [0, 0])
        nela = np.sum(s_occ[0])+np.sum(p_occ[0])+np.sum(d_occ[0])+np.sum(f_occ[0])
        nelb = np.sum(s_occ[1])+np.sum(p_occ[1])+np.sum(d_occ[1])+np.sum(f_occ[1])
        assert(nela+nelb == Nel)

        id0 = np.dot(d0, grid)
        id1 = np.dot(d1, grid)

        diff_0 = id0 - nela
        percent_diff_0 = 100*diff_0/nela

        # Check to catch for Hydrogen having no beta electrons
        if nelb > 0.0:
            diff_1 = id1 - nelb
            percent_diff_1 = 100*diff_1/nelb
        else:
            diff_1 = 0.0
            percent_diff_1 = 0.0

        print("{:>3} - N_0 = ({:4.1f}) {:+2.6e}%, N_1 = ({:4.1f}) {:+2.6e}%, {:}".format(a, id0, percent_diff_0, id1, percent_diff_1, "PASSED" if max(abs(diff_0), abs(diff_1)) < 1e-4 else "FAILED - "))

    print("\nINTEGRATED KINETIC TEST")
    print("=======================")
    for a in list(data.ke_test.keys()):
        atom = Atom(a)
        t_bm = data.ke_test[a]
        d0, d1, g0, g1, t0, t1, l0, l1 = atom.get_densities(r)

        it0 = np.dot(t0, grid)
        it1 = np.dot(t1, grid)
        itot = it0 + it1

        diff = itot - t_bm
        print("{:>3} - T = {:+.6e}%, {:}".format(a, 100*diff/t_bm, "PASSED" if abs(100*diff/t_bm) < 1e-2 else "FAILED - "))


    # The integral of the Laplacian over all space should be 0. Check that.
    print("\nINTEGRATED LAPLACIAN TEST")
    print("=========================")
    for a in list(AtomData.ke_test.keys()):
        atom = Atom(a)

        d0, d1, g0, g1, t0, t1, l0, l1 = atom.get_densities(r)

        il0 = np.dot(grid, l0)
        il1 = np.dot(grid, l1)
        print("{:>3} - L_0 = {:+.6e}, L_1 = {:+.6e}, {:}".format(a, il0, il1, "PASSED" if max(abs(il0), abs(il1)) < 1e-6 else "FAILED - "))


    print("\nFINITE DIFFERENCE GRADIENT TEST")
    print("===============================")
    print("Testing gradient evaluation function against finite difference estimate...")
    ne = Atom("Ne")  # Let's use "the guvnor"
    # We only need to test a few points around the core
    fdh = 1e-8
    fdr = np.arange(0.9, 0.9+fdh*10, fdh)
    d0, d1, g0, g1, t0, t1, l0, l1 = ne.get_densities(fdr)

    # First the first central difference
    fdiff = (d0[2:]-d0[:-2])/(2*fdh)  # Construct the central difference
    if np.allclose(fdiff, g0[1:-1], atol=1e-1):  # finite difference is not perfect, so lenient tolerance
        print("Gradient: PASSED")
    else:
        print("Gradient: FAILED -")

    print("\nELEMENT COLOR FUNCTIONS TEST")
    print("===========================")
    test_obj = [Atom("H"), Atom("C"), Atom("O")]
    test_str = ["H", "C", "O"]
    ref = np.array([[1., 1., 1.], [0.565, 0.565, 0.565], [1.   , 0.051, 0.051]])

    if np.allclose( np.array(get_colors_for_elements(test_obj)), ref):
        print("\nColor from objects: PASSED")
    else:
        print("\nColor from objects: FAILED -")

    if np.allclose( np.array(get_colors_for_elements(test_str)), ref):
        print("Color from strings: PASSED")
    else:
        print("Color from strings: FAILED -")

    if HAVE_LIBXC:
        test_functional='GGA_X_PBE'
        print("\nATOMIC EXCHANGE ENERGIES WITH {}".format(test_functional))
        print("============================================")
        for a in list(data.ke_test.keys()):
            atom = Atom(a)
            nE, vrho, vsigma, vtau, vlapl = atom.libxc_eval(r, functional=test_functional, restricted=False)
            Exc = (np.dot(nE, grid)).item()
            print('{:3s} {:.10f}'.format(a, Exc))
    else:
        print("\nNot doing energy calculations due to lack of libxc.\n")

class Atom:
    """
    Object initialised to contain the relevant data
    for the element provided to the constructor.

    The returned Atom object then has the core method
        get_densities(r)
    that takes a distance (or numpy array of distances)
    and returns the corresponding densities.
    """

    def __init__(self, element):
        # Currently only deal with atomic symbols.
        # Eventually add dictionary to support full names and numbers?
        self.element = element
        try:
            u_atom = element.upper()
            d_atom = element.lower()

            # set nuclear charge
            self.nuclear_charge = AtomData.nuclear_charge[u_atom]

            # We assume that every atom has S electrons
            # So access with index notation to raise KeyError on missing atom.
            self.s_exp = AtomData.s_exp[u_atom]
            self.s_coef = AtomData.s_coef[u_atom]
            self.s_n = AtomData.s_n[u_atom]
            self.s_occ = AtomData.s_occ[u_atom]

            # It is possible to be missing P and D electrons, so use get with defaults
            self.p_exp = AtomData.p_exp.get(u_atom, None)
            self.p_coef = AtomData.p_coef.get(u_atom, None)
            self.p_n = AtomData.p_n.get(u_atom, None)
            self.p_occ = AtomData.p_occ.get(u_atom, [0, 0])
            self.d_exp = AtomData.d_exp.get(u_atom, None)
            self.d_coef = AtomData.d_coef.get(u_atom, None)
            self.d_n = AtomData.d_n.get(u_atom, None)
            self.d_occ = AtomData.d_occ.get(u_atom, [0, 0])
            self.f_exp = AtomData.f_exp.get(u_atom, None)
            self.f_coef = AtomData.f_coef.get(u_atom, None)
            self.f_n = AtomData.f_n.get(u_atom, None)
            self.f_occ = AtomData.f_occ.get(u_atom, [0, 0])
        except KeyError:
            raise KeyError('Error: Atom data for "{:}" missing'.format(element))

    def get_densities(self, r):
        """
        Core function returning all densities for the initialised atom.
        Makes use of numpy vectorisation to efficiently compute an array
        of r.

        In:
            r       space coordinate (distance from nucleus) [numpy array or scalar]
        Out (Throughout X denotes spin 0/1 (up/down). All numpy arrays.):
            den_X   Electron density
            grd_X   Gradient of electron density
            tau_X   Orbital kinetic energy density
            lap_X   Laplacian of electron density
        """
        # Handle scalar r values as single element array
        if np.isscalar(r):
            r = np.array([r])

        assert np.min(r) > 0, "Error: distances must be non-zero and positive."

        if self.s_exp is not None:
            oS, doS, ddoS = self.get_orbitals(self.s_n, self.s_exp, self.s_coef, r)

            den_0 = np.sum(self.s_occ[0][:,None]*oS**2, axis=0)
            den_1 = np.sum(self.s_occ[1][:,None]*oS**2, axis=0)

            grd_0 = np.sum(self.s_occ[0][:,None]*(oS*doS), axis=0)
            grd_1 = np.sum(self.s_occ[1][:,None]*(oS*doS), axis=0)

            tau_0 = np.sum(self.s_occ[0][:,None]*doS**2, axis=0)
            tau_1 = np.sum(self.s_occ[1][:,None]*doS**2, axis=0)

            lap_s = oS*ddoS + doS**2 + 2*oS*doS/r
            lap_0 = np.sum(self.s_occ[0][:,None]*lap_s, axis=0)
            lap_1 = np.sum(self.s_occ[1][:,None]*lap_s, axis=0)
        else:
            # Otherwise supply zeros in place
            den_0 = np.zeros(r.shape)
            den_1 = np.zeros(r.shape)

            grd_0 = np.zeros(r.shape)
            grd_1 = np.zeros(r.shape)

            tau_0 = np.zeros(r.shape)
            tau_1 = np.zeros(r.shape)

            lap_0 = np.zeros(r.shape)
            lap_1 = np.zeros(r.shape)

        # Check if atom has occupied P orbitals
        if self.p_exp is not None:
            oP, doP, ddoP = self.get_orbitals(self.p_n, self.p_exp, self.p_coef, r)

            den_0 += np.sum(self.p_occ[0][:,None]*oP**2, axis=0)
            den_1 += np.sum(self.p_occ[1][:,None]*oP**2, axis=0)

            grd_0 += np.sum(self.p_occ[0][:,None]*oP*doP, axis=0)
            grd_1 += np.sum(self.p_occ[1][:,None]*oP*doP, axis=0)

            tau_0 += np.sum(self.p_occ[0][:,None]*(doP**2 + 2*(oP/r)**2), axis=0)
            tau_1 += np.sum(self.p_occ[1][:,None]*(doP**2 + 2*(oP/r)**2), axis=0)

            lap_p = oP*ddoP + doP**2 + 2*oP*doP/r
            lap_0 += np.sum(self.p_occ[0][:,None]*lap_p, axis=0)
            lap_1 += np.sum(self.p_occ[1][:,None]*lap_p, axis=0)

        # Check if atom has occupied D orbitals
        if self.d_exp is not None:
            oD, doD, ddoD = self.get_orbitals(self.d_n, self.d_exp, self.d_coef, r)
            den_0 += np.sum(self.d_occ[0][:,None]*oD**2, axis=0)
            den_1 += np.sum(self.d_occ[1][:,None]*oD**2, axis=0)

            grd_0 += np.sum(self.d_occ[0][:,None]*oD*doD, axis=0)
            grd_1 += np.sum(self.d_occ[1][:,None]*oD*doD, axis=0)

            tau_0 += np.sum(self.d_occ[0][:,None]*(doD**2 + 6*(oD/r)**2), axis=0)
            tau_1 += np.sum(self.d_occ[1][:,None]*(doD**2 + 6*(oD/r)**2), axis=0)

            lap_d = oD*ddoD + doD**2 + 2*oD*doD/r
            lap_0 += np.sum(self.d_occ[0][:,None]*lap_d, axis=0)
            lap_1 += np.sum(self.d_occ[1][:,None]*lap_d, axis=0)

        # Check if atom has occupied F orbitals
        if self.f_exp is not None:
            oF, doF, ddoF = self.get_orbitals(self.f_n, self.f_exp, self.f_coef, r)
            den_0 += np.sum(self.f_occ[0][:,None]*oF**2, axis=0)
            den_1 += np.sum(self.f_occ[1][:,None]*oF**2, axis=0)

            grd_0 += np.sum(self.f_occ[0][:,None]*oF*doF, axis=0)
            grd_1 += np.sum(self.f_occ[1][:,None]*oF*doF, axis=0)

            tau_0 += np.sum(self.f_occ[0][:,None]*(doF**2 + 12*(oF/r)**2), axis=0)
            tau_1 += np.sum(self.f_occ[1][:,None]*(doF**2 + 12*(oF/r)**2), axis=0)

            lap_f = oF*ddoF + doF**2 + 2*oF*doF/r
            lap_0 += np.sum(self.f_occ[0][:,None]*lap_f, axis=0)
            lap_1 += np.sum(self.f_occ[1][:,None]*lap_f, axis=0)

        # Take care of scaling
        den_0 /= 4*pi
        den_1 /= 4*pi

        grd_0 /= 2*pi
        grd_1 /= 2*pi

        tau_0 /= 8*pi
        tau_1 /= 8*pi

        lap_0 /= 2*pi
        lap_1 /= 2*pi

        return den_0, den_1, grd_0, grd_1, tau_0, tau_1, lap_0, lap_1

    def get_range(self):
        """
        Evaluates the maximum range of the basis, at which the exponential factor for the smallest exponent is numerically zero.
        """
        min_exp = np.finfo(np.core.numerictypes.float64).max
        if self.s_exp is not None:
            min_exp = min(min_exp, min(self.s_exp))
        if self.p_exp is not None:
            min_exp = min(min_exp, min(self.p_exp))
        if self.d_exp is not None:
            min_exp = min(min_exp, min(self.d_exp))
        if self.f_exp is not None:
            min_exp = min(min_exp, min(self.f_exp))
        max_range = -log(np.finfo(np.core.numerictypes.float64).tiny)/min_exp
        return max_range

    def get_orbitals(self, q_numbers, exponents, coefficients, r):
        """
        Evaluates Slater orbitals at position r.
        IN:
            q_numbers = array of principal quantum numbers
            exponents = array of exponents
            coefficients = linear combination coefficients for orbitals
            r = space coordinates
        OUT:
            of = orbital values at each r dimension (orbitals, r)
            dof = first orbital derivatives at r
            ddof = second orbital derivatives
        """
        # Begin by calculating the values of the Slater function, and its first 2 derivatives
        # for the given exponents.
        # gives (n, r) dimension [where n is number of functions]
        f = self.G(q_numbers, exponents, r)
        df = self.DG(q_numbers, exponents, r, f)
        ddf = self.DDG(q_numbers, exponents, r, f)

        # Now construct the orbitals by multiplication with orbital coefficients
        of = np.einsum('ij,jk->ik', coefficients, f)  # (i=orbital, j=function, k=r)
        dof = np.einsum('ij,jk->ik', coefficients, df)
        ddof = np.einsum('ij,jk->ik', coefficients, ddf)
        return of, dof, ddof

    def G(self, n, zeta, r):
        """Evaluates the radial Slater orbital R(r) = N r^{n-1} exp(-zeta r)

        arguments:
            n: principal quantum number
         zeta: exponent
            r: distance from nucleus

        """

        # Principal quantum number factors for STO normalization
        FACTORS = np.array([factorial(2*nn) for nn in range(1,max(n)+1)])**(-0.5)
        n_facs = FACTORS[n - 1]
        try:
            c = n_facs*(2.0*zeta)**(n + 0.5)
        except ValueError:
            print("Exponents and principal number factors are different shapes.")
            print("Did you typo a ',' for a decimal point? e.g. '1,23456' for '1.23456'")
            raise ValueError("exponent or principal number error")
        rn = np.power.outer(r, (n - 1))
        es = np.einsum('j,ij->ji', c, rn)
        pw = np.exp(-np.outer(zeta, r))
        return es*pw

    def DG(self, n, e, r, f):
        """Evaluates the first derivative R'(r) of the radial Slater orbital
        R(r) = N r^{n-1} exp(-zeta r) as R'(r) = [(n-1)/r - zeta] R(r).

        arguments:
            n: principal quantum number
         zeta: exponent
            r: distance from nucleus
            f: undifferentiated function

        """

        pre = -e[:, None] + np.divide.outer((n - 1), r)
        return pre*f

    def DDG(self, n, e, r, f):
        """Evaluates the second derivative R''(r) of the radial Slater orbital
        R(r) = N r^{n-1} exp(-zeta r) as
        R''(r) = {[(n-1)/r - zeta]^2 - (n-1)/r^2} R(r)

        arguments:
            n: principal quantum number
         zeta: exponent
            r: distance from nucleus
            f: undifferentiated function

        """
        pre = (-e[:, None] + np.divide.outer((n - 1), r))**2
        pre -= np.divide.outer((n - 1), r**2)
        return pre*f

    def get_nuclear_potential(self, r):
        """
        Returns the -Z/r potential energy curve for the atom for all points r
        """

        return -self.nuclear_charge/r

    def get_gaussian_nuclear_potential(self, r, gamma=0.2):
        """
        Returns a Gaussian approximation to the nuclear potential as defined
        in
        F. Brockherde, L. Vogt, L. Li, M. E. Tuckerman, K. Burke, and K. R. Müller, Nat. Commun. 8, (2017).
        DOI: 10.1038/s41467-017-00839-3

        v(r) = Z*exp(-r/(2*gamma**2))
        where Z is the nuclear charge and gamma is a width parameter chosen as 0.2 in the reference.
        """

        return -self.nuclear_charge*np.exp(-r**2/(2*gamma**2))

    def get_color(self):
        """
        Returns RGB color of element for plotting.
        """
        return COLOR_DICT[self.element]

    def libxc_eval(self, r, functional='gga_x_pbe', restricted=False, density_threshold=None, sigma_threshold=None, nan_check=False, ext_params=None):
        '''Evaluates a functional with the atomic density data using libxc'''

        d0, d1, g0, g1, t0, t1, l0, l1 = self.get_densities(r)

        if not HAVE_LIBXC:
            raise ImportError('Cannot evaluate functional since pylibxc could not be imported.')

        func = pylibxc.LibXCFunctional(functional, "unpolarized" if restricted else "polarized")

        # Did we get a threshold?
        if density_threshold is not None:
            func.set_dens_threshold(density_threshold)
        if sigma_threshold is not None:
            func.set_sigma_threshold(sigma_threshold)

        # Did we get external parameters?
        if ext_params is not None:
            func.set_ext_params(ext_params)

        # Create input
        inp = {}
        if restricted:
            inp["rho"] = d0+d1
            inp["sigma"] = np.multiply(g0+g1,g0+g1)
            inp["lapl"]= l0+l1
            inp["tau"]= t0+t1
        else:
            rho_array = np.zeros((d0.size,2), dtype='float64')
            sigma_array = np.zeros((d0.size,3), dtype='float64')
            lapl_array = np.zeros((d0.size,2), dtype='float64')
            tau_array = np.zeros((d0.size,2), dtype='float64')

            rho_array[:,0]=d0
            rho_array[:,1]=d1

            sigma_array[:,0]=np.multiply(g0, g0)
            sigma_array[:,1]=np.multiply(g0, g1)
            sigma_array[:,2]=np.multiply(g1, g1)

            lapl_array[:,0]=l0
            lapl_array[:,1]=l1

            tau_array[:,0]=t0
            tau_array[:,1]=t1

            inp["rho"] = rho_array
            inp["sigma"] = sigma_array
            inp["lapl"]= lapl_array
            inp["tau"]= tau_array

        # Compute functional
        ret = func.compute(inp)

        # Get energy density per particle
        zk = ret.get("zk", np.zeros_like(d0))
        if zk.shape != d0.shape:
            zk = np.reshape(zk, d0.shape)
        # Energy density
        nE = np.multiply(zk,d0+d1)

        # First derivatives
        vrho = ret["vrho"]
        vsigma = ret.get("vsigma", np.zeros_like(inp["sigma"]))
        vtau = ret.get("vtau", np.zeros_like(inp["tau"]))
        vlapl = ret.get("vlapl", np.zeros_like(inp["lapl"]))
        # Earlier versions of PyLibXC return the wrong shape, so reshape
        # just to be sure
        vrho = np.reshape(vrho, inp["rho"].shape)
        vsigma = np.reshape(vsigma, inp["sigma"].shape)
        vlapl = np.reshape(vlapl, inp["lapl"].shape)
        vtau = np.reshape(vtau, inp["tau"].shape)

        if nan_check:
            # Indices of NaNs
            nanidx = np.isnan(nE)
            if len(nE[nanidx]):
                print('nanidx len={} with {} nans'.format(nanidx.size, len(nE[nanidx])))
            #print('NaN densities\n{}'.format((d0+d1)[nanidx]))
            for i in range(len(nanidx)):
                if nanidx[i]:
                    if restricted:
                        print('NaN at rho= {:e} sigmaa= {:e} lapl= {: e} tau={:e}'.format(rho_array[i],sigma_array[i], lapl_array[i], tau_array[i]))
                    else:
                        print('NaN at rhoa= {:e} rhob= {:e} sigmaaa= {:e} sigmaab= {: e} sigmabb= {:e} lapla= {: e} laplb= {: e} taua= {:e} taub={:e}'.format(d0[i],d1[i],sigma_array[i,0], sigma_array[i,1], sigma_array[i,2], l0[i], l1[i], t0[i], t1[i]))

        return nE, vrho, vsigma, vtau, vlapl


    def xcfun_eval(self, r, functional='PBEX', restricted=False):
        '''Evaluates a functional with the atomic density data using XCFun'''

        d0, d1, g0, g1, t0, t1, l0, l1 = self.get_densities(r)

        if not HAVE_XCFUN:
            raise ImportError('Cannot evaluate functional since xcfun could not be imported.')

        # Initialize functional
        func = xcfun.Functional(functional)

        # Create input
        if restricted:
            density = d0+d1
            densgrad = np.zeros((r.size, 3))
            densgrad[:,0] = g0 + g1
            res = func.eval_energy_n(density, densgrad)

        else:
            density = np.zeros((r.size, 2))
            density[:,0] = d0
            density[:,1] = d1

            densgrad = np.zeros((r.size, 3, 2))
            densgrad[:,0,0] = g0
            densgrad[:,0,1] = g1
            res = func.eval_energy_ab(density, densgrad)

        return res


def static_init(cls):
    if getattr(cls, "static_init", None):
        cls.static_init()
    return cls

@static_init
class AtomData:
    """
    Class encapsulating raw data for all atoms.
    Primarily used to populate Atom class on instantiation.

    No need to give entries if the atom has none of that orbital, e.g. P and D for Li

    In these cases the default None and [0, 0] occupation should be used.
    Achieved by .get() accessing
    """

    # Testing data
    nuclear_charge = {
        'H'  : 1.0,
        'HE' : 2.0,
        'LI' : 3.0,
        'BE' : 4.0,
        'B'  : 5.0,
        'C'  : 6.0,
        'N'  : 7.0,
        'O'  : 8.0,
        'F'  : 9.0,
        'NE' : 10.0,
        'NA' : 11.0,
        'MG' : 12.0,
        'AL' : 13.0,
        'SI' : 14.0,
        'P'  : 15.0,
        'S'  : 16.0,
        'CL' : 17.0,
        'AR' : 18.0,
        'K'  : 19.0,
        'CA' : 20.0,
        'SC' : 21.0,
        'TI' : 22.0,
        'V'  : 23.0,
        'CR' : 24.0,
        'MN' : 25.0,
        'FE' : 26.0,
        'CO' : 27.0,
        'NI' : 28.0,
        'CU' : 29.0,
        'ZN' : 30.0,
        'GA' : 31.0,
        'GE' : 32.0,
        'AS' : 33.0,
        'SE' : 34.0,
        'BR' : 35.0,
        'KR' : 36.0,
        'RB' : 37.0,
        'SR' : 38.0,
        'Y'  : 39.0,
        'ZR' : 40.0,
        'NB' : 41.0,
        'MO' : 42.0,
        'TC' : 43.0,
        'RU' : 44.0,
        'RH' : 45.0,
        'PD' : 46.0,
        'AG' : 47.0,
        'CD' : 48.0,
        'IN' : 49.0,
        'SN' : 50.0,
        'SB' : 51.0,
        'TE' : 52.0,
        'I'  : 53.0,
        'XE' : 54.0,
        'CS' : 55.0,
        'BA' : 56.0,
        'LA' : 57.0,
        'CE' : 58.0,
        'PR' : 59.0,
        'ND' : 60.0,
        'PM' : 61.0,
        'SM' : 62.0,
        'EU' : 63.0,
        'GD' : 64.0,
        'TB' : 65.0,
        'DY' : 66.0,
        'HO' : 67.0,
        'ER' : 68.0,
        'TM' : 69.0,
        'YB' : 70.0,
        'LU' : 71.0,
        'HF' : 72.0,
        'TA' : 73.0,
        'W'  : 74.0,
        'RE' : 75.0,
        'OS' : 76.0,
        'IR' : 77.0,
        'PT' : 78.0,
        'AU' : 79.0,
        'HG' : 80.0,
        'TL' : 81.0,
        'PB' : 82.0,
        'BI' : 83.0,
        'PO' : 84.0,
        'AT' : 85.0,
        'RN' : 86.0,
        'FR' : 87.0,
        'RA' : 88.0,
        'AC' : 89.0,
        'TH' : 90.0,
        'PA' : 91.0,
        'U'  : 92.0,
        'NP' : 93.0,
        'PU' : 94.0,
        'AM' : 95.0,
        'CM' : 96.0,
        'BK' : 97.0,
        'CF' : 98.0,
        'ES' : 99.0,
        'FM' : 100.0,
        'MD' : 101.0,
        'NO' : 102.0,
        'LR' : 103.0
    }

    electron_count = {}
    ke_test = {}

    s_exp = {}
    s_coef = {}
    s_n = {}
    s_occ = {}

    p_exp = {}
    p_coef = {}
    p_n = {}
    p_occ = {}

    d_exp = {}
    d_coef = {}
    d_n = {}
    d_occ = {}

    f_exp = {}
    f_coef = {}
    f_n = {}
    f_occ = {}

    @classmethod
    def add_entry(self, a, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc):
        '''Adds an entry to the database'''

        self.ke_test[a] = Ekin
        for ishell in range(len(ams)):
            if ams[ishell] == 0:
                self.s_exp[a] = xs[ishell]
                self.s_coef[a] = np.transpose(cs[ishell])
                self.s_n[a] = ns[ishell]
                self.s_occ[a] = socc
            elif ams[ishell] == 1:
                self.p_exp[a] = xs[ishell]
                self.p_coef[a] = np.transpose(cs[ishell])
                self.p_n[a] = ns[ishell]
                self.p_occ[a] = pocc
            elif ams[ishell] == 2:
                self.d_exp[a] = xs[ishell]
                self.d_coef[a] = np.transpose(cs[ishell])
                self.d_n[a] = ns[ishell]
                self.d_occ[a] = docc
            elif ams[ishell] == 3:
                self.f_exp[a] = xs[ishell]
                self.f_coef[a] = np.transpose(cs[ishell])
                self.f_n[a] = ns[ishell]
                self.f_occ[a] = focc
            else:
                raise ValueError('Angular momentum too large')

    @classmethod
    def static_init(self):
        '''Initialize the data storage by reading in the tabulated wave functions'''

        # Find where the current file is
        curpath = os.path.dirname(os.path.abspath(__file__))

        # Add neutral atoms
        for a in self.nuclear_charge.keys():
            # Light atoms
            infile='{}/literature_data/k99l/neutral/{}'.format(curpath, a.lower())
            if os.path.isfile(infile):
                self.electron_count[a]=self.nuclear_charge[a]
                Etot, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc = parse(infile)
                self.add_entry(a, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc)
            # Heavy atoms
            infile='{}/literature_data/k00heavy/{}'.format(curpath, a.lower())
            if os.path.isfile(infile):
                self.electron_count[a]=self.nuclear_charge[a]
                Etot, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc = parse(infile)
                self.add_entry(a, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc)

        # Add cations and anions
        neutral_atoms = self.nuclear_charge.copy()
        for a in neutral_atoms.keys():
            infile='{}/literature_data/k99l/cation/{}.cat'.format(curpath, a.lower())
            if os.path.isfile(infile):
                cata="{}+".format(a)
                self.nuclear_charge[cata]=neutral_atoms[a]
                self.electron_count[cata]=neutral_atoms[a]-1
                Etot, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc = parse(infile)
                self.add_entry(cata, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc)

            infile='{}/literature_data/k99l/anion/{}.an'.format(curpath, a.lower())
            if os.path.isfile(infile):
                ana="{}-".format(a)
                self.nuclear_charge[ana]=neutral_atoms[a]
                self.electron_count[ana]=neutral_atoms[a]+1
                Etot, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc = parse(infile)
                self.add_entry(ana, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc)

class GridGenerator:
    """
    Static class encapsulating grid generation methods.

    Public facing method is make_grid(n)

    Then spherical integration can be achieved using
        np.sum(f(r)*4*pi*weight)
    """
    @staticmethod
    def make_grid(n, method='ahlrichsm3', R=1.0, quad='chebyshev2_mod'):
        """
        Generate a radial integration grid containing n points.
            n: desired grid points
            method:
               'ahlrichsm3'    : parameter-free Ahlrichs M3 grid (default)
               'ahlrichsm4'    : Ahlrichs M4 grid with alpha=0.6
               'handy'         : Murray-Handy-Laming quadrature
               'muraknowles'   : Mura-Knowles quadrature
            R: atomic size adjustment parameter, default R=1.0
            quad: quadrature scheme, 'chebyshev2_mod' (default)
        OUT:
            n: number of grid points (note: may be different from that requested)
            r: numpy array of coordinates
            wt: numpy array of weights
        """

        if  method=='ahlrichsm3' or method == 'krack':
            n, r, wt = GridGenerator.radial_ahlrichs(n, alpha=0.0, quad=quad)
        elif method == 'ahlrichsm4':
            n, r, wt = GridGenerator.radial_ahlrichs(n, alpha=0.6, quad=quad)
# The Laguerre quadrature doesn't appear to work for the wanted numbers of quadrature points
#        elif method == 'laguerre':
#            n, r, wt = GridGenerator.radial_laguerre(n)
        elif method == 'handy':
            n, r, wt = GridGenerator.radial_handy(n, quad=quad)
        elif method == 'muraknowles':
            n, r, wt = GridGenerator.radial_muraknowles(n, quad=quad)
        elif method == 'becke':
            n, r, wt = GridGenerator.radial_becke(n, quad=quad)
        else:
            raise ValueError('Unknown grid {}'.format(method))

        # Eliminate any zero weights
        r = r[wt!=0.0]
        wt = wt[wt!=0.0]
        n = len(r)

        # Atomic scaling adjustment
        r = r*R
        wt = wt*R**3

        return n, r, wt

    @staticmethod
    def chebyshev1(n):
        """Gauss-Chebyshev quadrature of the first kind for
        calculating

        .. math:: \\int_{-1}^{1} f(x) dx = \\sum_i w_i f(x_i).

        Returns n, x, w.

        See Abramowitz-Stegun p. 889

        """

        # Index vector
        ivec = np.asarray([ n+1-i for i in range(1,n+1)])
        # Angles for sine and cosine
        angles = (2*ivec-1)*pi/(2*n)

        # Integration nodes
        x = np.cos(angles)

        # Integration weights
        w = pi/n*np.sqrt(1-np.square(x))

        return n, x, w

    @staticmethod
    def chebyshev2(n):
        """Gauss-Chebyshev quadrature of the second kind for
        calculating

        .. math:: \\int_{-1}^{1} f(x) dx = \\sum_i w_i f(x_i).

        Returns n, x, w.

        See Abramowitz-Stegun p. 889

        """

        # Index vector
        ivec = np.asarray([ n+1-i for i in range(1,n+1)])
        # Angles for sine and cosine
        angles = ivec*pi/(n+1)

        # Integration nodes
        x = np.cos(angles)

        # Integration weights
        w = pi/(n+1)*np.sqrt(1-np.square(x))

        return n, x, w

    @staticmethod
    def chebyshev2_modified(n):
        """Modified Gauss-Chebyshev quadrature of the second kind for
        calculating

        .. math:: \\int_{-1}^{1} f(x) dx = \\sum_i w_i f(x_i).

        Returns n, x, w.

        See eqns (31)-(33) in J. M. Pérez‐Jordá, A. D. Becke, and
        E. San‐Fabián, Automatic numerical integration techniques for
        polyatomic molecules, J. Chem. Phys. 100, 6520 (1994);
        doi:10.1063/1.467061

        """

        # 1/(n+1)
        oonpp=1.0/(n+1.0)

        # Index vector
        ivec = np.asarray([ i for i in range(1,n+1)])
        # Angles for sine and cosine
        angles = ivec*pi*oonpp
        # Sines and cosines
        sines = np.sin(angles)
        cosines = np.cos(angles)
        # Sine squared
        sinesq = np.power(sines,2)
        sinecos = np.multiply(sines,cosines)

        # Integration weights
        w = 16.0/3.0/(n+1.0) * np.power(sinesq,2)
        # Integration nodes
        x = 1.0 - 2.0*ivec*oonpp + 2/pi*np.multiply(1.0 + 2.0/3.0*sinesq, sinecos)

        return n, x, w

    @staticmethod
    def chebyshev3(n):
        """Gauss-Chebyshev quadrature of the third kind for
        calculating

        .. math:: \\int_{-1}^{1} f(x) dx = \\sum_i w_i f(x_i).

        Returns n, x, w.

        See Abramowitz-Stegun p. 889

        """

        # Index vector
        ivec = np.asarray([ n+1-i for i in range(1,n+1)])
        # Angles for sine and cosine
        angles = 0.5*(2*ivec-1)*pi/(2*n+1)

        # Original integration nodes and weights
        x = np.square(np.cos(angles))
        w = 2*pi/(2*n+1)*x*np.sqrt(np.divide(1-x,x))

        # Transform from [0,1] to [-1,1]
        x = 2*x-1
        w = 2*w

        return n, x, w

    @staticmethod
    def quadrature(n, quad='chebyshev2_mod'):
        """Quadrature rule for calculating

        .. math:: \\int_{-1}^{1} f(x) dx = \\sum_i w_i f(x_i).

        Returns n, x, w.

        Input:
           n: number of quadrature points
           quad: quadrature rule

        """
        if quad == 'chebyshev1':
            n, xi, wi = GridGenerator.chebyshev1(n)
        elif quad == 'chebyshev2':
            n, xi, wi = GridGenerator.chebyshev2(n)
        elif quad == 'chebyshev2_mod':
            n, xi, wi = GridGenerator.chebyshev2_modified(n)
        elif quad == 'chebyshev3':
            n, xi, wi = GridGenerator.chebyshev3(n)
        elif quad == 'legendre':
            import scipy
            xi, wi = scipy.special.roots_legendre(n)
        elif quad == 'trapezoidal':
            # Form trapezoidal rule. This agrees with the description
            # in eqns (18)-(19) in P. M. W. Gill, S.-H. Chien, Radial
            # Quadrature for Multiexponential Integrands,
            # J. Comput. Chem. 24, 732 (2003). doi:10.1002/jcc.10211

            # Starting with the rule in [0, 1]: x_i = i/(N+1), w_i =
            # 1/(N+1) gives us
            xi = np.asarray([1.0 - 2*j/(n+1) for j in range(1,n+1)])
            wi = 2.0/(n+1.0)

        else:
            raise ValueError('Unknown quadrature rule\n')

        return n, xi, wi

    @staticmethod
    def quadrature_halfinterval(n, quad):
        """Quadrature rule of the second kind for
        calculating

        .. math:: \\int_{0}^{1} f(x) dx = \\sum_i w_i f(x_i).

        Returns n, x, w.

        See eqns (31)-(33) in J. M. Pérez‐Jordá, A. D. Becke, and
        E. San‐Fabián, Automatic numerical integration techniques for
        polyatomic molecules, J. Chem. Phys. 100, 6520 (1994);
        doi:10.1063/1.467061

        """
        n, xc, wc = GridGenerator.quadrature(n, quad)
        # Translate weights from [-1, 1] to [0, 1]
        xi = 0.5*(xc+1.0)
        wi = 0.5*wc
        return n, xi, wi

    @staticmethod
    def radial_ahlrichs(n, alpha=0.6, quad='chebyshev2_mod'):
        """Treutler-Ahlrichs M4 quadrature for calculating

        .. math:: \\int_{0}^{\\infty} r^2 f(r) dr = \\sum_i w_i r_i^2 f(r_i).

        Returns n, x, w.

        See O. Treutler and R. Ahlrichs, Efficient molecular numerical
        integration schemes, J. Chem. Phys. 102, 346 (1995), eqn (19).

        The radial transformation is given by

        r = 1/log(2) (1+x)^\alpha log(2/(1-x))

        where x are quadrature weights for the [-1, 1] interval, which
        are obtained in this routine from the Chebyshev rule which is
        described to have good performance in

        M. Krack and A. M. Köster, An adaptive numerical integrator
        for molecular integrals, J. Chem. Phys. 108, 3226 (1998);
        doi:10.1063/1.475719

        """

        n, x, w = GridGenerator.quadrature(n, quad)

        if alpha == 0.0:
            r = 1.0/log(2.0)*np.log(2.0/(1.0-x))
            dr = 1.0/(log(2.0)*(1-x))
        else:
            r = 1.0/log(2.0)*(1+x)**alpha*np.log(2.0/(1.0-x))
            dr = 1.0/log(2.0)*(alpha*(1+x)**(alpha-1.0)*np.log(2.0/(1.0-x)) + (1+x)**alpha/(1-x))

        wr = w*dr*r**2

        # Get radii in increasing order
        return n, np.flip(r), np.flip(wr)

    @staticmethod
    def radial_laguerre(n):
        """Gauss-Laguerre quadrature of the second kind for
        calculating

        .. math:: \\int_{0}^{\\infty} x^2 f(x) dx = \\sum_i w_i f(x_i).

        Returns n, x, w.

        See eqns (8)-(10) in P. M. W. Gill, S.-H. Chien, Radial
        Quadrature for Multiexponential Integrands,
        J. Comput. Chem. 24, 732 (2003). doi:10.1002/jcc.10211

        """

        if not HAVE_SCIPY:
            raise RuntimeError('Need scipy for Laguerre weights')
        # Get the roots of the Laguerre polynomials
        from scipy.special import roots_laguerre, eval_laguerre
        [xi, wi] = roots_laguerre(n)

        # Integration weights
        w = np.power(xi,3)*np.exp(xi)/((n+1) * eval_laguerre(n+1, xi))**2
        # Integration nodes
        x = xi

        return n, x, w

    @staticmethod
    def radial_handy(n, quad='chebyshev2_mod'):
        """Handy grid for calculating

        .. math:: \\int_{0}^{\\infty} x^2 f(x) dx = \\sum_i w_i f(x_i).

        Described in C. W. Murray, N. C. Handy, and G. J. Laming,
        Quadrature schemes for integrals of density functional theory,
        Mol. Phys. 78, 997 (1993). doi:10.1080/00268979300100651

        Returns n, x, w.

        See eqns (18)-(20) in P. M. W. Gill, S.-H. Chien, Radial
        Quadrature for Multiexponential Integrands,
        J. Comput. Chem. 24, 732 (2003). doi:10.1002/jcc.10211 Note
        that this routine uses Chebyshev quadrature by default instead
        of the trapezoidal quadrature of the above paper, depending on
        the quad argument.

        """

        # Get the quadrature rule in [0, 1]
        n, xi, wi = GridGenerator.quadrature_halfinterval(n, quad)
        # Integration nodes
        x = np.divide(np.power(xi, 2), np.power(1.0-xi,2))
        # Integration weights
        w = np.multiply(wi, np.divide(2*np.power(xi,5), np.power(1.0-xi,7)))

        return n, x, w

    @staticmethod
    def radial_muraknowles(n, m=3, quad='chebyshev2_mod'):
        """Mura-Knowles grid for calculating

        .. math:: \\int_{0}^{\\infty} x^2 f(x) dx = \\sum_i w_i f(x_i).

        Described in M. E. Mura, P. J. Knowles, Improved radial grids
        for quadrature in molecular density-functional calculations,
        J. Chem. Phys. 104, 9848 (1996). doi:10.1063/1.471749

        By default, generates the Log3 grid corresponding to m=3; the
        value of m can be given as an optional input parameter.

        Returns n, x, w.

        See eqns (28)-(30) in P. M. W. Gill, S.-H. Chien, Radial
        Quadrature for Multiexponential Integrands,
        J. Comput. Chem. 24, 732 (2003). doi:10.1002/jcc.10211.

        However, that paper seems to be wrong in the quadrature rule;
        Molpro's manual states that Gauss quadrature is used in the x
        space instead of the trapezoidal rule, so this routine uses
        Chebyshev quadrature as default.
        """

        # Get the quadrature rule in [0, 1]
        n, xi, wi = GridGenerator.quadrature_halfinterval(n, quad)
        # Form the quadrature rule
        x = -np.log(1-xi**m)
        w = np.multiply(wi, np.divide(m*(xi**(m-1))*np.log(1-(xi**m))**2, (1.0-xi**m)))

        return n, x, w

    @staticmethod
    def radial_becke(n, quad='chebyshev2'):
        """Becke quadrature for calculating

        .. math:: \\int_{0}^{\\infty} r^2 f(r) dr = \\sum_i w_i r_i^2 f(r_i).

        Returns n, x, w.

        See A. Becke, J. Chem. Phys. 88, 2547 (1988), eqn (25).

        The radial transformation is given by

        r = (1+x)/(1-x)

        where x are quadrature weights for the [-1, 1] interval.
        """

        n, x, w = GridGenerator.quadrature(n, quad)

        r = (1+x)/(1-x)
        dr = 2/(x-1)**2

        wr = w*dr*r**2

        # Get radii in increasing order
        return n, np.flip(r), np.flip(wr)

"""
Defines the "color" of each element, same as those used by Jmol, roughly follows CPK coloring
"""
COLOR_DICT = {
    'H':  np.array([1.000, 1.000, 1.000]),
    'HE': np.array([0.851, 1.000, 1.000]),
    'LI': np.array([0.800, 0.502, 1.000]),
    'BE': np.array([0.761, 1.000, 0.000]),
    'B':  np.array([1.000, 0.710, 0.710]),
    'C':  np.array([0.565, 0.565, 0.565]),
    'N':  np.array([0.188, 0.314, 0.973]),
    'O':  np.array([1.000, 0.051, 0.051]),
    'F':  np.array([0.565, 0.878, 0.314]),
    'NE': np.array([0.702, 0.890, 0.961]),
    'NA': np.array([0.671, 0.361, 0.949]),
    'MG': np.array([0.541, 1.000, 0.000]),
    'AL': np.array([0.749, 0.651, 0.651]),
    'SI': np.array([0.941, 0.784, 0.627]),
    'P':  np.array([1.000, 0.502, 0.000]),
    'S':  np.array([1.000, 1.000, 0.188]),
    'CL': np.array([0.122, 0.941, 0.122]),
    'AR': np.array([0.502, 0.820, 0.890]),
    'K':  np.array([0.561, 0.251, 0.831]),
    'CA': np.array([0.239, 1.000, 0.000]),
    'SC': np.array([0.902, 0.902, 0.902]),
    'TI': np.array([0.749, 0.761, 0.780]),
    'V':  np.array([0.651, 0.651, 0.671]),
    'CR': np.array([0.541, 0.600, 0.780]),
    'MN': np.array([0.612, 0.478, 0.780]),
    'FE': np.array([0.878, 0.400, 0.200]),
    'CO': np.array([0.941, 0.565, 0.627]),
    'NI': np.array([0.314, 0.816, 0.314]),
    'CU': np.array([0.784, 0.502, 0.200]),
    'ZN': np.array([0.490, 0.502, 0.690]),
    'GA': np.array([0.761, 0.561, 0.561]),
    'GE': np.array([0.400, 0.561, 0.561]),
    'AS': np.array([0.741, 0.502, 0.890]),
    'SE': np.array([1.000, 0.631, 0.000]),
    'BR': np.array([0.651, 0.161, 0.161]),
    'KR': np.array([0.361, 0.722, 0.820]),
    'RB': np.array([0.439, 0.180, 0.690]),
    'SR': np.array([0.000, 1.000, 0.000]),
    'Y':  np.array([0.580, 1.000, 1.000]),
    'ZR': np.array([0.580, 0.878, 0.878]),
    'NB': np.array([0.451, 0.761, 0.788]),
    'MO': np.array([0.329, 0.710, 0.710]),
    'TC': np.array([0.231, 0.620, 0.620]),
    'RU': np.array([0.141, 0.561, 0.561]),
    'RH': np.array([0.039, 0.490, 0.549]),
    'PD': np.array([0.000, 0.412, 0.522]),
    'AG': np.array([0.753, 0.753, 0.753]),
    'CD': np.array([1.000, 0.851, 0.561]),
    'IN': np.array([0.651, 0.459, 0.451]),
    'SN': np.array([0.400, 0.502, 0.502]),
    'SB': np.array([0.620, 0.388, 0.710]),
    'TE': np.array([0.831, 0.478, 0.000]),
    'I':  np.array([0.580, 0.000, 0.580]),
    'XE': np.array([0.259, 0.620, 0.690]),
    'CS': np.array([0.341, 0.090, 0.561]),
    'BA': np.array([0.000, 0.788, 0.000]),
    'LA': np.array([0.439, 0.831, 1.000]),
    'CE': np.array([1.000, 1.000, 0.780]),
    'PR': np.array([0.851, 1.000, 0.780]),
    'ND': np.array([0.780, 1.000, 0.780]),
    'PM': np.array([0.639, 1.000, 0.780]),
    'SM': np.array([0.561, 1.000, 0.780]),
    'EU': np.array([0.380, 1.000, 0.780]),
    'GD': np.array([0.271, 1.000, 0.780]),
    'TB': np.array([0.188, 1.000, 0.780]),
    'DY': np.array([0.122, 1.000, 0.780]),
    'HO': np.array([0.000, 1.000, 0.612]),
    'ER': np.array([0.000, 0.902, 0.459]),
    'TM': np.array([0.000, 0.831, 0.322]),
    'YB': np.array([0.000, 0.749, 0.220]),
    'LU': np.array([0.000, 0.671, 0.141]),
    'HF': np.array([0.302, 0.761, 1.000]),
    'TA': np.array([0.302, 0.651, 1.000]),
    'W':  np.array([0.129, 0.580, 0.839]),
    'RE': np.array([0.149, 0.490, 0.671]),
    'OS': np.array([0.149, 0.400, 0.588]),
    'IR': np.array([0.090, 0.329, 0.529]),
    'PT': np.array([0.816, 0.816, 0.878]),
    'AU': np.array([1.000, 0.820, 0.137]),
    'HG': np.array([0.722, 0.722, 0.816]),
    'TL': np.array([0.651, 0.329, 0.302]),
    'PB': np.array([0.341, 0.349, 0.380]),
    'BI': np.array([0.620, 0.310, 0.710]),
    'PO': np.array([0.671, 0.361, 0.000]),
    'AT': np.array([0.459, 0.310, 0.271]),
    'RN': np.array([0.259, 0.510, 0.588]),
    'FR': np.array([0.259, 0.000, 0.400]),
    'RA': np.array([0.000, 0.490, 0.000]),
    'AC': np.array([0.439, 0.671, 0.980]),
    'TH': np.array([0.000, 0.729, 1.000]),
    'PA': np.array([0.000, 0.631, 1.000]),
    'U':  np.array([0.000, 0.561, 1.000]),
    'NP': np.array([0.000, 0.502, 1.000]),
    'PU': np.array([0.000, 0.420, 1.000]),
    'AM': np.array([0.329, 0.361, 0.949]),
    'CM': np.array([0.471, 0.361, 0.890]),
    'BK': np.array([0.541, 0.310, 0.890]),
    'CF': np.array([0.631, 0.212, 0.831]),
    'ES': np.array([0.702, 0.122, 0.831]),
    'FM': np.array([0.702, 0.122, 0.729]),
    'MD': np.array([0.702, 0.051, 0.651]),
    'NO': np.array([0.741, 0.051, 0.529]),
    'LR': np.array([0.780, 0.000, 0.400]),
    'RF': np.array([0.800, 0.000, 0.349]),
    'DB': np.array([0.820, 0.000, 0.310]),
    'SG': np.array([0.851, 0.000, 0.271]),
    'BH': np.array([0.878, 0.000, 0.220]),
    'HS': np.array([0.902, 0.000, 0.180]),
    'MT': np.array([0.922, 0.000, 0.149]),
}

def get_colors_for_elements(elist):
    """
    Takes a list of element labels and returns the color.
    elist can be list of strings or Atom objects
    """
    if not isinstance(elist, list) and not isinstance(elist, tuple):
        elist = [elist]
    try:
        return [COLOR_DICT[e.element.upper()] if isinstance(e, Atom) else COLOR_DICT[e.upper()] for e in elist]
    except AttributeError:
        raise ValueError("Input should be list of strings or Atom objects")


if __name__ == "__main__":
    test_densities()
