import numpy as np
from math import pi, log
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

    actual, r, wt = GridGenerator.make_grid(400)
    grid = 4*pi*r**2*wt

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
    if np.allclose(fdiff, g0[1:-1], atol=1e-1):  # finite difference is not perfect, so lenient tollerance
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

    # Static principal quantum number factors
    FACTORS = np.array([2.0, 24.0, 720.0, 40320.0, 3628800.0])**(-0.5)


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
        if not np.isscalar(r):
            r = np.array((r))

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

    def G(self, n, e, r):
        """
        Evaluates slater orbitals.
            n: principal quantum number
            e: exponent
            r: space coordinate. (distance from nucleus)
        """
        n_facs = self.FACTORS[n - 1]
        try:
            c = n_facs*(2.0*e)**(n + 0.5)
        except ValueError:
            print("Exponents and principal number factors are different shapes.")
            print("Did you typo a ',' for a decimal point? e.g. '1,23456' for '1.23456'")
            raise ValueError("exponent or principal number error")
        rn = np.power.outer(r, (n - 1))
        es = np.einsum('j,ij->ji', c, rn)
        pw = np.exp(-np.outer(e, r))
        return es*pw

    def DG(self, n, e, r, f):
        """
        Evaluates first derivative of slater orbitals.
            n: principal quantum number
            e: exponent
            r: space coordinate. (distance from nucleus)
            f: Undifferentiated function
        """

        pre = np.add(-e[:, None], np.divide.outer((n - 1), r))
        return pre*f

    def DDG(self, n, e, r, f):
        """
        Evaluates second derivative of slater orbitals.
            n: principal quantum number
            e: exponent
            r: space coordinate. (distance from nucleus)
            f: Undifferentiated function
        """
        pre = np.add(-e[:, None], np.divide.outer((n - 1), r))**2
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

    def libxc_eval(self, r, functional='gga_x_pbe', restricted=False, threshold=None):
        '''Evaluates a functional with the atomic density data using libxc'''

        d0, d1, g0, g1, t0, t1, l0, l1 = self.get_densities(r)

        if not HAVE_LIBXC:
            raise ImportError('Cannot evaluate functional sinc pylibxc could not be imported.')

        func = pylibxc.LibXCFunctional(functional, "unpolarized" if restricted else "polarized")

        # Did we get a threshold?
        if threshold is not None:
            func.set_dens_threshold(threshold)

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

        return nE, vrho, vsigma, vtau, vlapl


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

        # Add neutral atoms
        for a in self.nuclear_charge.keys():
            # Light atoms
            infile='literature_data/k99l/neutral/{}'.format(a.lower())
            if os.path.isfile(infile):
                self.electron_count[a]=self.nuclear_charge[a]
                Etot, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc = parse(infile)
                self.add_entry(a, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc)
            # Heavy atoms
            infile='literature_data/k00heavy/{}'.format(a.lower())
            if os.path.isfile(infile):
                self.electron_count[a]=self.nuclear_charge[a]
                Etot, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc = parse(infile)
                self.add_entry(a, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc)

        # Add cations and anions
        neutral_atoms = self.nuclear_charge.copy()
        for a in neutral_atoms.keys():
            infile='literature_data/k99l/cation/{}.cat'.format(a.lower())
            if os.path.isfile(infile):
                cata="{}+".format(a)
                self.nuclear_charge[cata]=neutral_atoms[a]
                self.electron_count[cata]=neutral_atoms[a]-1
                Etot, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc = parse(infile)
                self.add_entry(cata, Ekin, ams, ns, xs, cs, socc, pocc, docc, focc)

            infile='literature_data/k99l/anion/{}.an'.format(a.lower())
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

    Then spherical integration can be achieved using,
        np.sum(f(r)*4*pi*weight*r**2)
    """
    @staticmethod
    def make_grid(n, gl=False):
        """
        Generate a radial integration grid containing n points.
            n: desired grid points
           gl: two-part Gauss-Legendre instead of one-part modified Chebyshev
        OUT:
            n: number of grid points (note: may be different from that requested)
            r: numpy array of coordinates
            wt: numpy array of weights
        """

        if gl:
            low = 0.0 # Lower Range
            high = 1.0
            p = 0.5

            # The method here uses 2*n points so halve it
            n, r, wt = GridGenerator.gaussp(low, high, n//2)
            r = np.concatenate((r, np.zeros((n))))
            wt = np.concatenate((wt, np.zeros((n))))
            for i in range(n):
                r[2*n-(i+1)] = (1.0/r[i])**2
                wt[2*n-(i+1)] = (wt[i]/p)*r[2*n - (i+1)]**1.5
        else:
            n, r, wt = GridGenerator.radial_chebyshev(n)

        return n, r, wt

    @staticmethod
    def chebyshev(n):
        """Modified Gauss-Chebyshev quadrature of the second kind for
        calculating \int_{-1}^{1} f(x) dx = \sum_i w_i f(x_i).

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
    def radial_chebyshev(n):
        """Gauss-Chebyshev quadrature for calculating \int_{0}^{\infty} r^2
        f(r) dr = \sum_i w_i r_i^2 f(r_i).

        Returns n, x, w.

        Uses chebyshev() to get the quadrature weights for the [-1, 1]
        interval, and then uses a logarithmic transformation

        r = 1/log(2) log(2/(1-x)) <=> x = 1 - 2^(1-r)

        to get the corresponding radial rule.

        See eqn (13) in M. Krack and A. M. Köster, An adaptive
        numerical integrator for molecular integrals,
        J. Chem. Phys. 108, 3226 (1998); doi:10.1063/1.475719

        """

        n, x, w = GridGenerator.chebyshev(n)
        r = 1.0/log(2.0)*np.log(2.0/(1.0-x))
        wr = np.multiply(w, 1.0/(log(2.0)*(1.0-x)))

        return n, r, wr

    @staticmethod
    def gaussp(y1, y2, n):
        """
        Generates an integration grid.
        First it will try the pre-stored meshes, failing that it defines
        a mesh by iteration of Chebyshev points.
        points/weights are defined for the interval [-1, 1]

        The integration can then be evaluated as,
            np.sum(f(r)4*pi*weight*r**2)
        Using numpy vectorisation.
        """

        # First check for trivial or stupid requests
        if n <= 0:
            raise ValueError("Zero (or less) grid points is stupid. Stop it.")
        if n == 1:
            r = np.array([0.5*(y2 + y1)])
            wt = np.array([y2 - y1])
            return r, wt
        N_pi = 3.14159265358979323844  # Fortran uses stupid pi because of course it does
        EPS = 1e-14  # Desired accuracy
        n_sav = -1

        if n != n_sav:
            n_sav = n
            m = n
            m, r, wt = GridGenerator.gausspp(m)
            m = 0

            if m != n:
                m = int((n+1)/2)  # Care, integer division
                x = np.zeros((2*m))  # Working r, not returned
                w = np.zeros((2*m))  # Working wt, not returned
                r = np.zeros((2*m))
                wt = np.zeros((2*m))
                for i in range(m):
                    r[i] = N_pi*(i+0.75)/(n+0.5)
                r = np.cos(r)

                for i in range(m):
                    z = r[i]
                    z1 = 1e20  # Arbitrary large number to ensure at least 1 loop
                    while abs(z-z1) > EPS:
                        p1 = 1.0
                        p2 = 0.0
                        for j in range(n):
                            p3 = p2
                            p2 = p1
                            p1 = ((2*(j + 1) - 1)*z*p2 - j*p3)/(j + 1)
                        pp = n*(z*p1 - p2)/(z*z - 1.0)
                        z1 = z
                        z = z1 - p1/pp
                    x[i] = -z
                    x[n - (i + 1)] = z
                    w[i] = 2.0/((1.0 - z*z)*pp*pp)
                    w[n - (i + 1)] = w[i]

        for i in range(n):
            fact = 0.5*(y2-y1)
            r[i] = y1 + fact*(x[i] + 1.0)
            wt[i] = fact*w[i]

        return n, r, wt

    @staticmethod
    def gausspp(npt):
        """
        Sets up the integration points and weights for an
        npt point Gauss-Legendre integration in the interval (-1, 1)

        The integration can then be evaluated using numpy vectorisation as,
            np.sum(f(r)4*pi*weight*r**2)

        The Fortran source tells me:
            FOR NPT=1-14:   VALUES FROM Z. KOPAL, NUMERICAL ANALYSIS, 1961, A4
            NPT=16-64:      FROM ALCHEMY GAUSS ROUTINE
            NPT=96:         FROM DECK OF R. NERF JUNE 1973.
        """
        if npt <= 0:
            raise ValueError("Can't generate grid for <= 0 points")
            return
        if npt == 1:
            xpt = np.array([0.0])
            wht = np.array([2.0])
            return xpt, wht

        # Each mesh is stored as a section of a big array.
        # These store its number and start index is here
        mesh_npts = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,20,24,28,32,40,48,64,96]

        # First, look to see if the mesh is stored.
        # If not we take the largest number that is lower than that stored.
        for i in range(len(mesh_npts)):
            mesh_idx = i
            if mesh_npts[i] >= npt:
                break
        npt = mesh_npts[mesh_idx]
        n2 = int((npt+1)/2.0) # Care: Integer division!
        iof = npt

        # The stored grid parameters are accessed as a dict of arrays.
        x = {
            2 : [0.577350269189626e0],
            3 : [0.774596669241483e0, 0.0e0],
            4 : [0.861136311594053e0, 0.339981043584856e0],
            5 : [0.906179845938664e0, 0.538469310105683e0, 0.0e0],
            6 : [0.932469514203152e0, 0.661209386466265e0, 0.238619186083197e0],
            7 : [0.949107912342759e0, 0.741531185599394e0, 0.405845151377397e0, 0.0e0],
            8 : [0.960289856497536e0, 0.796666477413627e0, 0.525532409916329e0, 0.183434642495650e0],
            9 : [0.968160239507626e0, 0.836031107326636e0, 0.613371432700590e0, 0.324253423403809e0,
                0.0e0],
            10 : [0.973906528517172e0, 0.865063366688985e0, 0.679409568299024e0, 0.433395394129247e0,
                0.148874338981631e0],
            11 : [0.978228658146057e0, 0.887062599768095e0, 0.730152005574049e0, 0.519096129206812e0,
                0.269543155952345e0, 0.0e0],
            12 : [0.981560634246719e0, 0.904117256370475e0, 0.769902674194305e0, 0.587317954286617e0,
                0.367831498998180e0, 0.125233408511469e0],
            13 : [0.984183054718588e0, 0.917598399222978e0, 0.801578090733310e0, 0.642349339440340e0,
                0.448492751036447e0, 0.230458315955135e0, 0.0e0],
            14 : [0.986283808696812e0, 0.928434883663574e0, 0.827201315069765e0, 0.687292904811685e0,
                0.515248636358154e0, 0.319112368927890e0, 0.108054948707344e0],
            16 : [0.989400934991650e0, 0.944575023073232e0, 0.865631202387832e0, 0.755404408355003e0,
                0.617876244402644e0, 0.458016777657227e0, 0.281603550779259e0, 0.950125098376369e-1],
            20 : [0.993128599185095e0, 0.963971927277914e0, 0.912234428251326e0, 0.839116971822219e0,
                0.746331906460151e0, 0.636053680726515e0, 0.510867001950827e0, 0.373706088715419e0,
                0.227785851141645e0, 0.765265211334969e-1],
            24 : [0.995187219997021e0, 0.974728555971309e0, 0.938274552002733e0, 0.886415527004401e0,
                0.820001985973903e0, 0.740124191578554e0, 0.648093651936975e0, 0.545421471388839e0,
                0.433793507626045e0, 0.315042679696163e0, 0.191118867473616e0, 0.640568928626059e-1],
            28 : [0.996442497573954e0, 0.981303165370873e0, 0.954259280628938e0, 0.915633026392132e0,
                0.865892522574395e0, 0.805641370917179e0, 0.735610878013632e0, 0.656651094038865e0,
                0.569720471811402e0, 0.475874224955118e0, 0.376251516089079e0, 0.272061627635178e0,
                0.164569282133381e0, 0.550792898840340e-1],
            32 : [0.997263861849481e0, 0.985611511545268e0, 0.964762255587506e0, 0.934906075937740e0,
                0.896321155766052e0, 0.849367613732570e0, 0.794483795967942e0, 0.732182118740290e0,
                0.663044266930215e0, 0.587715757240762e0, 0.506899908932229e0, 0.421351276130635e0,
                0.331868602282128e0, 0.239287362252137e0, 0.144471961582796e0, 0.483076656877380e-1],
            40 : [0.998237709710559e0, 0.990726238699457e0, 0.977259949983774e0, 0.957916819213792e0,
                0.932812808278676e0, 0.902098806968874e0, 0.865959503212259e0, 0.824612230833312e0,
                0.778305651426519e0, 0.727318255189927e0, 0.671956684614179e0, 0.612553889667980e0,
                0.549467125095128e0, 0.483075801686179e0, 0.413779204371605e0, 0.341994090825758e0,
                0.268152185007254e0, 0.192697580701371e0, 0.116084070675255e0, 0.387724175060510e-1],
            48 : [0.998771007252426e0, 0.993530172266351e0, 0.984124583722827e0, 0.970591592546247e0,
                0.952987703160431e0, 0.931386690706554e0, 0.905879136715570e0, 0.876572020274248e0,
                0.843588261624393e0, 0.807066204029443e0, 0.767159032515740e0, 0.724034130923815e0,
                0.677872379632664e0, 0.628867396776514e0, 0.577224726083973e0, 0.523160974722233e0,
                0.466902904750958e0, 0.408686481990717e0, 0.348755886292161e0, 0.287362487355455e0,
                0.224763790394689e0, 0.161222356068892e0, 0.970046992094629e-1, 0.323801709628690e-1],
            64 : [0.999305041735772e0, 0.996340116771955e0, 0.991013371476744e0, 0.983336253884626e0,
                0.973326827789911e0, 0.961008799652054e0, 0.946411374858403e0, 0.929569172131939e0,
                0.910522137078503e0, 0.889315445995114e0, 0.865999398154093e0, 0.840629296252580e0,
                0.813265315122797e0, 0.783972358943341e0, 0.752819907260532e0, 0.719881850171611e0,
                0.685236313054233e0, 0.648965471254657e0, 0.611155355172393e0, 0.571895646202634e0,
                0.531279464019894e0, 0.489403145707053e0, 0.446366017253464e0, 0.402270157963992e0,
                0.357220158337668e0, 0.311322871990211e0, 0.264687162208767e0, 0.217423643740007e0,
                0.169644420423993e0, 0.121462819296120e0, 0.729931217877989e-1, 0.243502926634240e-1],
            96 : [0.999689503883230e0, 0.998364375863181e0, 0.995981842987209e0, 0.992543900323762e0,
                0.988054126329623e0, 0.982517263563014e0, 0.975939174585136e0, 0.968326828463264e0,
                0.959688291448742e0, 0.950032717784437e0, 0.939370339752755e0, 0.927712456722308e0,
                0.915071423120898e0, 0.901460635315852e0, 0.886894517402420e0, 0.871388505909296e0,
                0.854959033434601e0, 0.837623511228187e0, 0.819400310737931e0, 0.800308744139140e0,
                0.780369043867433e0, 0.759602341176647e0, 0.738030643744400e0, 0.715676812348967e0,
                0.692564536642171e0, 0.668718310043916e0, 0.644163403784967e0, 0.618925840125468e0,
                0.593032364777572e0, 0.566510418561397e0, 0.539388108324357e0, 0.511694177154667e0,
                0.483457973920596e0, 0.454709422167743e0, 0.425478988407300e0, 0.395797649828908e0,
                0.365696861472313e0, 0.335208522892625e0, 0.304364944354496e0, 0.273198812591049e0,
                0.241743156163840e0, 0.210031310460567e0, 0.178096882367618e0, 0.145973714654896e0,
                0.113695850110665e0, 0.812974954644249e-1, 0.488129851360490e-1, 0.162767448496020e-1]
        }
        wt = {
            2 : [0.999999999999999e0],
            3 : [0.555555555555556e0, 0.888888888888889e0],
            4 : [0.347854845137454e0, 0.652145154862546e0],
            5 : [0.236926885056189e0, 0.478628670499366e0, 0.568888888888889e0],
            6 : [0.171324492379170e0, 0.360761573048139e0, 0.467913934572691e0],
            7 : [0.129484966168870e0, 0.279705391489277e0, 0.381830050505119e0, 0.417959183673469e0],
            8 : [0.101228536290376e0, 0.222381034453374e0, 0.313706645877887e0, 0.362683783378362e0],
            9 : [0.812743883615739e-1, 0.180648160694857e0, 0.260610696402935e0, 0.312347077040003e0,
                0.330239355001260e0],
            10 : [0.666713443086879e-1, 0.149451349150581e0, 0.219086362515982e0, 0.269266719309996e0,
                0.295524224714753e0],
            11 : [0.556685671161740e-1, 0.125580369464905e0, 0.186290210927734e0, 0.233193764591990e0,
                0.262804544510247e0, 0.272925086777901e0],
            12 : [0.471753363865120e-1, 0.106939325995318e0, 0.160078328543346e0, 0.203167426723066e0,
                0.233492536538355e0, 0.249147045813403e0],
            13 : [0.404840047653160e-1, 0.921214998377279e-1, 0.138873510219787e0, 0.178145980761946e0,
                0.207816047536889e0, 0.226283180262897e0, 0.232551553230874e0],
            14 : [0.351194603317520e-1, 0.801580871597599e-1, 0.121518570687903e0, 0.157203167158194e0,
                0.185538397477938e0, 0.205198463721296e0, 0.215263853463158e0],
            16 : [0.271524594117540e-1, 0.622535239386480e-1, 0.951585116824929e-1, 0.124628971255534e0,
                0.149595988816577e0, 0.169156519395002e0, 0.182603415044923e0, 0.189450610455068e0],
            20 : [0.176140071391520e-1, 0.406014298003870e-1, 0.626720483341089e-1, 0.832767415767049e-1,
                0.101930119817240e0, 0.118194531961518e0, 0.131688638449177e0, 0.142096109318382e0,
                0.149172986472604e0, 0.152753387130726e0],
            24 : [0.123412297999870e-1, 0.285313886289340e-1, 0.442774388174200e-1, 0.592985849154370e-1,
                0.733464814110799e-1, 0.861901615319529e-1, 0.976186521041139e-1, 0.107444270115966e0,
                0.115505668053726e0, 0.121670472927803e0, 0.125837456346828e0, 0.127938195346752e0],
            28 : [0.912428259309400e-2, 0.211321125927710e-1, 0.329014277823040e-1, 0.442729347590040e-1,
                0.551073456757170e-1, 0.652729239669989e-1, 0.746462142345689e-1, 0.831134172289009e-1,
                0.905717443930329e-1, 0.969306579979299e-1, 0.102112967578061e0, 0.106055765922846e0,
                0.108711192258294e0, 0.110047013016475e0],
            32 : [0.701861000947000e-2, 0.162743947309060e-1, 0.253920653092620e-1, 0.342738629130210e-1,
                0.428358980222270e-1, 0.509980592623760e-1, 0.586840934785350e-1, 0.658222227763619e-1,
                0.723457941088479e-1, 0.781938957870699e-1, 0.833119242269469e-1, 0.876520930044039e-1,
                0.911738786957639e-1, 0.938443990808039e-1, 0.956387200792749e-1, 0.965400885147279e-1],
            40 : [0.452127709853300e-2, 0.104982845311530e-1, 0.164210583819080e-1, 0.222458491941670e-1,
                0.279370069800230e-1, 0.334601952825480e-1, 0.387821679744720e-1, 0.438709081856730e-1,
                0.486958076350720e-1, 0.532278469839370e-1, 0.574397690993910e-1, 0.613062424929290e-1,
                0.648040134566009e-1, 0.679120458152339e-1, 0.706116473912869e-1, 0.728865823958039e-1,
                0.747231690579679e-1, 0.761103619006259e-1, 0.770398181642479e-1, 0.775059479784249e-1],
            48 : [0.315334605230600e-2, 0.732755390127600e-2, 0.114772345792340e-1, 0.155793157229440e-1,
                0.196161604573550e-1, 0.235707608393240e-1, 0.274265097083570e-1, 0.311672278327980e-1,
                0.347772225647700e-1, 0.382413510658310e-1, 0.415450829434650e-1, 0.446745608566940e-1,
                0.476166584924900e-1, 0.503590355538540e-1, 0.528901894851940e-1, 0.551995036999840e-1,
                0.572772921004030e-1, 0.591148396983960e-1, 0.607044391658940e-1, 0.620394231598930e-1,
                0.631141922862539e-1, 0.639242385846479e-1, 0.644661644359499e-1, 0.647376968126839e-1],
            64 : [0.178328072169600e-2, 0.414703326056200e-2, 0.650445796897800e-2, 0.884675982636400e-2,
                0.111681394601310e-1, 0.134630478967190e-1, 0.157260304760250e-1, 0.179517157756970e-1,
                0.201348231535300e-1, 0.222701738083830e-1, 0.243527025687110e-1, 0.263774697150550e-1,
                0.283396726142590e-1, 0.302346570724020e-1, 0.320579283548510e-1, 0.338051618371420e-1,
                0.354722132568820e-1, 0.370551285402400e-1, 0.385501531786160e-1, 0.399537411327200e-1,
                0.412625632426230e-1, 0.424735151236530e-1, 0.435837245293230e-1, 0.445905581637560e-1,
                0.454916279274180e-1, 0.462847965813140e-1, 0.469681828162100e-1, 0.475401657148300e-1,
                0.479993885964580e-1, 0.483447622348030e-1, 0.485754674415030e-1, 0.486909570091400e-1],
            96 : [0.796792065552010e-3, 0.185396078894692e-2, 0.291073181793495e-2, 0.396455433844469e-2,
                0.501420274292752e-2, 0.605854550423596e-2, 0.709647079115386e-2, 0.812687692569876e-2,
                0.914867123078339e-2, 0.101607705350080e-1, 0.111621020998380e-1, 0.121516046710880e-1,
                0.131282295669610e-1, 0.140909417723140e-1, 0.150387210269940e-1, 0.159705629025620e-1,
                0.168854798642450e-1, 0.177825023160450e-1, 0.186606796274110e-1, 0.195190811401450e-1,
                0.203567971543330e-1, 0.211729398921910e-1, 0.219666444387440e-1, 0.227370696583290e-1,
                0.234833990859260e-1, 0.242048417923640e-1, 0.249006332224830e-1, 0.255700360053490e-1,
                0.262123407356720e-1, 0.268268667255910e-1, 0.274129627260290e-1, 0.279700076168480e-1,
                0.284974110650850e-1, 0.289946141505550e-1, 0.294610899581670e-1, 0.298963441363280e-1,
                0.302999154208270e-1, 0.306713761236690e-1, 0.310103325863130e-1, 0.313164255968610e-1,
                0.315893307707270e-1, 0.318287588944110e-1, 0.320344562319920e-1, 0.322062047940300e-1,
                0.323438225685750e-1, 0.324471637140640e-1, 0.325161187138680e-1, 0.325506144923630e-1]
        }

        # Now calculate the grid and weighting from these data chosen by npt

        mesh_r = x[npt]
        mesh_wt = wt[npt]

        r = np.zeros((2*n2))
        weight = np.zeros((2*n2))

        for i in range(n2):
            r[i] = -mesh_r[i]
            r[iof - (i + 1)] = mesh_r[i]
            weight[i] = mesh_wt[i]
            weight[iof - (i + 1)] = mesh_wt[i]

        return npt, r, weight

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
