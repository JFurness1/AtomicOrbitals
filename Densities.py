
import numpy as np
from math import pi

"""
Code to generate atomic electron densities, analytical gradients and orbital kinetic energy 
densities from Slater orbitals.

Core functionality comes from Atom() class which implements the get_densities(r) function 
to return quantities for numpy array of radial points r.

Orbital data are stored in static AtomData class which implements dictionaries of parameters
for different atom types accessed by upper case element symbols. New elements should be added
as entries into these dictionaries. Default orbital parameters are taken from:

The GridGenerator gives a simple implementation of an efficient Gauss-Legendre quadrature
grid.

Enrico Clementi, Carla Roetti,
Roothaan-Hartree-Fock atomic wavefunctions: Basis functions and their coefficients for
ground and certain excited states of neutral and ionized atoms, Z<=54,
Atomic Data and Nuclear Data Tables,
Volume 14, Issues 3-4,
1974,
Pages 177-478,
ISSN 0092-640X,
https://doi.org/10.1016/S0092-640X(74)80016-1.

Module is Python 2 and 3 compatible.

Example use can be seen in test_densities() function.
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

    actual, r, wt = GridGenerator.make_grid(200)
    grid = 4*pi*r**2*wt
    print("\nINTEGRATED DENSITY TEST")
    print("=======================")
    for a in list(AtomData.N_test.keys()):
        atom = Atom(a)
        d_bm = AtomData.N_test[a]
        d0, d1, g0, g1, t0, t1, l0, l1 = atom.get_densities(r)

        id0 = np.dot(d0, grid)
        id1 = np.dot(d1, grid)

        diff_0 = id0 - d_bm[0]
        percent_diff_0 = 100*diff_0/d_bm[0]
        if d_bm[1] > 0.0:
            diff_1 = id1 - d_bm[1]
            percent_diff_1 = 100*diff_1/d_bm[1]
        else:
            diff_1 = 0.0
            percent_diff_1 = 0.0
        print("{:>3} - N_0 = {:+2.6e}%, N_1 = {:+2.6e}%, {:}".format(a, percent_diff_0, percent_diff_1, "PASSED" if max(abs(diff_0), abs(diff_1)) < 1e-4 else "FAILED - "))

    print("\nINTEGRATED KINETIC TEST")
    print("=======================")
    for a in list(AtomData.ke_test.keys()):
        atom = Atom(a)
        t_bm = AtomData.ke_test[a]
        d0, d1, g0, g1, t0, t1, l0, l1 = atom.get_densities(r)

        it0 = np.dot(t0, grid)
        it1 = np.dot(t1, grid)

        diff = it0 + it1 - t_bm
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
    xe = Atom("Ne")
    # We only need to test a few points around the core
    fdh = 1e-8
    fdr = np.arange(0.9, 0.9+fdh*10, fdh)
    d0, d1, g0, g1, t0, t1, l0, l1 = xe.get_densities(fdr)

    # First the first central difference
    fdiff = (d0[2:]-d0[:-2])/(2*fdh)  # Construct the central difference
    if np.allclose(fdiff, g0[1:-1], atol=1e-1):  # finite difference is not perfect, so lenient tollerance
        print("Gradient: PASSED")
    else:
        print("Gradient: FAILED -")


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
            self.s_exp = AtomData.s_exp.get(u_atom, None)
            self.s_coef = AtomData.s_coef.get(u_atom, None)
            self.s_n = AtomData.s_n.get(u_atom, None)
            self.s_occ = AtomData.s_occ.get(u_atom, [0, 0])
            self.p_exp = AtomData.p_exp.get(u_atom, None)
            self.p_coef = AtomData.p_coef.get(u_atom, None)
            self.p_n = AtomData.p_n.get(u_atom, None)
            self.p_occ = AtomData.p_occ.get(u_atom, [0, 0])
            self.d_exp = AtomData.d_exp.get(u_atom, None)
            self.d_coef = AtomData.d_coef.get(u_atom, None)
            self.d_n = AtomData.d_n.get(u_atom, None)
            self.d_occ = AtomData.d_occ.get(u_atom, [0, 0])
        except KeyError:
            raise KeyError('Error: Atom data for "{:}" incomplete'.format(element))

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

        # Check that atom has occupied P orbitals
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

        # Check that atom has occupied D orbitals
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
            q_numbers = array of principle quantum numbers
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
            n: principle quantum number
            e: exponent
            r: space coordinate. (distance from nucleus)
        """
        FACTORS = np.array([2.0, 24.0, 720.0, 40320.0, 3628800.0])**(-0.5)
        n_facs = FACTORS[n - 1]
        c = n_facs*(2.0*e)**(n + 0.5)
        rn = np.power.outer(r, (n - 1))
        es = np.einsum('j,ij->ji', c, rn)
        pw = np.exp(-np.outer(e, r))
        return es*pw

    def DG(self, n, e, r, f):
        """
        Evaluates first derivative of slater orbitals.
            n: principle quantum number
            e: exponent
            r: space coordinate. (distance from nucleus)
            f: Undifferentiated function
        """

        pre = np.add(-e[:, None], np.divide.outer((n - 1), r))
        return pre*f

    def DDG(self, n, e, r, f):
        """
        Evaluates second derivative of slater orbitals.
            n: principle quantum number
            e: exponent
            r: space coordinate. (distance from nucleus)
            f: Undifferentiated function
        """
        pre = np.add(-e[:, None], np.divide.outer((n - 1), r))**2
        pre -= np.divide.outer((n - 1), r**2)
        return pre*f


class AtomData:
    """
    Class encapsulating raw data for all atoms.
    Primarily used to populate Atom class on instantiation.

    No need to give entries if the atom has none of that orbital, e.g. P and D for Li

    In these cases the default None and [0, 0] occupation should be used.
    Achieved by .get() accessing
    """

    # Testing data

    # Integrating tau_0 + tau_1 should approximate this closely
    ke_test = {
        'H'  : 0.5,
        'HE' : 0.28617128e1,
        'LI' : 0.74327544e1,
        'BE' : 0.14573036e2,
        'B'  : 0.24528956e2,
        'C'  : 0.37688357e2,
        'N'  : 0.54400522e2,
        'O'  : 0.74809346e2,
        'F'  : 0.99409140e2,
        'NE' : 0.12854681e3,
        'NA' : 0.16185668e3,
        'MG' : 0.19961025e3,
        'SI' : 0.28885431e3,
        'P'  : 0.34071381e3,
        'CL' : 0.45946376e3,
        'AR' : 0.52681380e3,
        'K'  : 0.59916415e3,
        'SC' : 0.75969760e3,  # Sc 4S1,3D2 (High spin)
        'CR' : 0.10433478e4,  # High spin Cr[Ar]4S1,3D5
        'CU' : 0.16389601e4,  # Spherical High Spin Cu[Ar]4S1, 3D10
        'CU+': 0.16387196e4,  # Low Spin Cu+[Ar]3D10
        'AS' : 0.22342374e4,
        'KR' : 0.27520481e4,
        'AG' : 0.51975130e4,      
        'XE' : 0.72320470e4
    }

    # Integrating den_0 and den_1 should approximate these closely
    N_test = {
        'H'  : [1.0, 0.0],
        'HE' : [1.0, 1.0],
        'LI' : [2.0, 1.0],
        'BE' : [2.0, 2.0],
        'B'  : [3.0, 2.0],
        'C'  : [4.0, 2.0],
        'N'  : [5.0, 2.0],
        'O'  : [5.0, 3.0],
        'F'  : [5.0, 4.0],
        'NE' : [5.0, 5.0],
        'NA' : [6.0, 5.0],
        'MG' : [6.0, 6.0],
        'SI' : [8.0, 6.0],
        'P'  : [9.0, 6.0],
        'CL' : [9.0, 8.0],
        'AR' : [9.0, 9.0],
        'K'  : [10.0, 9.0],
        'SC' : [12.0, 9.0],
        'CR' : [15.0, 9.0],
        'CU' : [15.0, 14.0], 
        'CU+': [14.0, 14.0],
        'AS' : [18.0, 15.0],
        'KR' : [18.0, 18.0],
        'AG' : [24.0, 23.0],
        'XE' : [27.0, 27.0]
    }

    ##################
    # S orbitals

    # S orbital exponents
    s_exp = {
        'H'  : np.array([1.0]),
        'HE' : np.array([1.41714, 2.37682, 4.39628, 6.52599, 7.94242]),
        'LI' : np.array([2.47673, 4.69873, 0.38350, 0.66055, 1.07000, 1.63200]),
        'BE' : np.array([3.47116, 6.36861, 0.77820, 0.94067, 1.48725, 2.71830]),
        'B'  : np.array([4.44561, 7.91796, 0.86709, 1.21924, 2.07264, 3.44332]),
        'C'  : np.array([5.43599, 9.48256, 1.05749, 1.52427, 2.68435, 4.20096]),
        'N'  : np.array([6.45739, 11.17200, 1.36405, 1.89734, 3.25291, 5.08238]),
        'O'  : np.array([7.61413, 13.75740, 1.69824, 2.48022, 4.31196, 5.86596]),
        'F'  : np.array([8.55760, 14.97660, 1.82142, 2.67295, 4.90066, 6.57362]),
        'NE' : np.array([9.48486, 15.56590, 1.96184, 2.86423, 4.82530, 7.79242]),
        'NA' : np.array([11.01230, 12.66010, 8.36156, 5.73805, 3.61287, 2.25096, 1.11597, 0.71028]),
        'MG' : np.array([12.01140, 13.91620, 9.48612, 6.72188, 4.24466, 2.53466, 1.46920, 0.89084]),
        'SI' : np.array([14.01420, 16.39320, 10.87950, 7.72709, 5.16500, 2.97451, 2.14316, 1.31306]),
        'P'  : np.array([15.01120, 17.31520, 11.77300, 8.66300, 5.90778, 3.69253, 2.47379, 1.51103]),
        'CL' : np.array([17.00140, 19.26490, 13.45290, 10.04290, 6.93920, 4.43640, 2.90570, 1.81900]),
        'AR' : np.array([18.01640, 22.04650, 16.08250, 11.63570, 7.70365, 4.87338, 3.32987, 2.02791]),
        'K'  : np.array([19.13500, 31.52500, 16.49860, 7.67410, 6.68508, 4.04102, 2.66919, 2.59794, 0.56203, 1.29017, 0.76641]),
        'SC' : np.array([33.43000, 20.88130, 18.18780, 8.42700, 7.45380, 4.78240, 3.24065, 2.65023, 1.47432, 1.06625, 0.79396]),
        'CR' : np.array([35.59110, 23.95440, 21.65020, 10.09060, 9.65415, 5.90457, 4.09494, 3.12628, 1.76632, 1.07837, 0.75455]),
        'CU' : np.array([28.48390, 42.50560, 23.54780, 13.26670, 11.52060, 8.09772, 6.70827, 5.07948, 3.19095, 1.53564, 0.87051]),
        'CU+': np.array([28.45300, 42.53270, 23.10340, 13.01550, 11.52770, 6.72990, 4.61704, 3.53495]),
        'AS' : np.array([31.34600, 38.60770, 26.26730, 14.94890, 13.45360, 8.13994, 5.53193, 3.14867, 2.01557, 1.42236]),
        'KR' : np.array([32.83510, 40.94470, 27.45800, 16.06660, 14.29620, 9.10937, 6.37181, 3.84546, 2.57902, 1.77192]),
        'AG' : np.array([48.27140, 34.04940, 21.93670, 21.32090, 14.14920, 10.13800, 5.87182, 3.98770, 2.66401, 1.65008, 1.04186]),
        'XE' : np.array([55.30720, 37.80730, 27.92970, 23.69210, 15.03530, 12.67230, 7.60195, 5.73899, 4.17583, 2.99772, 1.98532])
    }

    # Matrix of S orbital coefficients
    s_coef = {
        'H'  : np.array([[1.0]]),
        'HE' : np.array([
            [0.76838, 0.22346, 0.04082, -0.00994, 0.00230]
            ]),
        'LI' : np.array([
            [0.89786, 0.11131, -0.00008, 0.00112, -0.00216, 0.00884],
            [-0.14629, -0.01516, 0.00377, 0.98053, 0.10971, -0.11021]
            ]),
        'BE' : np.array([
            [0.91796, 0.08724, 0.00108, -0.00199, 0.00176, 0.00628],
            [-0.17092, -0.01455, 0.21186, 0.62499, 0.26662, -0.09919]
            ]),
        'B'  : np.array([
            [0.92705, 0.07780, 0.00088, -0.00200, 0.00433, 0.00270],
            [-0.19484, -0.01254, 0.06941, 0.75234, 0.31856, -0.12642]
            ]),
        'C'  : np.array([
            [0.93262, 0.06931, 0.00083, -0.00176, 0.00559, 0.00382],
            [-0.20814, -0.01071, 0.08099, 0.75045, 0.33549, -0.14765]
            ]),
        'N'  : np.array([
            [0.93780, 0.05849, 0.00093, -0.00170, 0.00574, 0.00957],
            [-0.21677, -0.00846, 0.17991, 0.67416, 0.31297, -0.14497]
            ]),
        'O'  : np.array([
            [0.94516, 0.03391, -0.00034, 0.00241, -0.00486, 0.03681],
            [-0.22157, -0.00476, 0.34844, 0.60807, 0.25365, -0.19183]
            ]),
        'F'  : np.array([
            [0.94710, 0.03718, 0.00013, 0.00093, 0.00068, 0.02602],
            [-0.22694, -0.00530, 0.23918, 0.68592, 0.31489, -0.21822]
            ]),
        'NE' : np.array([
            [0.93717, 0.04899, 0.00058, -0.00064, 0.00551, 0.01999],
            [-0.23093, -0.00635, 0.18620, 0.66899, 0.30910, -0.13871]
            ]),
        'NA' : np.array([
            [0.96179, 0.04052, 0.01919, -0.00298, 0.00191, -0.00049, 0.00016, -0.00007],
            [-0.23474, -0.00606, 0.11154, 0.43179, 0.51701, 0.04747, -0.00324, 0.00124],
            [0.03527, 0.00121, -0.01889, -0.06808, -0.09232, 0.00076, 0.40764, 0.64467]
            ]),
        'MG' : np.array([
            [0.96430, 0.03548, 0.02033, -0.00252, 0.00162,-0.00038, 0.00015, -0.00004],
            [-0.24357, -0.00485, 0.08002, 0.39902, 0.57358, 0.05156,-0.00703, 0.00161],
            [0.04691, 0.00144,-0.01850, -0.07964, -0.13478, -0.01906, 0.48239, 0.60221]
            ]),
        'SI' : np.array([
            [0.96800, 0.03033, 0.02248, -0.00617, 0.00326, -0.00143, 0.00081, -0.00016],
            [-0.25755, -0.00446, 0.11153, 0.40339, 0.55032, 0.03381, -0.00815, 0.00126],
            [0.06595, 0.00185, -0.03461, -0.10378, -0.19229, -0.06561, 0.59732, 0.55390]
            ]),
        'P'  : np.array([
            [0.96992, 0.02944, 0.01933, -0.00403, 0.00196, -0.00051, 0.00016, -0.00004],
            [-0.26326, -0.00434, 0.10333, 0.34612, 0.58778, 0.06043, -0.00901, 0.00193],
            [0.07230, 0.00186, -0.03447, -0.09503, -0.21241, -0.09001, 0.60361, 0.56185]
            ]),
        'CL' : np.array([
            [0.97335, 0.02682,0.01612,-0.00266, 0.00129,-0.00029, 0.00005,-0.00002],
            [-0.27278, -0.00266, 0.09766, 0.34603, 0.59594, 0.04978, -0.00324, 0.00121],
            [0.08249, 0.00237, -0.04193, -0.08968, -0.27243, -0.03736, 0.67062,0.47342]
            ]),
        'AR' : np.array([
            [0.97349, 0.01684, 0.02422, -0.00114, 0.00123, -0.00039, 0.00010, -0.00003],
            [0.27635, 0.00289, -0.03241, -0.33229, -0.65828, -0.06834, 0.00623, -0.00174],
            [0.08634, 0.00186, -0.01540, -0.10236, -0.27614, -0.11879, 0.68436, 0.52050]
            ]),
        'K'  : np.array([
            [-0.93619, -0.01385, -0.06342, -0.00014, -0.00139, 0.00189, -0.00212, 0.00118, -0.00005, -0.00015, 0.00011],
            [0.27612, 0.00055, 0.14725, -0.95199, -0.19289, -0.00059, -0.00704, 0.00327, -0.00016, -0.00045, 0.00033],
            [0.09267, -0.00066, 0.04980, -0.33547, -0.21345, 0.43855, 0.65200, 0.09749, 0.00560, 0.01932, -0.01161],
            [-0.01825, 0.00031, -0.00899, 0.06350, 0.05015, -0.11346, -0.11474, -0.03065, 0.05190, 0.33431, 0.70417]
            ]),
        'SC' : np.array([
            [-0.02013, -0.94537, -0.04302, -0.00472, 0.00331, -0.00239, 0.00177, -0.00078, 0.00041, -0.00034, 0.00012],
            [-0.00114, -0.28660, -0.14705, 1.01171, 0.12731, 0.01004, -0.00016, 0.00040, -0.00003, 0.00004, 0.00001],
            [-0.00011, -0.10055, -0.05537, 0.38472, 0.21487, -0.39743, -0.73852, -0.08548, -0.00356, 0.00277, -0.00046],
            [-0.00013, -0.02178, -0.01254, 0.08561, 0.05105, -0.09233, -0.20685, 0.03083, 0.37107, 0.35736, 0.34952]
            ]),
        'CR' : np.array([
            [-0.02722, -0.93357, -0.0472, -0.00604, 0.00363, -0.00151, 0.001, -0.00034, 0.00014, -0.00009, 0.00004],
            [-0.00977, -0.27707, -0.15417, 0.93227, 0.19827, 0.02629, -0.00339, 0.00078, 0.00009, 0.00005, 0.00006],
            [-0.00127, -0.10429, -0.05407, 0.34969, 0.2394, -0.25716, -0.82284, -0.12469, 0.00348, -0.00196, 0.00086],
            [-0.00018, -0.02218, -0.01125, 0.07371, 0.05676, -0.05977, -0.21516, 0.01657, 0.40517, 0.57019, 0.12089]
            ]),
        'CU' : np.array([
            [-0.95789, -0.02856, -0.01759, -0.00457, 0.00440, -0.00584, 0.00407, -0.00063, 0.00015, -0.00003, 0.00001],
            [-0.31805, 0.00412, -0.19769, 0.93663, 0.29775, -0.04582, 0.04007, -0.00460, 0.00127, -0.00028, 0.00010],
            [-0.11906, 0.00160, -0.08169, 0.38342, 0.28466, 0.04973, -0.83111, -0.44639, -0.03257, 0.00086, -0.00037],
            [-0.02267, 0.00143, -0.00941, 0.05604, 0.08953, -0.11580, -0.03951, -0.15751, 0.17463, 0.66426, 0.34469]
            ]),
        'CU+': np.array([
            [-0.95851, -0.02906, -0.01709, -0.00297, 0.00168, -0.00089, 0.00078, -0.00031],
            [-0.32587, 0.00819, -0.19715, 0.98177, 0.23062, 0.01499, -0.00405, 0.00191],
            [0.1216, -0.00297, 0.08245, -0.40335, -0.25689, 0.55723, 0.54604, 0.11723]
            ]),
        'AS' : np.array([
            [0.80944, 0.18590, 0.00949, -0.00069, 0.00121, -0.00083, 0.00039, -0.00014, 0.00012, -0.00005],
            [0.35297, -0.01548, 0.18189, -0.98989, -0.22138, -0.01725, 0.00209, -0.00065, 0.00050, -0.00020],
            [-0.13676, 0.00550, -0.08033, 0.42323, 0.27579, -0.49356, -0.72763, -0.01353, 0.00483, -0.00178],
            [-0.03678, 0.00086, -0.02303, 0.11843, 0.08597, -0.16171, -0.24575, 0.45102, 0.54525, 0.13765]
            ]),
        'KR' : np.array([
            [0.71521, 0.29911, -0.01854, 0.00897, -0.00464, 0.00190, -0.00088, 0.00026, -0.00018, 0.00006],
            [0.38139, -0.01823, 0.17175, -1.07160, -0.14913, -0.01920, 0.00401, -0.00122, 0.00092, -0.00031],
            [-0.14543, 0.00181, -0.09037, 0.49528, 0.25451, -0.48504, -0.75593, -0.01203, 0.00218, -0.00085],
            [-0.04349, -0.00148, -0.03219, 0.16451, 0.08852, -0.16671, -0.33291, 0.46913, 0.55106, 0.13572]
            ]),
        'AG' : np.array([
            [0.85321, 0.16287, -0.28452, 0.27326, -0.00757, 0.00328, -0.00088, 0.00058, -0.00024, 0.00013, -0.00005],
            [0.01541, 0.48995, 1.81227, -2.97601, -0.05501, 0.01094, -0.00178, 0.00118, -0.00045, 0.00024, -0.00008],
            [0.00405, 0.21698, 3.45888, -4.17359, 0.04946, 1.13958, 0.02993, -0.00847, 0.00277, -0.00132, 0.00044],
            [-0.00037, -0.09248, -1.55034, 1.86019, 0.04019, -0.71940, 0.77034, 0.42464, 0.01133, -0.00164, 0.00074],
            [-0.00019, 0.01987,  0.31024, -0.37499, -0.01483, 0.16544, -0.21760, -0.05567, 0.25280, 0.51023, 0.39485]
            ]),
        'XE' : np.array([
            [0.87059, 0.14926, -0.06259, 0.04643, -0.01383, 0.01030, -0.00254, 0.00201, -0.00085, 0.00045, -0.00011],
            [0.02107, 0.51209, -0.01873, -1.18386, -0.06502, 0.03432, -0.00469, 0.00352, -0.00136, 0.00069, -0.00017],
            [-0.00868, -0.23044, -0.38195, 1.12481, 0.23955, -1.41092, -0.06111, 0.02591, -0.00759, 0.00353, -0.00076],
            [0.00237, 0.10784, 0.19149, -0.54498, -0.35456, 1.13006, -0.63451, -0.58291, -0.02272, 0.00218, -0.00092],
            [0.00070, 0.03815, 0.06768, -0.19267, -0.15274, 0.44776, -0.30543, -0.24664, 0.27675, 0.59862, 0.30408]
            ])
    }

    # Principle quantum numbers of S functions. !Must be integer array for indexing!
    s_n = {
        'H' : np.array([1], dtype='int64'),
        'HE' : np.array([1, 1, 1, 1, 1], dtype='int64'),
        'LI' : np.array([1, 1, 2, 2, 2, 2], dtype='int64'),
        'BE' : np.array([1, 1, 2, 2, 2, 2], dtype='int64'),
        'B'  : np.array([1, 1, 2, 2, 2, 2], dtype='int64'),
        'C'  : np.array([1, 1, 2, 2, 2, 2], dtype='int64'),
        'N'  : np.array([1, 1, 2, 2, 2, 2], dtype='int64'),
        'O'  : np.array([1, 1, 2, 2, 2, 2], dtype='int64'),
        'F'  : np.array([1, 1, 2, 2, 2, 2], dtype='int64'),
        'NE' : np.array([1, 1, 2, 2, 2, 2], dtype='int64'),
        'NA' : np.array([1, 3, 3, 3, 3, 3, 3, 3], dtype='int64'),
        'MG' : np.array([1, 3, 3, 3, 3, 3, 3, 3], dtype='int64'),
        'SI' : np.array([1, 3, 3, 3, 3, 3, 3, 3], dtype='int64'),
        'P'  : np.array([1, 3, 3, 3, 3, 3, 3, 3], dtype='int64'),
        'CL' : np.array([1, 3, 3, 3, 3, 3, 3, 3], dtype='int64'),
        'AR' : np.array([1, 3, 3, 3, 3, 3, 3, 3], dtype='int64'),
        'K'  : np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype='int64'),
        'SC' : np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype='int64'),
        'CR' : np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype='int64'),
        'CU' : np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype='int64'),
        'CU+': np.array([1, 1, 2, 2, 3, 3, 3, 3], dtype='int64'),
        'AS' : np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4], dtype='int64'),
        'KR' : np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4], dtype='int64'),
        'AG' : np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5], dtype='int64'),
        'XE' : np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5], dtype='int64')
    }

    # The S orbitals (ascending) that are occupied [spin 0 (up), spin 1 (down)]
    s_occ = {
        'H'  : [np.array([1.0]), np.array([0.0])],
        'HE' : [np.array([1.0]), np.array([1.0])],
        'LI' : [np.array([1.0, 1.0]), np.array([1.0, 0.0])],
        'BE' : [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        'B'  : [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        'C'  : [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        'N'  : [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        'O'  : [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        'F'  : [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        'NE' : [np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        'NA' : [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.0])],
        'MG' : [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])], 
        'SI' : [np.array([1.0, 1.0, 1.0]),np.array([1.0, 1.0, 1.0])],
        'P'  : [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])],
        'CL' : [np.array([1.0, 1.0, 1.0]),np.array([1.0, 1.0, 1.0])],
        'AR' : [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])],
        'K'  : [np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 0.0])],
        'SC' : [np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 0.0])],
        'CR' : [np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 0.0])],
        'CU' : [np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 0.0])],
        'CU+': [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])],
        'AS' : [np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])],
        'KR' : [np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])],
        'AG' : [np.array([1.0, 1.0, 1.0, 1.0, 1.0]),np.array([1.0, 1.0, 1.0, 1.0, 0.0])],
        'XE' : [np.array([1.0, 1.0, 1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])]
    }

    ##################
    # P orbitals

    # P orbital exponents
    p_exp = {
        'B'  : np.array([0.87481, 1.36992, 2.32262, 5.59481]),
        'C'  : np.array([0.98073, 1.44361, 2.60051, 6.51003]),
        'N'  : np.array([1.16068, 1.70472, 3.03935, 7.17482]),
        'O'  : np.array([1.14394, 1.81730, 3.44988, 7.56484]),
        'F'  : np.array([1.26570, 2.05803, 3.92853, 8.20412]),
        'NE' : np.array([1.45208, 2.38168, 4.48489, 9.13464]),
        'MG' : np.array([5.92580, 7.98979, 5.32964, 3.71678, 2.59986]),
        'NA' : np.array([5.54977, 8.66846, 5.43460, 3.55503, 2.31671]),
        'SI' : np.array([7.14360, 16.25720, 10.79720, 6.89724, 4.66598, 2.32046, 1.33470, 0.79318]),
        'P'  : np.array([7.60940, 13.97590, 11.89390, 7.55531, 5.17707, 2.62934, 1.50494, 0.77783]),
        'CL' : np.array([8.50000, 15.01240, 12.32570, 8.37240, 6.10920, 3.19310, 1.78630, 0.92930]),
        'AR' : np.array([9.05477, 15.54410, 12.39970, 8.56120, 5.94658, 3.42459, 1.96709, 1.06717]),
        'K'  : np.array([8.64187, 15.19360, 6.91359, 3.26163, 2.00984, 1.68876]),
        'SC' : np.array([16.63600, 9.58828, 7.84586, 4.19594, 2.91102, 2.03624]),
        'CR' : np.array([16.08310, 10.47530, 9.35810, 5.64044, 3.41092, 1.98103]),
        'CU' : np.array([11.88610, 19.58060, 10.83980, 7.30670, 4.57017, 2.89365]),
        'CU+': np.array([12.36430, 19.52640, 11.52780, 7.15679, 4.50992, 2.86886]),
        'AS' : np.array([14.05460, 22.78430, 13.52830, 8.37724, 5.57821, 4.34244, 2.42567, 1.45140, 0.91898]),
        'KR' : np.array([17.03660, 26.04380, 15.51000, 9.49403, 6.57275, 5.38507, 3.15603, 2.02966, 1.42733]),
        'AG' : np.array([30.61540, 19.99580, 11.00880, 8.86584, 5.90958, 3.92436, 2.55069]),
        'XE' : np.array([34.88440, 23.30470, 12.54120, 12.02300, 7.72390, 5.40562, 3.32661, 2.09341, 1.36686])
    }

    # Matrix of P orbital coefficients
    p_coef = {
        'B'  : np.array([
            [0.53622, 0.40340, 0.11653, 0.00821]
            ]),
        'C'  : np.array([
            [0.28241, 0.54697, 0.23195, 0.01025]
            ]),
        'N'  : np.array([
            [0.26639, 0.52319, 0.27353, 0.01292]
            ]),
        'O'  : np.array([
            [0.16922, 0.57974, 0.32352, 0.01660]
            ]),
        'F'  : np.array([
            [0.17830, 0.56185, 0.33658, 0.01903]
            ]),
        'NE' : np.array([
            [0.21799, 0.53338, 0.32933, 0.01872]
            ]),
        'NA' : np.array([
            [0.46417, 0.03622, 0.29282, 0.31635, 0.07543]
            ]),
        'MG' : np.array([
            [0.52391, 0.07012, 0.31965, 0.20860, 0.03888]
            ]),
        'SI' : np.array([
            [0.54290, 0.00234, 0.04228, 0.32155, 0.22474, 0.00732, -0.00105, 0.00041],
            [-0.11535, -0.00189, -0.00473, -0.07552, 0.01041, 0.46075, 0.57665, 0.06274]
            ]),
        'P'  : np.array([
            [0.57352, 0.00664, 0.02478, 0.30460, 0.21442, 0.00552, -0.00045, 0.00011],
            [-0.13569, -0.00813, 0.00586, -0.08424, 0.02002, 0.51314, 0.55176, 0.02781]
            ]),
        'CL' : np.array([
            [0.63254, 0.00287, 0.03393, 0.27156, 0.16389, 0.00707, -0.00034, 0.00036],
            [-0.16954, -0.00982, 0.01280, -0.10925, 0.07066, 0.56909, 0.49144, 0.02336]
            ]),
        'AR' : np.array([
            [0.64116, 0.00865, 0.04186, 0.31735, 0.09642, 0.00003, 0.00055, -0.00013],
            [-0.17850, -0.00812, 0.00520, -0.10986, 0.10994, 0.56149, 0.46314, 0.02951]
            ]),
        'K'  : np.array([
            [0.66743, 0.04207, 0.34752, 0.01398, -0.00944, 0.00526],
            [-0.20797, -0.01176, -0.12744, 0.56718, 0.45273, 0.09340]
            ]),
        'SC' : np.array([
            [0.04468, 0.68927, 0.31591, 0.01334, -0.00350, 0.00090],
            [-0.01420, -0.23164, -0.14293, 0.39187, 0.51763, 0.22030]
            ]),
        'CR' : np.array([
            [0.11416, 0.6753, 0.24355, 0.01945, -0.00112, 0.00034],
            [-0.04558, -0.22347, -0.16991, 0.36285, 0.72269, 0.08413]
            ]),
        'CU' : np.array([
            [0.84302, 0.11714, 0.04499, 0.03012, -0.00511, 0.00182],
            [-0.32074, -0.04070, -0.10529, 0.37164, 0.67096, 0.14959]
            ]),
        'CU+': np.array([
            [0.76853, 0.11626, 0.12971, 0.02772, -0.00468, 0.00165],
            [-0.28521, -0.04270, -0.12476, 0.37278, 0.66459, 0.13742]
            ]),
        'AS' : np.array([
            [0.83284, 0.10192, 0.06278, 0.04654, -0.01655, 0.00612, -0.00205, 0.00109, -0.00039],
            [0.33887, 0.03589, 0.08838, -0.33676, -0.73040, -0.09452, -0.00136, -0.00010, 0.00004],
            [0.07735, 0.00736, 0.01932, -0.08004, -0.19430, 0.02331, 0.50866, 0.53655, 0.05044]
            ]),
        'KR' : np.array([
            [0.72322, 0.06774, 0.22056, 0.04478, -0.01672, 0.00609, -0.00195, 0.00111, -0.00040],
            [0.30185, 0.02508, 0.15903, -0.28475, -0.76440, -0.10670, -0.00562, 0.00137, -0.00053],
            [0.08488, 0.00571, 0.04169, -0.07425, -0.26866, 0.01341, 0.51241, 0.42557, 0.18141]
            ]),
        'AG' : np.array([
            [-0.13383, -0.86745, -0.03675, 0.02046, -0.00291, 0.00130, -0.00032],
            [0.03049, 0.47268, -0.65496, -0.46200, -0.02649, 0.00687, -0.00213],
            [0.00947, 0.19286, -0.26069, -0.32251, 0.59654, 0.53658, 0.07101]
            ]),
        'XE' : np.array([
            [0.13527, 0.86575, 0.11362, -0.09833, 0.00123, -0.00028, -0.00003, 0.00004, -0.00002],
            [0.02765, 0.49883, -0.48416, -0.61656, -0.05986, 0.01605, -0.00407, 0.00238, -0.00087],
            [-0.00908, -0.22945, -0.34216, 1.02476, -0.53369, -0.67016, -0.02313, 0.00433, -0.00136],
            [-0.00277, -0.07054, -0.18148, 0.40692, -0.22741, -0.21144, 0.49354, 0.53529, 0.13666]
            ])
    }

    # Principle quantum numbers of P orbitals. Must be integer array.
    p_n = {
        'B'  : np.array([2, 2, 2, 2], dtype='int64'),
        'C'  : np.array([2, 2, 2, 2], dtype='int64'),
        'N'  : np.array([2, 2, 2, 2], dtype='int64'),
        'O'  : np.array([2, 2, 2, 2], dtype='int64'),
        'F'  : np.array([2, 2, 2, 2], dtype='int64'),
        'NE' : np.array([2, 2, 2, 2], dtype='int64'),
        'NA' : np.array([2, 4, 4, 4, 4], dtype='int64'),
        'MG' : np.array([2, 4, 4, 4, 4], dtype='int64'),
        'SI' : np.array([2, 4, 4, 4, 4, 4, 4, 4], dtype='int64'),
        'P'  : np.array([2, 4, 4, 4, 4, 4, 4, 4], dtype='int64'),
        'CL' : np.array([2, 4, 4, 4, 4, 4, 4, 4], dtype='int64'),
        'AR' : np.array([2, 4, 4, 4, 4, 4, 4, 4], dtype='int64'),
        'K'  : np.array([2, 2, 3, 3, 3, 3], dtype='int64'),
        'SC' : np.array([2, 2, 3, 3, 3, 3], dtype='int64'),
        'CR' : np.array([2, 2, 3, 3, 3, 3], dtype='int64'),
        'CU' : np.array([2, 2, 3, 3, 3, 3], dtype='int64'),
        'CU+': np.array([2, 2, 3, 3, 3, 3], dtype='int64'),
        'AS' : np.array([2, 2, 3, 3, 3, 4, 4, 4, 4], dtype='int64'),
        'KR' : np.array([2, 2, 3, 3, 3, 4, 4, 4, 4], dtype='int64'),
        'AG' : np.array([2, 2, 3, 3, 4, 4, 4], dtype='int64'),
        'XE' : np.array([2, 2, 3, 3, 4, 4, 5, 5, 5], dtype='int64')
    }

    # P orbitals that are occupied [spin 0 (up), spin 1 (down)]
    p_occ = {
        'B'  : [np.array([1.0]), np.array([0.0])],
        'C'  : [np.array([2.0]), np.array([0.0])],
        'N'  : [np.array([3.0]), np.array([0.0])],
        'O'  : [np.array([3.0]), np.array([1.0])],
        'F'  : [np.array([3.0]), np.array([2.0])],
        'NE' : [np.array([3.0]), np.array([3.0])],
        'NA' : [np.array([3.0]), np.array([3.0])],
        'MG' : [np.array([3.0]), np.array([3.0])],
        'SI' : [np.array([3.0, 2.0]), np.array([3.0, 0.0])],
        'P'  : [np.array([3.0, 3.0]), np.array([3.0, 0.0])],
        'CL' : [np.array([3.0, 3.0]), np.array([3.0, 2.0])],
        'AR' : [np.array([3.0, 3.0]), np.array([3.0, 3.0])],
        'K'  : [np.array([3.0, 3.0]), np.array([3.0, 3.0])],
        'SC' : [np.array([3.0, 3.0]), np.array([3.0, 3.0])],
        'CR' : [np.array([3.0, 3.0]), np.array([3.0, 3.0])],
        'CU' : [np.array([3.0, 3.0]), np.array([3.0, 3.0])],
        'CU+': [np.array([3.0, 3.0]), np.array([3.0, 3.0])],
        'AS' : [np.array([3.0, 3.0, 3.0]), np.array([3.0, 3.0, 0.0])],
        'KR' : [np.array([3.0, 3.0, 3.0]), np.array([3.0, 3.0, 3.0])],
        'AG' : [np.array([3.0, 3.0, 3.0]), np.array([3.0, 3.0, 3.0])],
        'XE' : [np.array([3.0, 3.0, 3.0, 3.0]), np.array([3.0, 3.0, 3.0, 3.0])]
    }

    ##################
    # D orbitals

    # D orbital exponents
    d_exp = {
        'SC' : np.array([9.82074, 5.12071, 3.44801, 1.96214, 0.97339]),
        'CR' : np.array([9.76336, 5.47299, 3.45917, 2.18964, 1.26443]),
        'CU' : np.array([5.21851, 12.96880, 7.61139, 3.18734, 1.66248]),
        'CU+': np.array([4.70557, 13.47820, 7.38627, 2.93665, 1.69283]),
        'AS' : np.array([4.15670, 2.56420, 6.36170, 9.09175, 15.54610]),
        'KR' : np.array([5.30650, 3.36240, 7.94963, 10.35430, 17.11420]),
        'AG' : np.array([16.46200, 9.36028, 5.93684, 3.53384, 1.88607]),
        'XE' : np.array([20.08240, 11.78600, 7.30842, 4.88400, 3.19850])
    }

    # Matrix of D orbital coefficients
    d_coef = {
        'SC' : np.array([[0.01277, 0.08923, 0.23086, 0.44607, 0.43395]]),
        'CR' : np.array([[0.03088, 0.20871, 0.33792, 0.35411, 0.25996]]),
        'CU' : np.array([[0.29853, 0.02649, 0.18625, 0.42214, 0.26291]]),
        'CU+': np.array([[0.33575, 0.02401, 0.23802, 0.36175, 0.22921]]),
        'AS' : np.array([[0.44874, 0.11466, 0.30600, 0.22855, 0.02710]]),
        'KR' : np.array([[0.50854, 0.11070, 0.24778, 0.20584, 0.02863]]),
        'AG' : np.array([
            [0.23488, 0.78019, 0.05511, -0.00618, 0.00117],
            [-0.07446, -0.25709, 0.29070, 0.60532, 0.30158]
            ]),
        'XE' : np.array([
            [-0.19493, -0.80743, -0.06830, 0.02129, -0.00536],
            [-0.08265, -0.34860, 0.40928, 0.59391, 0.14481]
            ])
    }

    # Principle quantum numbers of D orbitals. Must be integer array.
    d_n = {
        'SC' : np.array([3, 3, 3, 3, 3], dtype='int64'),
        'CR' : np.array([3, 3, 3, 3, 3], dtype='int64'),
        'CU' : np.array([3, 3, 3, 3, 3], dtype='int64'),
        'CU+': np.array([3, 3, 3, 3, 3], dtype='int64'),
        'AS' : np.array([3, 3, 3, 3, 3], dtype='int64'),
        'KR' : np.array([3, 3, 3, 3, 3], dtype='int64'),
        'AG' : np.array([3, 3, 4, 4, 4], dtype='int64'),
        'XE' : np.array([3, 3, 4, 4, 4], dtype='int64')
    }

    # number of the D orbitals that are occupied [spin 0 (up), spin 1 (down)]
    d_occ = {
        'SC' : [np.array([2.0]), np.array([0.0])],
        'CR' : [np.array([5.0]), np.array([0.0])],
        'CU' : [np.array([5.0]), np.array([5.0])],
        'CU+': [np.array([5.0]), np.array([5.0])],
        'AS' : [np.array([5.0]), np.array([5.0])],
        'KR' : [np.array([5.0]), np.array([5.0])],
        'AG' : [np.array([5.0, 5.0]), np.array([5.0, 5.0])],
        'XE' : [np.array([5.0, 5.0]), np.array([5.0, 5.0])]
    }


class GridGenerator:
    """
    Static class encapsulating grid generation methods.

    Public facing method is make_grid(n)

    Then spherical integration can be achieved using,
        np.sum(f(r)*4*pi*weight*r**2)
    """
    @staticmethod
    def make_grid(n):
        """
        Generate a Gauss-Legendre grid for n points.
            n: desired grid points
        OUT:
            n: number of grid points (note: may be different to that requested)
            r: numpy array of coordinates
            wt: numpy array of weights
        """
        low = 0.0 # Lower Range
        high = 1.0
        p = 0.5

        n, r, wt = GridGenerator.gaussp(low, high, n)
        r = np.concatenate((r, np.zeros((n))))
        wt = np.concatenate((wt, np.zeros((n))))
        for i in range(n):
            r[2*n-(i+1)] = (1.0/r[i])**2
            wt[2*n-(i+1)] = (wt[i]/p)*r[2*n - (i+1)]**1.5

        return n, r, wt

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


if __name__ == "__main__":
    test_densities()
