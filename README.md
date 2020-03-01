# AtomicOrbitals

<p xmlns:dct="http://purl.org/dc/terms/" xmlns:vcard="http://www.w3.org/2001/vcard-rdf/3.0#">
  <a rel="license"
     href="http://creativecommons.org/publicdomain/zero/1.0/">
    <img src="https://licensebuttons.net/p/zero/1.0/80x15.png" style="border-style: none;" alt="CC0" />
  </a>
  <br />
  To the extent possible under law,
  <a rel="dct:publisher"
     href="https://www.jfurness.uk/44-2/hartree-fock-slater-orbitals-for-spherical-atoms/">
    <span property="dct:title">James Furness</span></a>
  has waived all copyright and related or neighboring rights to
  <span property="dct:title">Hartree-Fock Slater Orbitals for Spherical Atoms</span>.
This work is published from:
<span property="vcard:Country" datatype="dct:ISO3166"
      content="US" about="https://www.jfurness.uk/44-2/hartree-fock-slater-orbitals-for-spherical-atoms/">
  United States</span>.
</p>

A python module implementing accurate Hartree-Fock Slater orbitals for atoms under spherical symmetry.


<p>You are free to use it however you see fit. 



<p>Much of my recent work has been in developing new <a href="https://en.wikipedia.org/wiki/Density_functional_theory#Approximations_(exchange%E2%80%93correlation_functionals)">semi-local density functional approximations</a> following a non-empirical philosophy of adherence to exact mathematical constraints. These constraints help form the body of the functional then we use simple systems, such as atoms, to set the remaining degrees of freedom. Having a computationally efficient set of high-accuracy orbitals has been a real asset for searching large parameter spaces.</p>



<p>This python module implements the evaluation routines for the sets of orbitals published by <a href="http://dx.doi.org/10.1016/S0092-640X(74)80016-1">Clementi and Roetti in their 1974 paper</a>. As described in the publication these are the Slater type orbitals that minimise the spherically symmetric <a href="https://en.wikipedia.org/wiki/Roothaan_equations">Roothaan-Hartree-Fock Hamiltonian</a>. The code uses vectorised <code>numpy</code> functions to provide a fast evaluation of the electron density, electron density gradient, orbital kinetic energy density, and electron density Laplacian from the published Slater orbitals.</p>



<p>The user should be aware that these orbitals impose spherical symmetry on the system, including the <em>p</em> and <em>d</em> orbitals. This imposed symmetry is not a limitation, the orbitals are optimal for the spherical Hamiltonian, but it does mean that the energy from these orbitals for atoms with partially filled valence shells (e.g. carbon) will be different to the energy obtained by breaking the spatial symmetry.</p>



<p>The module's main interface is the <code>Atom</code> class that is initialised with the desired atomic element symbol:</p>



```python
import Densities
neon = Densities.Atom("Ne")
```


<p>Calling the initialised atom's <code>get_densities(r)</code> method with a distance from the nucleus (in atomic units) will return the spin resolved (0 or 1): electron density <code>d0, d1</code>, density gradient <code>g0, g1</code>, orbital kinetic energy <code>t0, t1</code>, and density Laplacian <code>l0, l1</code>, at the given distance from the nucleus. Due to the nature of the orbitals the nuclear distances must be positive and non-zero.</p>


```python
r = 1.5
d0, d1, g0, g1, t0, t1, l0, l1 = neon.get_densities(r)
```


<p>For many points in space it is best to use a <code>numpy</code> array to take advantage of the massive speed up offered by the vectorised routines:</p>


```python
r = np.linspace(0, 5, 500)
d0, d1, g0, g1, t0, t1, l0, l1 = neon.get_densities(r)
```


<p>The spherical symmetry then allows a simple shortcut to evaluating integrals over all space:</p>

```python
# Create a simple integration grid and evaluate the orbitals
r, h = np.linspace(1e-6, 25, 5000, retstep=True)
d0, d1, g0, g1, t0, t1, l0, l1 = neon.get_densities(r)

# Then integrate to find the total number of electrons.
# For neon this should be 10
density = np.sum(4*h*np.pi*r**2*(d0 + d1))
print("Total Density: {:.6f}".format(density))
```


<p>This is a viable and simple way of accessing the densities, but such evenly spaced integration grids need more points for accurate integrals than more refined methods. This module implements a simple Gauss-Legendre integration grid that can give better accuracy with a smaller number of points than a simple grid:</p>

```python
grid_level = 100  # Defines the accuracy of the grid. 100 is typically sufficient.
n, r, weights = Densities.GridGenerator.make_grid(grid_level)
d0, d1, g0, g1, t0, t1, l0, l1 = neon.get_densities(r)
density = np.sum(weights*(d0 + d1))
print("Total Density: {:.6f}".format(density))
```


<p>The returned values are numpy arrays that can be combined as normal, to generate the <a href="https://www.jfurness.uk/Publications/Furness2019.pdf">β iso-orbital indicator</a>:</p>

```python
from Densities import Atom
import numpy as np
import matplotlib.pyplot as plt

argon = Atom("Ar") # Initialise a new Argon atom

r = np.linspace(0.0001, 5, 500) # Create an array of points to evaluate

d0, d1, g0, g1, t0, t1, l0, l1 = argon.get_densities(r)

# Uniform electron gas kinetic energy density
tau_ueg_0 = 3.0/10.0*(3*np.pi**2)**(2.0/3.0)*d0**(5.0/3.0)
# von-Weizsacker (single-orbital) kinetic energy density
tau_vw_0 = g0**2/(8*d0)
# Calculate the beta iso-orbital indicator
beta_0 = (t0 - tau_vw_0)/(t0 + tau_ueg_0)

plt.plot(r, beta_0)
plt.xlabel("$r$")
plt.ylabel("$\\beta$ iso-orbital indicator")
plt.xlim([0,5])
plt.ylim([0,1])
plt.show()
```

<div class="wp-block-image"><figure class="aligncenter size-large is-resized"><img src="https://www.jfurness.uk/wp-content/uploads/2020/01/Argon_beta-1024x768.png" alt="" class="wp-image-377" width="512" height="384"/></figure></div>


<h2>Defining More Atoms</h2>


<p>At the time of writing the module contains orbitals for:</p>

<pre>{H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Si, P, Cl, Ar, K, Sc (4s1,3d2: High spin), Cr (4s1,3d5: High spin), Fe, Cu (4s1,3d10: High spin), Cu+ (3d10: Low spin), As, Kr, Ag, and Xe}</pre>

<p>The <a href="http://dx.doi.org/10.1016/S0092-640X(74)80016-1">Clementi and Roetti</a> paper contains orbital coefficients for all atoms with nuclear number ≤ 54 in many excited and ionic states. Unfortunately I haven't found a way to reliably automate reading the orbital coefficients from the paper, so they have all been entered by hand so far. The inefficiency of this process means I have only implemented atoms as they become useful to my work.</p>


<p>Such manual entry can be error prone, so the module contains an internal test routine to ensure correctness of evaluation functions, and to check that orbital coefficients produce the expected integrated quantities.</p>


<p>If you would like to add new atoms this can be done by adding the relevant coefficients to the dictionaries in the <code>AtomData</code> class, see code comments and the paper for details. Alongside the coefficients the integrated kinetic energy (from the paper) should be added to <code>ke_test</code> and the integrated electron density should be added to <code>N_test</code>. The test routine will automatically check the orbitals integrate to these values when the test set is run:</p>


```python
import Densities
Densities.test_densities()
```

<p>Alternatively running the module as main runs the test:</p>

```bash
python Densities.py
```

<p>And that's it, happy calculating! If this tool has been useful to you I'd love to hear about it. I'm also happy to add the coefficients for any other atoms on request.</p>

