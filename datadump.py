'''
2022-05-20 Susi Lehtola

This simple script can be used to dump the density data from the
code in a format that is easy to interface with low-level routines in
other languages.
'''

from AtomicOrbitals import GridGenerator, AtomData, Atom
from math import pi

actual, r, wt = GridGenerator.make_grid(2000)
grid = 4*pi*wt

data = AtomData()
for a in ['H', 'He', 'Li', 'Be', 'N', 'Ne', 'Na', 'Mg', 'Ar', 'Kr', 'Xe']:
    outf=open('{}.dat'.format(a),'w')
    outf.write('{}\n'.format(r.size))
    atom = Atom(a)
    d0, d1, g0, g1, t0, t1, l0, l1 = atom.get_densities(r)
    for i in range(r.size):
        outf.write('{: .16e} {: .16e} {: .16e} {: .16e} {: .16e} {: .16e} {: .16e} {: .16e} {: .16e} {: .16e} {: .16e}\n'.format(r[i], grid[i], d0[i], d1[i], g0[i]*g0[i], g0[i]*g1[i], g1[i]*g1[i], t0[i], t1[i], l0[i], l1[i]))
    outf.close()
