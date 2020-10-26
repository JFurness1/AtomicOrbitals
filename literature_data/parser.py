"""
Parsers for ATOM output to be used with James Furness' AtomicOrbitals program.

Written by Susi Lehtola, 2020-10-26
"""

import numpy

def read_block(data, iline):
    '''Reads in exponents and orbital coefficients'''

    amtype=data[iline].split()[0]
    if amtype == 'S':
        am=0
    elif amtype == 'P':
        am=1
    elif amtype == 'D':
        am=2
    elif amtype == 'F':
        am=3
    else:
        raise ValueError('Invalid angular momentum "{}"!'.format(amtype))

    # Number of molecular orbitals is
    nmo=len(data[iline].split())-1

    # Proceed by reading in data
    exps=[]
    coeffs=[]
    nvals=[]
    iline += 2 # skip the line for BASIS/ORB. ENERGY
    if data[iline].split()[0] == "CUSP": # K99 has this line, too
        iline+=1

    while iline < len(data) and len(data[iline].split()) == nmo+2:
        entries = data[iline].split()

        # Principal quantum number
        assert(entries[0][1] == amtype)
        nvals.append(int(entries[0][0]))

        # Exponent
        exps.append(entries[1])
        # and coefficients
        c=[]
        for imo in range(nmo):
            c.append(entries[imo+2])
        coeffs.append(numpy.asarray(c, dtype='float64'))
        iline+=1

    return iline, am, numpy.asarray(nvals, dtype='int64'), numpy.asarray(exps, dtype='float64'), numpy.asarray(coeffs, dtype='float64')

def parse_occupation(line):
    '''Parses the occupation data'''
    occdata = line.split()[1].replace(",","").replace("("," ").replace(")"," ").split()
    s_occ = [[], []]
    p_occ = [[], []]
    d_occ = [[], []]
    f_occ = [[], []]

    def hunds_rule(nelstr, am):
        '''Apply Hund's rule to determine spin-up and spin-down occupation'''
        nel = int(nelstr)
        upocc = min(nel,2*am+1)
        downocc = nel-upocc
        return upocc, downocc

    for ishell in range(0,len(occdata),2):
        if occdata[ishell] == 'K':
            assert(occdata[ishell+1] == '2')
            s_occ[0].append('1')
            s_occ[1].append('1')
        elif occdata[ishell] == 'L':
            assert(occdata[ishell+1] == '8')
            s_occ[0].append('1')
            s_occ[1].append('1')
            p_occ[0].append('3')
            p_occ[1].append('3')
        elif occdata[ishell] == 'M':
            assert(occdata[ishell+1] == '18')
            s_occ[0].append('1')
            s_occ[1].append('1')
            p_occ[0].append('3')
            p_occ[1].append('3')
            d_occ[0].append('5')
            d_occ[1].append('5')
        elif occdata[ishell][1] == 'S':
            upocc,downocc = hunds_rule(occdata[ishell+1], 0)
            if upocc+downocc > 0:
                s_occ[0].append(upocc)
                s_occ[1].append(downocc)
        elif occdata[ishell][1] == 'P':
            upocc,downocc = hunds_rule(occdata[ishell+1], 1)
            if upocc+downocc > 0:
                p_occ[0].append(upocc)
                p_occ[1].append(downocc)
        elif occdata[ishell][1] == 'D':
            upocc,downocc = hunds_rule(occdata[ishell+1], 2)
            if upocc+downocc > 0:
                d_occ[0].append(upocc)
                d_occ[1].append(downocc)
        elif occdata[ishell][1] == 'F':
            upocc,downocc = hunds_rule(occdata[ishell+1], 3)
            if upocc+downocc > 0:
                f_occ[0].append(upocc)
                f_occ[1].append(downocc)
        else:
            raise ValueError('Error parsing occupation line {}'.format(occdata))

    s_occ = [numpy.asarray(s_occ[0], dtype='float64'), numpy.asarray(s_occ[1], dtype='float64')]
    p_occ = [numpy.asarray(p_occ[0], dtype='float64'), numpy.asarray(p_occ[1], dtype='float64')]
    d_occ = [numpy.asarray(d_occ[0], dtype='float64'), numpy.asarray(d_occ[1], dtype='float64')]
    f_occ = [numpy.asarray(f_occ[0], dtype='float64'), numpy.asarray(f_occ[1], dtype='float64')]

    return s_occ, p_occ, d_occ, f_occ


def parse(filename):
    '''Parses Thakkar's output files'''

    with open(filename, "r") as f:
        data = f.readlines()

        # Line number
        iline = 0

        # Read occupation data
        s_occ, p_occ, d_occ, f_occ = parse_occupation(data[0])

        # Total energy
        while len(data[iline].strip())==0 or data[iline].split()[0] != "E":
            iline = iline + 1
        assert(data[iline].split()[0] == 'E')
        Etot = float(data[iline].split()[2])

        # Kinetic energy
        while len(data[iline].strip())==0 or data[iline].split()[0] != "T":
            iline = iline + 1
            assert(data[iline].split()[0] == 'T')
        Ekin = float(data[iline].split()[2])

        # Skip forward to basis function data
        while data[iline].strip() != "ORBITAL ENERGIES AND EXPANSION COEFFICIENTS":
            iline = iline + 1
        iline += 1

        # Read matrices
        ns = []
        xs = []
        cs = []
        ams = []
        while iline < len(data)-1:
            iline, am, nvals, exps, coeffs = read_block(data, iline)
            ams.append(am)
            ns.append(nvals)
            xs.append(exps)
            cs.append(coeffs)

            if am == 0:
                assert(coeffs.shape[1] == len(s_occ[0]))
                assert(coeffs.shape[1] == len(s_occ[1]))
            elif am == 1:
                assert(coeffs.shape[1] == len(p_occ[0]))
                assert(coeffs.shape[1] == len(p_occ[1]))
            elif am == 2:
                assert(coeffs.shape[1] == len(d_occ[0]))
                assert(coeffs.shape[1] == len(d_occ[1]))
            elif am == 3:
                assert(coeffs.shape[1] == len(f_occ[0]))
                assert(coeffs.shape[1] == len(f_occ[1]))
        return Etot, Ekin, ams, ns, xs, cs, s_occ, p_occ, d_occ, f_occ

if __name__ == "__main__":
    parse("neutral/xe")
