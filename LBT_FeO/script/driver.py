# import package 

import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.tools import molden

# set test parameters 

BASIS = 'cc-pVDZ'
step = 0.1
begin = 1.5
end = 3.0 + step * 0.1

# run 

MoleIrrep = {
    'A1': (10,8),
    'E1x': (4,3),
    'E1y': (4,3),
    'E2x': (1,0),
    'E2y': (1,0),
}

ehf = []
emc = []

def run(b, dm, mo, ci=None):
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'FeO-%2.1f.out' % b
    mol.atom = [
        ['Fe',(  0.000000,  0.000000, 0.000000)],
        ['O', (  0.000000,  0.000000, b)],
    ]
    mol.basis = BASIS
    mol.symmetry = 'Coov'
    mol.build()
    mol.spin = 6
    mf = scf.ROHF(mol)
    mf.irrep_nelec = MoleIrrep
    mf.level_shift = .4
    mf.max_cycle = 100
    mf.conv_tol = 1e-12
    if dm is None:
        mf.run()
    else:
        mf.kernel(dm)
    mf.analyze()
    ehf.append(mf.scf(dm))
    
    # dump molden
    
    molden.from_mo(mol, 'FeO-ROHF-%2.1f.molden' % b, mf.mo_coeff)

    # mc = mcscf.CASSCF(mf, 12, 12)
    # mc.fcisolver.conv_tol = 1e-9
    # # FCI solver with multi-threads is not stable enough for this sytem
    # mc.fcisolver.threads = 1
    # if mo is None:
    #     # the initial guess for b = 1.5
    #     ncore = {'A1g':5, 'A1u':5}  # Optional. Program will guess if not given
    #     ncas = {'A1g':2, 'A1u':2,
    #             'E1ux':1, 'E1uy':1, 'E1gx':1, 'E1gy':1,
    #             'E2ux':1, 'E2uy':1, 'E2gx':1, 'E2gy':1}
    #     mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    # else:
    #     mo = mcscf.project_init_guess(mc, mo)
    # emc.append(mc.kernel(mo, ci)[0])
    # mc.analyze()
    # return mf.make_rdm1(), mc.mo_coeff, mc.ci
    
    return mf.make_rdm1(), None, None

dm = mo = ci = None
for b in numpy.arange(begin, end, step):
    dm, mo, ci = run(b, dm, mo, ci)

#for b in reversed(numpy.arange(1.5, 3.01, .1)):
#    dm, mo, ci = run(b, dm, mo, ci)

x = numpy.arange(begin, end, step)
ehf1 = ehf[:len(x)]
# ehf2 = ehf[len(x):]
# emc1 = emc[:len(x)]
# emc2 = emc[len(x):]
# ehf2.reverse()
# emc2.reverse()
with open('FeO-scan.txt', 'w') as fout:
    fout.write('     HF 1.5->3.0     CAS(12,12)      HF 3.0->1.5     CAS(12,12)\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
#                   % (xi, ehf1[i], emc1[i], ehf2[i], emc2[i]))
                     % (xi, ehf1[i], 0.0, 0.0, 0.0))

# import matplotlib.pyplot as plt
# plt.plot(x, ehf1, label='HF,1.5->3.0')
# plt.plot(x, ehf2, label='HF,3.0->1.5')
# plt.plot(x, emc1, label='CAS(12,12),1.5->3.0')
# plt.plot(x, emc2, label='CAS(12,12),3.0->1.5')
# plt.legend()
# plt.show()