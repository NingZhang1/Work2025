import pyscf
import numpy
from functools import reduce
from pyscf_util.misc.misc import (
    read_mcscf_mo_coeff_from_molden,
)
from pyscf_util.Integrals.integral_sfX2C import *
from pyscf_util.Integrals.integral_CASCI import *
from pyscf_util.Integrals.integral_MRPT2 import *
from pyscf_util.MeanField.iciscf import kernel

from CONFIG import *

if __name__ == "__main__":

    Mol = pyscf.gto.Mole()
    Mol.atom = GEOMETRY
    Mol.basis = "cc-pvdz"
    Mol.symmetry = "d2h"
    Mol.spin = 4
    Mol.charge = 0
    Mol.verbose = 4
    Mol.unit = "bohr"
    Mol.build()

    SCF = pyscf.scf.RHF(Mol)
    SCF.kernel()

    SCF.analyze()

    mo_coeff = SCF.mo_coeff

    file_cmoao.Dump_Cmoao("FeP_ROHF", mo_coeff)
