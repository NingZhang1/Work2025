import pyscf
from pyscf import scf, tools
import numpy
from functools import reduce
from pyscf_util.misc.mole import get_orbsym
from pyscf_util.misc.misc import (
    read_mo_coeff_from_molden,
    read_mcscf_mo_coeff_from_molden,
)

from CONFIG import *

if __name__ == "__main__":

    task_info = TaskInfoBDF[0]

    # (1) build mol

    molden_file = "molden/" + task_info["title"] + ".mcscf.molden"

    Mol = pyscf.gto.Mole()
    Mol.atom = task_info["mol"]
    Mol.basis = "cc-pvdz"
    Mol.symmetry = "d2h"
    Mol.spin = task_info["unpair"]
    Mol.charge = 0
    Mol.verbose = 4
    Mol.unit = "bohr"
    Mol.build()

    # _, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = tools.molden.load(molden_file)
    # orbsym = get_orbsym(Mol, mo_coeff)  # safer to rebuild it!
    # print(mo_energy)
    # print(irrep_labels)
    # print(spins)

    # print(orbsym)

    mo_coeff, mo_energy, orbsym_id = read_mo_coeff_from_molden(Mol, molden_file)

    print(mo_energy)
    print(orbsym_id)

    mo_coeff, mo_energy, mo_occ, orbsym_id, nfzc, nact, nvir = (
        read_mcscf_mo_coeff_from_molden(
            Mol, molden_file, task_info["NFZC"], task_info["NACT"], task_info["NVIR"]
        )
    )

    print(mo_energy)
    print(mo_occ)
    print(orbsym_id)

    print(nfzc, nact, nvir)
