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

    # 两边不知道怎么搞的，反正 ovlp check 过不去，重新优化吧 !

    # task_info = TaskInfoBDF[0]

    # (1) build mol

    molden_file = "data/FePOrb/CAS_14_18/" + "CAS-dz-3E1" + ".mcscf.molden"

    Mol = pyscf.gto.Mole()
    Mol.atom = GEOMETRY
    Mol.basis = "cc-pvdz"
    Mol.symmetry = True
    Mol.spin = 0
    Mol.charge = 0
    Mol.verbose = 4
    Mol.unit = "bohr"
    Mol.build()

    _, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = tools.molden.load(molden_file)

    px_id = []
    py_id = []
    pz_id = []

    for id, x in enumerate(Mol.ao_labels()):
        # print(x)
        if "pz" in x:
            pz_id.append(id)
        if "px" in x:
            px_id.append(id)
        if "py" in x:
            py_id.append(id)

    # print(px_id)
    # print(py_id)
    # print(pz_id)

    ordering = list(range(Mol.nao))
    # print(ordering)

    # 置换 #

    for x, y in zip(px_id, py_id):
        ordering[x] = y
    for x, y in zip(py_id, pz_id):
        ordering[x] = y
    for x, y in zip(pz_id, px_id):
        ordering[x] = y

    mo_coeff = mo_coeff[ordering, :]

    # exit(1)

    ovlp = Mol.intor("int1e_ovlp")
    print(ovlp)
    print(ovlp.dtype)
    print(mo_coeff.dtype)
    MO_ovlp = reduce(numpy.dot, (mo_coeff.T, ovlp, mo_coeff))
    # for i in range(Mol.nao):
    #     for j in range(Mol.nao):
    #         print(i, j, MO_ovlp[i][j])

    Mol = pyscf.gto.Mole()
    Mol.atom = "Fe 0 0 0"
    Mol.basis = "cc-pvdz"
    Mol.symmetry = True
    Mol.spin = 0
    Mol.charge = 0
    Mol.verbose = 4
    Mol.unit = "bohr"
    Mol.build()

    ovlp = Mol.intor("int1e_ovlp")
    print(ovlp)

    for i in range(Mol.nao):
        print(ovlp[i, i])

    # orbsym = get_orbsym(Mol, mo_coeff)  # safer to rebuild it!
    print(mo_energy)
    print(irrep_labels)
    print(spins)

    # print(orbsym)

    mo_coeff, mo_energy, orbsym_id = read_mo_coeff_from_molden(Mol, molden_file)

    print(mo_energy)
    print(orbsym_id)

    mo_coeff, mo_energy, mo_occ, orbsym_id, nfzc, nact, nvir = (
        read_mcscf_mo_coeff_from_molden(
            Mol,
            molden_file,
            ACTIVE_SPACE[(14, 18)]["NFZC"],
            ACTIVE_SPACE[(14, 18)]["NACT"],
        )
    )

    print(mo_energy)
    print(mo_occ)
    print(orbsym_id)

    print(nfzc, nact, nvir)
