import pyscf
from pyscf import scf, tools
import numpy
from functools import reduce
from pyscf_util.misc.mole import get_orbsym
from pyscf_util.misc.misc import (
    read_mo_coeff_from_molden,
    read_mcscf_mo_coeff_from_molden,
)
from pyscf_util.Integrals.integral_sfX2C import *
from pyscf_util.Integrals.integral_CASCI import *
from pyscf_util.Integrals.integral_MRPT2 import *

from CONFIG import *

if __name__ == "__main__":

    for task_info in TaskInfoBDF:

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

        # check bdf convention and pyscf convention

        _, mo_energy, mo_coeff, mo_occ, irrep_labels_bdf, spins = tools.molden.load(
            molden_file
        )

        _, orbsym = get_orbsym(Mol, mo_coeff)
        # print(orbsym)

        for x1, x2 in zip(irrep_labels_bdf, orbsym):
            # print(x1, x2)
            assert x1.lower() == x2.lower()

        # (2) read in mo_coeff

        mo_coeff, mo_energy, mo_occ, orbsym_id, nfzc, nact, nvir = (
            read_mcscf_mo_coeff_from_molden(
                Mol,
                molden_file,
                task_info["NFZC"],
                task_info["NACT"],
                task_info["NVIR"],
            )
        )

        # nelec = nact

        SCF = pyscf.scf.RHF(Mol)
        CASSCF_Driver = pyscf.mcscf.CASSCF(SCF, nact, nact)

        # generate fcidump to perform active space iCIPT2

        DumpFileName = (
            "FCIDUMP"
            + "_"
            + task_info["title"]
            + "_"
            + "cc-pvdz"
            + "_"
            + "cas_%d_%d" % (nact, nact)
            + "_"
            + "icipt2"
        )

        dump_heff_casci(
            Mol,
            CASSCF_Driver,
            mo_coeff[:, :nfzc],
            mo_coeff[:, nfzc : nfzc + nact],
            DumpFileName,
        )

        # generate fcidump to perform full space mrpt2

        DumpFileName = (
            "FCIDUMP"
            + "_"
            + task_info["title"]
            + "_"
            + "cc-pvdz"
            + "_"
            + "cas_%d_%d" % (nact, nact)
            + "_"
            + "mrpt2"
        )

        fcidump_mrpt2_outcore(Mol, SCF, mo_coeff, nfzc, nact, nvir, DumpFileName, 1e-10)
