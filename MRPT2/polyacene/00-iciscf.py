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

        if task_info["unpair"] == 0:
            irrep_id = 0
        else:
            irrep_id = 5

        # nelec = nact

        SCF = pyscf.scf.RHF(Mol)
        SCF.kernel()
        mo_coeff = SCF.mo_coeff

        CMIN = 0.0
        _mc_conv_tol = 1e-7
        do_internal_rotation = False

        if nact > 18:
            CMIN = 1e-5
            _mc_conv_tol = 1e-6
            do_internal_rotation = True

        iciscf_res = kernel(
            Mol,
            SCF,
            nact,
            nact,
            _mo_init=mo_coeff,
            _cas_list=task_info["NACT"],
            _mc_conv_tol=_mc_conv_tol,
            _ici_state=[[task_info["unpair"], irrep_id, 1]],
            _cmin=CMIN,
            _do_pyscf_analysis=True,
            _internal_rotation=do_internal_rotation,
        )

        mo_coeff = iciscf_res.mo_coeff

        # dump #

        file_cmoao.Dump_Cmoao("%s_%s" % (task_info["title"], "cc-pvdz"), mo_coeff)

        # check ortho #

        ovlp = Mol.intor("int1e_ovlp")
        MO_ovlp = reduce(numpy.dot, (mo_coeff.T, ovlp, mo_coeff))

        ## print out and check ##

        for i in range(Mol.nao):
            for j in range(Mol.nao):
                if not (
                    (abs(MO_ovlp[i][j]) < 1e-10) or (abs(MO_ovlp[i][j] - 1) < 1e-10)
                ):
                    print(i, j, MO_ovlp[i][j])
