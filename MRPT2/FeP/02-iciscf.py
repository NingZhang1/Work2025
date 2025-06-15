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
    # SCF.kernel()
    # SCF.analyze()

    mo_coeff_init = file_cmoao.ReadIn_Cmoao("FeP_ROHF", Mol.nao, Mol.nao)

    ### run iciscf for different active space ###

    for nelec, norb in ACTIVE_SPACE.keys():

        NFZC = ACTIVE_SPACE[(nelec, norb)]["NFZC"]
        NACT = ACTIVE_SPACE[(nelec, norb)]["NACT"]

        # ^5 Ag #

        # try 1e-4 first without internal rotation #

        CMIN = 1e-4
        _mc_conv_tol = 1e-7
        do_internal_rotation = False

        iciscf_res = kernel(
            Mol,
            SCF,
            nelec,
            norb,
            _mo_init=mo_coeff_init,
            _cas_list=NACT,
            _core_list=NFZC,
            _mc_conv_tol=_mc_conv_tol,
            _ici_state=[[4, 0, 1]],
            _cmin=CMIN,
            # _do_pyscf_analysis=True,
            _internal_rotation=do_internal_rotation,
        )

        mo_init2 = iciscf_res.mo_coeff

        file_cmoao.Dump_Cmoao("FeP_CAS_%d_%d_cmin1_5Ag_NoInternal", mo_init2)

        # try 1e-4 first with internal rotation #

        CMIN = 1e-4
        _mc_conv_tol = 1e-6
        do_internal_rotation = True

        iciscf_res = kernel(
            Mol,
            SCF,
            nelec,
            norb,
            _mo_init=mo_init2,
            _cas_list=NACT,
            _core_list=NFZC,
            _mc_conv_tol=_mc_conv_tol,
            _ici_state=[[4, 0, 1]],
            _cmin=CMIN,
            # _do_pyscf_analysis=True,
            _internal_rotation=do_internal_rotation,
        )

        mo_init3 = iciscf_res.mo_coeff

        file_cmoao.Dump_Cmoao("FeP_CAS_%d_%d_cmin1_5Ag", mo_init3)

        # try 1e-5 first without internal rotation #

        CMIN = 1e-5
        _mc_conv_tol = 1e-6
        do_internal_rotation = True

        iciscf_res = kernel(
            Mol,
            SCF,
            nelec,
            norb,
            _mo_init=mo_init3,
            _cas_list=NACT,
            _core_list=NFZC,
            _mc_conv_tol=_mc_conv_tol,
            _ici_state=[[4, 0, 1]],
            _cmin=CMIN,
            # _do_pyscf_analysis=True,
            _internal_rotation=do_internal_rotation,
        )

        mo_init4 = iciscf_res.mo_coeff

        file_cmoao.Dump_Cmoao("FeP_CAS_%d_%d_cmin2_5Ag", mo_init4)

        # ^3 B1g #

        # ^3 B2g #

        # ^3 B3g #
