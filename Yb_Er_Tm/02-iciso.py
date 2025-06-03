import pyscf
from pyscf import tools
from pyscf import symm
from pyscf.tools import fcidump
import pyscf.mcscf
from pyscf_util.MeanField.iciscf import *
from pyscf_util.File import file_cmoao
from pyscf_util.File import file_sodkh13
from pyscf_util.Relativisitc.sfX2C_soDKH import *
from pyscf import data
from pyscf_util.misc.icipt2_inputfile_generator import (
    _generate_task_spinarray_weight,
    _Generate_InputFile_iCI,
)


def build_dm1(nao, nelec_f, nelec_tot):
    ncore = (nelec_tot - nelec_f) // 2
    res = numpy.zeros((nao, nao))
    for i in range(ncore):
        res[i, i] = 2.0
    for i in range(ncore, ncore + 7):
        res[i, i] = float(nelec_f) / 7
    return res


def OrbSymInfo(Mol, mo_coeff):
    IRREP_MAP = {}
    nsym = len(Mol.irrep_name)
    for i in range(nsym):
        IRREP_MAP[Mol.irrep_name[i]] = i
    # print(IRREP_MAP)

    OrbSym = pyscf.symm.label_orb_symm(Mol, Mol.irrep_name, Mol.symm_orb, mo_coeff)
    IrrepOrb = []
    for i in range(len(OrbSym)):
        IrrepOrb.append(symm.irrep_name2id(Mol.groupname, OrbSym[i]))
    return IrrepOrb


if __name__ == "__main__":

    ICISO_APP = "/home/ningzhang1024sdu/GithubRepo/iCIPT2_CXX/bin/iCI_CPP_NEW.exe"

    BASIS = ["ano-rcc", "cc-pvtz-dk", "cc-pvqz-dk"]
    # BASIS = ["cc-pvdz-dk"]  # for test
    
    for basis in BASIS:

        print(
            " -------------------------------------- Calculation with basis %s --------------------------------------\n"
            % (basis)
        )

        # build Yb #

        Mol = pyscf.gto.Mole()
        Mol.atom = """
        Yb 0.0 0.0 0.0
        """
        Mol.basis = basis
        Mol.symmetry = "D2h"
        Mol.spin = 1
        Mol.charge = 3
        Mol.verbose = 2
        Mol.unit = "angstorm"
        Mol.build()

        ### dump fcidump and reldump ###

        DumpFileName = "FCIDUMP_Yb_%s" % (basis)
        RelDumpFileName = "RELDUMP_Yb_%s" % (basis)

        state = [
            [1, 4, 1, [1]],
            [1, 5, 2, [1, 1]],
            [1, 6, 2, [1, 1]],
            [1, 7, 2, [1, 1]],
        ]

        print(_generate_task_spinarray_weight(state)[0])
        task_name = "Yb_minmal_cas_%s" % (basis)
        task_str = _generate_task_spinarray_weight(state)[0]
        nelec_val = 13
        segment = "%d 0 3 4 0 %d" % (
            (Mol.nelectron - nelec_val) // 2,
            Mol.nao - 7 - (Mol.nelectron - nelec_val) // 2,
        )

        os.system("cp ../%s FCIDUMP" % (DumpFileName))
        os.system("cp ../%s RELDUMP" % (RelDumpFileName))

        _Generate_InputFile_iCI(
            task_name + ".inp",
            segment,
            nelec_val,
            0,
            "0.0",
            0,
            0,
            1,
            task_str,
            0,
            1e-10,
            1,
            "d2h",
            0,
        )

        os.system(
            "%s %s.inp 1>%s.out 2>%s.err" % (ICISO_APP, task_name, task_name, task_name)
        )

        # build Tm #

        Mol = pyscf.gto.Mole()
        Mol.atom = """
        Tm 0.0 0.0 0.0
        """
        Mol.basis = basis
        Mol.symmetry = "D2h"
        Mol.spin = 0
        Mol.charge = 3
        Mol.verbose = 2
        Mol.unit = "angstorm"
        Mol.build()

        ### dump fcidump and reldump ###

        DumpFileName = "FCIDUMP_Tm_%s" % (basis)
        RelDumpFileName = "RELDUMP_Tm_%s" % (basis)

        state = [
            # triplet 3H + 3F + 3P
            [2, 0, 2 + 1, [1, 1, 1]],
            [2, 1, 3 + 2 + 1, [1, 1, 1, 1, 1, 1]],
            [2, 2, 3 + 2 + 1, [1, 1, 1, 1, 1, 1]],
            [2, 3, 3 + 2 + 1, [1, 1, 1, 1, 1, 1]],
            # singlet 1G + 1D + 1I
            [0, 0, 3 + 2 + 4, [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [0, 1, 2 + 1 + 3, [1, 1, 1, 1, 1, 1]],
            [0, 2, 2 + 1 + 3, [1, 1, 1, 1, 1, 1]],
            [0, 3, 2 + 1 + 3, [1, 1, 1, 1, 1, 1]],
        ]

        print(_generate_task_spinarray_weight(state)[0])
        task_name = "Tm_minmal_cas_%s" % (basis)
        task_str = _generate_task_spinarray_weight(state)[0]
        nelec_val = 12
        segment = "%d 0 3 4 0 %d" % (
            (Mol.nelectron - nelec_val) // 2,
            Mol.nao - 7 - (Mol.nelectron - nelec_val) // 2,
        )

        os.system("cp ../%s FCIDUMP" % (DumpFileName))
        os.system("cp ../%s RELDUMP" % (RelDumpFileName))

        _Generate_InputFile_iCI(
            task_name + ".inp",
            segment,
            nelec_val,
            0,
            "0.0",
            0,
            0,
            1,
            task_str,
            0,
            1e-10,
            1,
            "d2h",
            0,
        )

        os.system(
            "%s %s.inp 1>%s.out 2>%s.err" % (ICISO_APP, task_name, task_name, task_name)
        )

        # build Er #

        Mol = pyscf.gto.Mole()
        Mol.atom = """
        Er 0.0 0.0 0.0
        """
        Mol.basis = basis
        Mol.symmetry = "D2h"
        Mol.spin = 1
        Mol.charge = 3
        Mol.verbose = 2
        Mol.unit = "angstorm"
        Mol.build()

        ### dump fcidump and reldump ###

        DumpFileName = "FCIDUMP_Er_%s" % (basis)
        RelDumpFileName = "RELDUMP_Er_%s" % (basis)

        state = [
            # 2S+1 = 4 4I + 4F + 4S + 4G
            [3, 4, 4 + 1 + 1, [1, 1, 1, 1, 1, 1]],
            [3, 5, 3 + 2, [1, 1, 1, 1, 1]],
            [3, 6, 3 + 2, [1, 1, 1, 1, 1]],
            [3, 7, 3 + 2, [1, 1, 1, 1, 1]],
            # [3, 4, 4, [1, 1, 1, 1]],
            # [3, 5, 3, [1, 1, 1]],
            # [3, 6, 3, [1, 1, 1]],
            # [3, 7, 3, [1, 1, 1]],
            # 2S+1 = 2 2H + 2F
            [1, 4, 2 + 1, [1, 1, 1]],
            [1, 5, 3 + 2, [1, 1, 1, 1, 1]],
            [1, 6, 3 + 2, [1, 1, 1, 1, 1]],
            [1, 7, 3 + 2, [1, 1, 1, 1, 1]],
        ]

        print(_generate_task_spinarray_weight(state)[0])
        task_name = "Er_minmal_cas_%s" % (basis)
        task_str = _generate_task_spinarray_weight(state)[0]
        nelec_val = 11
        segment = "%d 0 3 4 0 %d" % (
            (Mol.nelectron - nelec_val) // 2,
            Mol.nao - 7 - (Mol.nelectron - nelec_val) // 2,
        )

        os.system("cp ../%s FCIDUMP" % (DumpFileName))
        os.system("cp ../%s RELDUMP" % (RelDumpFileName))

        _Generate_InputFile_iCI(
            task_name + ".inp",
            segment,
            nelec_val,
            0,
            "0.0",
            0,
            0,
            1,
            task_str,
            0,
            1e-10,
            1,
            "d2h",
            0,
        )

        os.system(
            "%s %s.inp 1>%s.out 2>%s.err" % (ICISO_APP, task_name, task_name, task_name)
        )