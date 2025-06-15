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
from pyscf_util.Integrals.integral_sfX2C import fcidump_sfx2c


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

    # BASIS = ["ano-rcc", "cc-pvtz-dk", "cc-pvqz-dk"]
    BASIS = ["ano-rcc"]  # for test

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
        Mol.verbose = 4
        Mol.unit = "angstorm"
        Mol.build()

        mo_coeff = file_cmoao.ReadIn_Cmoao("Yb_%s" % (basis), Mol.nao, Mol.nao)
        mo_coeff_doublef = file_cmoao.ReadIn_Cmoao(
            "Yb_double_f_%s" % (basis), Mol.nao, Mol.nao
        )

        # heff #

        # dm1 = build_dm1(Mol.nao, 13, Mol.nelectron)
        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        # hsf = numpy.zeros((1, Mol.nao, Mol.nao))
        # hso = fetch_X2C_soDKH1(Mol, SCF, mo_coeff, dm1)
        # hso *= (data.nist.ALPHA**2) / 4.0
        # hso = numpy.vstack((hso, hsf))

        ### dump fcidump and reldump ###

        orbsym = OrbSymInfo(Mol, mo_coeff)
        hso = file_sodkh13.ReadIn_Relint_csv("RELDUMP_Yb_%s" % (basis), Mol.nao)

        ## mocoeff --> doublef_mocoeff ##

        ovlp = Mol.intor("int1e_ovlp")
        mocoeff_2_doublef_mocoeff = mo_coeff.T @ ovlp @ mo_coeff_doublef

        ### do integral transformation rather than calculation ###

        hso_doublef = numpy.zeros((4, Mol.nao, Mol.nao))
        for k in range(4):
            hso_doublef[k] = (
                mocoeff_2_doublef_mocoeff.T @ hso[k] @ mocoeff_2_doublef_mocoeff
            )

        file_sodkh13.Dump_Relint_iCI(
            "RELDUMP_Yb_%s_double_f" % (basis), hso_doublef, Mol.nao
        )

        ## run fcidump ##

        DumpFileName = "FCIDUMP_Yb_%s_double_f" % (basis)
        fcidump_sfx2c(Mol, SCF, mo_coeff, DumpFileName, 1e-12)

        # build Tm #

        Mol = pyscf.gto.Mole()
        Mol.atom = """
        Tm 0.0 0.0 0.0
        """
        Mol.basis = basis
        Mol.symmetry = "D2h"
        Mol.spin = 0
        Mol.charge = 3
        Mol.verbose = 4
        Mol.unit = "angstorm"
        Mol.build()

        mo_coeff = file_cmoao.ReadIn_Cmoao("Tm_%s" % (basis), Mol.nao, Mol.nao)
        mo_coeff_doublef = file_cmoao.ReadIn_Cmoao(
            "Tm_double_f_%s" % (basis), Mol.nao, Mol.nao
        )

        # reorder mo_coeff #

        # dm1 = build_dm1(Mol.nao, 12, Mol.nelectron)
        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        # hsf = numpy.zeros((1, Mol.nao, Mol.nao))
        # hso = fetch_X2C_soDKH1(Mol, SCF, mo_coeff, dm1)
        # hso *= (data.nist.ALPHA**2) / 4.0
        # hso = numpy.vstack((hso, hsf))

        ### dump fcidump and reldump ###

        orbsym = OrbSymInfo(Mol, mo_coeff)
        # DumpFileName = "FCIDUMP_Tm_%s" % (basis)
        DumpFileName = "FCIDUMP_Tm_%s_double_f" % (basis)
        # fcidump.from_mo(Mol, DumpFileName, mo_coeff, orbsym, tol=1e-12)
        fcidump_sfx2c(Mol, SCF, mo_coeff, DumpFileName, 1e-12)

        hso = file_sodkh13.ReadIn_Relint_csv("RELDUMP_Tm_%s" % (basis), Mol.nao)

        ## mocoeff --> doublef_mocoeff ##

        ovlp = Mol.intor("int1e_ovlp")
        mocoeff_2_doublef_mocoeff = mo_coeff.T @ ovlp @ mo_coeff_doublef

        hso_doublef = numpy.zeros((4, Mol.nao, Mol.nao))
        for k in range(4):
            hso_doublef[k] = (
                mocoeff_2_doublef_mocoeff.T @ hso[k] @ mocoeff_2_doublef_mocoeff
            )

        file_sodkh13.Dump_Relint_iCI(
            "RELDUMP_Tm_%s_double_f" % (basis), hso_doublef, Mol.nao
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
        Mol.verbose = 4
        Mol.unit = "angstorm"
        Mol.build()

        mo_coeff = file_cmoao.ReadIn_Cmoao("Er_%s" % (basis), Mol.nao, Mol.nao)
        mo_coeff_doublef = file_cmoao.ReadIn_Cmoao(
            "Er_double_f_%s" % (basis), Mol.nao, Mol.nao
        )

        # reorder mo_coeff #

        # dm1 = build_dm1(Mol.nao, 11, Mol.nelectron)
        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        # hsf = numpy.zeros((1, Mol.nao, Mol.nao))
        # hso = fetch_X2C_soDKH1(Mol, SCF, mo_coeff, dm1)
        # hso *= (data.nist.ALPHA**2) / 4.0
        # hso = numpy.vstack((hso, hsf))

        ### dump fcidump and reldump ###

        orbsym = OrbSymInfo(Mol, mo_coeff)
        DumpFileName = "FCIDUMP_Er_%s_double_f" % (basis)
        # fcidump.from_mo(Mol, DumpFileName, mo_coeff, orbsym, tol=1e-12)
        fcidump_sfx2c(Mol, SCF, mo_coeff, DumpFileName, 1e-12)

        hso = file_sodkh13.ReadIn_Relint_csv("RELDUMP_Er_%s" % (basis), Mol.nao)

        ## mocoeff --> doublef_mocoeff ##

        ovlp = Mol.intor("int1e_ovlp")
        mocoeff_2_doublef_mocoeff = mo_coeff.T @ ovlp @ mo_coeff_doublef

        has_doublef = numpy.zeros((4, Mol.nao, Mol.nao))
        for k in range(4):
            has_doublef[k] = (
                mocoeff_2_doublef_mocoeff.T @ hso[k] @ mocoeff_2_doublef_mocoeff
            )

        file_sodkh13.Dump_Relint_iCI(
            "RELDUMP_Er_%s_double_f" % (basis), has_doublef, Mol.nao
        )
