import pyscf
from pyscf import tools
from pyscf import symm
from pyscf.tools import fcidump
import pyscf.mcscf
from pyscf_util.MeanField.iciscf import *
from pyscf_util.File import file_cmoao


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


cas_space_symmetry = {
    "f+3": 2,
    "f+2": 2,
    "f+1": 2,
    "f0": 2,
    "f-1": 2,
    "f-2": 2,
    "f-3": 2,
}

if __name__ == "__main__":

    # BASIS = ["ano-rcc", "cc-pvtz-dk", "cc-pvqz-dk"]
    BASIS = ["cc-pvdz-dk"]  # for test

    for basis in BASIS:

        print(
            " -------------------------------------- Calculation with basis %s --------------------------------------\n"
            % (basis)
        )

        # build mole #

        Mol = pyscf.gto.Mole()
        Mol.atom = """
        Yb 0.0 0.0 0.0
        """
        Mol.basis = basis
        Mol.symmetry = True
        Mol.spin = 0
        Mol.charge = 0
        Mol.verbose = 4
        Mol.unit = "angstorm"
        Mol.build()

        # run scf #

        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 5e-12
        SCF.run()

        SCF.irrep_nelec = SCF.get_irrep_nelec()

        SCF.irrep_nelec["s+0"] = 12  # 12
        SCF.irrep_nelec["p+1"] = 8
        SCF.irrep_nelec["p-1"] = 8
        SCF.irrep_nelec["p+0"] = 8  # 24
        SCF.irrep_nelec["d-2"] = 4
        SCF.irrep_nelec["d-1"] = 4
        SCF.irrep_nelec["d+0"] = 4
        SCF.irrep_nelec["d+1"] = 4
        SCF.irrep_nelec["d+2"] = 4  # 20
        SCF.irrep_nelec["f-3"] = 2
        SCF.irrep_nelec["f-2"] = 2
        SCF.irrep_nelec["f-1"] = 2
        SCF.irrep_nelec["f+0"] = 2
        SCF.irrep_nelec["f+1"] = 2
        SCF.irrep_nelec["f+2"] = 2
        SCF.irrep_nelec["f+3"] = 2  # 14

        print(SCF.irrep_nelec)

        dm = SCF.make_rdm1()
        SCF.kernel(dm)

        SCF.analyze()

        # try iCISCF #

        mo_coeff = SCF.mo_coeff.copy()

        #### get the init guess ####

        Mol.spin = 1
        Mol.charge = 3
        # Mol.symmetry = "D2h"
        Mol.build()

        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 5e-12
        SCF.mo_coeff = mo_coeff

        mo_coeff = file_cmoao.ReadIn_Cmoao("Yb_%s" % (basis), Mol.nao, Mol.nao)

        norb = 7 * 2
        nelec = 13
        CASSCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)
        mo_init = pyscf.mcscf.sort_mo_by_irrep(
            CASSCF_Driver, CASSCF_Driver.mo_coeff, cas_space_symmetry
        )  # right!

        #### build d2h iciscf ####

        Mol.spin = 1
        Mol.charge = 3
        Mol.symmetry = "D2h"
        Mol.build()

        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 5e-12
        SCF.mo_coeff = mo_coeff

        SCF.mo_coeff = mo_init

        iCISCF_driver = iCISCF(
            SCF,
            norb,
            nelec,
            cmin=0.0,
            tol=1e-12,
            state=[
                [1, 4, 1, [1]],
                [1, 5, 2, [1, 1]],
                [1, 6, 2, [1, 1]],
                [1, 7, 2, [1, 1]],
            ],
        )

        energy, _, _, mo_coeff, mo_energy = iCISCF_driver.kernel(mo_coeff=mo_init)

        for i, ene in enumerate(mo_energy):
            print("%4d %15.8f" % (i, ene))

        file_cmoao.Dump_Cmoao("Yb_double_f_%s" % (basis), mo_coeff)

        # exit(1)

        # dump fci #

        # DumpFileName = "FCIDUMP" + "_" + case_str + "_" + mole_str + "_" + basis
        # fcidump.from_scf(SCF, DumpFileName, tol=1e-11)

        # build mole Tm #

        Mol = pyscf.gto.Mole()
        Mol.atom = """
        Tm 0.0 0.0 0.0
        """
        Mol.basis = basis
        Mol.symmetry = True
        Mol.spin = 0
        Mol.charge = -1
        Mol.verbose = 4
        Mol.unit = "angstorm"
        Mol.build()

        # run scf #

        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 5e-12
        SCF.run()

        SCF.irrep_nelec = SCF.get_irrep_nelec()
        SCF.irrep_nelec["s+0"] = 12  # 12
        SCF.irrep_nelec["p+1"] = 8
        SCF.irrep_nelec["p-1"] = 8
        SCF.irrep_nelec["p+0"] = 8  # 24
        SCF.irrep_nelec["d-2"] = 4
        SCF.irrep_nelec["d-1"] = 4
        SCF.irrep_nelec["d+0"] = 4
        SCF.irrep_nelec["d+1"] = 4
        SCF.irrep_nelec["d+2"] = 4  # 20
        SCF.irrep_nelec["f-3"] = 2
        SCF.irrep_nelec["f-2"] = 2
        SCF.irrep_nelec["f-1"] = 2
        SCF.irrep_nelec["f+0"] = 2
        SCF.irrep_nelec["f+1"] = 2
        SCF.irrep_nelec["f+2"] = 2
        SCF.irrep_nelec["f+3"] = 2  # 14

        # print(SCF.irrep_nelec)

        dm = SCF.make_rdm1()
        SCF.run(dm)
        SCF.analyze()

        # try iCISCF #

        mo_coeff = SCF.mo_coeff.copy()

        #### get the init guess ####

        Mol.spin = 0
        Mol.charge = 3
        # Mol.symmetry = "D2h"
        Mol.build()

        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 5e-12
        SCF.mo_coeff = mo_coeff
        
        mo_coeff = file_cmoao.ReadIn_Cmoao("Tm_%s" % (basis), Mol.nao, Mol.nao)

        norb = 7 * 2
        nelec = 12
        CASSCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)
        mo_init = pyscf.mcscf.sort_mo_by_irrep(
            CASSCF_Driver, CASSCF_Driver.mo_coeff, cas_space_symmetry
        )  # right!

        #### build d2h iciscf ####

        Mol.spin = 0
        Mol.charge = 3
        Mol.symmetry = "D2h"
        Mol.build()

        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 5e-12
        SCF.mo_coeff = mo_coeff

        SCF.mo_coeff = mo_init

        iCISCF_driver = iCISCF(
            SCF,
            norb,
            nelec,
            cmin=0.0,
            tol=1e-12,
            state=[
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
            ],
        )

        energy, _, _, mo_coeff, mo_energy = iCISCF_driver.kernel(mo_coeff=mo_init)

        for i, ene in enumerate(mo_energy):
            print("%4d %15.8f" % (i, ene))

        file_cmoao.Dump_Cmoao("Tm_double_f_%s" % (basis), mo_coeff)

        # exit(1)

        # build mole Er #

        Mol = pyscf.gto.Mole()
        Mol.atom = """
        Er 0.0 0.0 0.0
        """
        Mol.basis = basis
        Mol.symmetry = True
        Mol.spin = 0
        Mol.charge = -2
        Mol.verbose = 4
        Mol.unit = "angstorm"
        Mol.build()

        # run scf #

        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 5e-12
        SCF.run()

        SCF.irrep_nelec = SCF.get_irrep_nelec()
        SCF.irrep_nelec["s+0"] = 12  # 12
        SCF.irrep_nelec["p+1"] = 8
        SCF.irrep_nelec["p-1"] = 8
        SCF.irrep_nelec["p+0"] = 8  # 24
        SCF.irrep_nelec["d-2"] = 4
        SCF.irrep_nelec["d-1"] = 4
        SCF.irrep_nelec["d+0"] = 4
        SCF.irrep_nelec["d+1"] = 4
        SCF.irrep_nelec["d+2"] = 4  # 20
        SCF.irrep_nelec["f-3"] = 2
        SCF.irrep_nelec["f-2"] = 2
        SCF.irrep_nelec["f-1"] = 2
        SCF.irrep_nelec["f+0"] = 2
        SCF.irrep_nelec["f+1"] = 2
        SCF.irrep_nelec["f+2"] = 2
        SCF.irrep_nelec["f+3"] = 2  # 14

        dm = SCF.make_rdm1()
        SCF.run(dm)
        SCF.analyze()

        # try iCISCF #

        mo_coeff = SCF.mo_coeff.copy()

        #### get the init guess ####

        Mol.spin = 1
        Mol.charge = 3
        # Mol.symmetry = "D2h"
        Mol.build()

        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 5e-12
        SCF.mo_coeff = mo_coeff

        mo_coeff = file_cmoao.ReadIn_Cmoao("Er_%s" % (basis), Mol.nao, Mol.nao)
        
        norb = 7 * 2
        nelec = 11
        CASSCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)
        mo_init = pyscf.mcscf.sort_mo_by_irrep(
            CASSCF_Driver, CASSCF_Driver.mo_coeff, cas_space_symmetry
        )  # right!

        #### build d2h iciscf ####

        Mol.spin = 1
        Mol.charge = 3
        Mol.symmetry = "D2h"
        Mol.build()

        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 5e-12
        SCF.mo_coeff = mo_coeff

        SCF.mo_coeff = mo_init

        iCISCF_driver = iCISCF(
            SCF,
            norb,
            nelec,
            cmin=0.0,
            tol=1e-12,
            state=[
                # 2S+1 = 4 4I + 4F + 4S + 4G
                [3, 4, 4 + 1 + 1 + 3, [1, 1, 1, 1, 1, 1]],
                [3, 5, 3 + 2 + 2, [1, 1, 1, 1, 1]],
                [3, 6, 3 + 2 + 2, [1, 1, 1, 1, 1]],
                [3, 7, 3 + 2 + 2, [1, 1, 1, 1, 1]],
                # [3, 4, 4, [1, 1, 1, 1]],
                # [3, 5, 3, [1, 1, 1]],
                # [3, 6, 3, [1, 1, 1]],
                # [3, 7, 3, [1, 1, 1]],
                # 2S+1 = 2 2H + 2F
                [1, 4, 2 + 1, [1, 1, 1]],
                [1, 5, 3 + 2, [1, 1, 1, 1, 1]],
                [1, 6, 3 + 2, [1, 1, 1, 1, 1]],
                [1, 7, 3 + 2, [1, 1, 1, 1, 1]],
            ],
        )

        energy, _, _, mo_coeff, mo_energy = iCISCF_driver.kernel(mo_coeff=mo_init)

        file_cmoao.Dump_Cmoao("Er_double_f_%s" % (basis), mo_coeff)

        for i, ene in enumerate(mo_energy):
            print("%4d %15.8f" % (i, ene))
