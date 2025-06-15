# coding=UTF-8

import pyscf
from pyscf import tools
from pyscf import symm
from pyscf.tools import fcidump
import pyscf.mcscf
from pyscf_util.MeanField import iciscf
from pyscf_util.File import file_cmoao
from pyscf_util.Integrals.integral_sfX2C import *
from pyscf_util.Integrals.integral_CASCI import *
from pyscf_util.Integrals.integral_MRPT2 import get_generalized_fock
from pyscf_util.File import file_rdm
import os

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


def get_sym(IrrepMap, Occ):
    res = 0
    for i in range(len(Occ)):
        if Occ[i] == 1:
            res ^= IrrepMap[i] % 10
    return res


cas_space_symmetry = {
    "A1u": 1 + 1,  # 5
    "A1g": 1 + 1,  # 0
    "E1ux": 1,  # 7
    "E1gy": 1,  # 3
    "E1gx": 1,  # 2
    "E1uy": 1,  # 6
    "E2gy": 1,  # 1
    "E2gx": 1,  # 0
    "E2uy": 1,  # 4
    "E2ux": 1,  # 5
}

cas_space_44_symmetry = {
    "A1u": 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,  # 5
    "A1g": 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,  # 0
    "E1ux": 1 + 1 + 1 + 1 + 1,  # 7
    "E1gy": 1 + 1 + 1 + 1 + 1,  # 3
    "E1gx": 1 + 1 + 1 + 1 + 1,  # 2
    "E1uy": 1 + 1 + 1 + 1 + 1,  # 6
    "E2gy": 1 + 1,  # 1
    "E2gx": 1 + 1,  # 0
    "E2uy": 1 + 1,  # 4
    "E2ux": 1 + 1,  # 5
}

cas_space_58_symmetry = {
    "A1u": 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,  # 5
    "A1g": 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,  # 0
    "E1ux": 1 + 1 + 1 + 1 + 1 + 1,  # 7
    "E1gy": 1 + 1 + 1 + 1 + 1 + 1,  # 3
    "E1gx": 1 + 1 + 1 + 1 + 1 + 1,  # 2
    "E1uy": 1 + 1 + 1 + 1 + 1 + 1,  # 6
    "E2gy": 1 + 1 + 1,  # 1
    "E2gx": 1 + 1 + 1,  # 0
    "E2uy": 1 + 1 + 1,  # 4
    "E2ux": 1 + 1 + 1,  # 5
    "E3gy": 1,  # 1
    "E3gx": 1,  # 0
    "E3uy": 1,  # 4
    "E3ux": 1,  # 5
}

# cas_space_symmetry = {
#     'Ag': 3,
#     'B1g':1,
#     'B2g':1,
#     'B3g':1,
#     'Au': 1,
#     'B1u':3,
#     'B2u':1,
#     'B3u':1,
# }


if __name__ == "__main__":

    BOND_LENGTH = [1.3, 1.4, 1.5, 1.6, 1.68, 1.8, 1.9, 2.0, 2.2, 2.5, 2.8, 3.2]
    # BOND_LENGTH = [1.68]
    BASIS = ["ccpvdz-dk", "ccpvtz-dk", "ccpvqz-dk", "ccpv5z-dk"]

    for basis in BASIS:
        for BondLength in BOND_LENGTH:

            Mol = pyscf.gto.Mole()
            Mol.atom = """
    Cr     0.0000      0.0000  %f 
    Cr     0.0000      0.0000  -%f 
    """ % (
                BondLength / 2,
                BondLength / 2,
            )
            Mol.basis = basis
            Mol.symmetry = "Dooh"
            Mol.spin = 2
            Mol.charge = 0
            Mol.verbose = 4  # print everything!
            Mol.unit = "angstorm"
            Mol.build()
            SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
            SCF.max_cycle = 32
            SCF.conv_tol = 1e-9
            # SCF.run()

            # print(SCF.energy)

            bond_int = int(BondLength * 100)

            Mol.symmetry = "D2h"
            Mol.spin = 0
            Mol.build()

            # iCISCF

            # the driver

            norb = 12
            nelec = 12
            CASSCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)

            mo_coeff = file_cmoao.ReadIn_Cmoao(
                "../cas_28_44_%s_%d" % (basis, BondLength * 100), Mol.nao, Mol.nao
            )
            
            ##### generate generalized fock matrix #####
            
            dump_heff_casci(
                Mol,
                CASSCF_Driver,
                mo_coeff[:, :18],
                mo_coeff[:, 18:30],
                _filename="FCIDUMP_Cr2_cas1212",
            )
            
            from pyscf_util.iCIPT2.iCIPT2 import kernel
            
            kernel(
            IsCSF=True,
            task_name="cr2_rdm1",
            fcidump="FCIDUMP_Cr2",
            segment="0 0 6 6 0 0",
            nelec_val=12,
            rotatemo=0,
            cmin=0.0,
            perturbation=0,
            dumprdm=1,
            relative=0,
            Task="0 0 1 1",
            inputocfg=0,
            etol=1e-10,
            selection=1,
            doublegroup=None,
            direct=None,
            start_with=None,
            end_with=[".csv"],
        )
            os.system("mv rdm1.csv cr2_rdm1.csv")

            rdm1 = file_rdm.ReadIn_rdm1("cr2_rdm1", 12, 12)
            
            gfock = get_generalized_fock(CASSCF_Driver, mo_coeff, rdm1)
            
            # dump the gfock matrix
            
            filename = "gfock_cr2_%s_%d.csv" % (basis, bond_int)
            file = open(filename, "w")
            file.write("i,j,fock\n")
            for i in range(Mol.nao):
                for j in range(Mol.nao):
                    if abs(gfock[i][j]) > 1e-10:
                        file.write(
                            "%d,%d,%20.12e\n" % (i, j, gfock[i][j])
                        )
            file.close()