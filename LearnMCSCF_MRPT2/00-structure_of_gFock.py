# coding=UTF-8

import pyscf
from pyscf import tools
from pyscf import symm
from pyscf.tools import fcidump
import pyscf.mcscf
from pyscf_util.MeanField import iciscf
from pyscf.mcscf.casci import get_fock


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
    "A1u": 2,  # 5
    "A1g": 2,  # 0
    "E1ux": 1,  # 7
    "E1gy": 1,  # 3
    "E1gx": 1,  # 2
    "E1uy": 1,  # 6
    "E2gy": 1,  # 1
    "E2gx": 1,  # 0
    "E2uy": 1,  # 4
    "E2ux": 1,  # 5
}


if __name__ == "__main__":

    bondlength = [1.68]

    for BondLength in bondlength:

        Mol = pyscf.gto.Mole()
        Mol.atom = """
Cr     0.0000      0.0000  %f 
Cr     0.0000      0.0000  -%f 
""" % (
            BondLength / 2,
            BondLength / 2,
        )
        Mol.basis = "ccpvdz-dk"
        Mol.symmetry = "Dooh"
        Mol.spin = 2
        Mol.charge = 0
        Mol.verbose = 2
        Mol.unit = "angstorm"
        Mol.build()
        SCF = pyscf.scf.sfx2c(pyscf.scf.RHF(Mol))
        SCF.max_cycle = 32
        SCF.conv_tol = 1e-9
        SCF.run()

        # print(SCF.energy)

        DumpFileName = (
            "FCIDUMP"
            + "_"
            + "CR2"
            + "_"
            + "ccpvdz-dk"
            + "_"
            + str(int(BondLength * 100))
            # + "_SCF"
        )

        bond_int = int(BondLength * 100)

        Mol.spin = 0
        Mol.build()

        # iCISCF

        # the driver

        norb = 12
        nelec = 12
        CASSCF_Driver = pyscf.mcscf.CASSCF(SCF, norb, nelec)
        mo_init = pyscf.mcscf.sort_mo_by_irrep(
            CASSCF_Driver, CASSCF_Driver.mo_coeff, cas_space_symmetry
        )  # right!

        energy = CASSCF_Driver.kernel(mo_coeff=mo_init)[0]

        Mol.symmetry = "D2h"
        Mol.build()

        fock_mat = get_fock(CASSCF_Driver)

        ncore = CASSCF_Driver.ncore
        ncas = CASSCF_Driver.ncas
        nvir = mo_init.shape[1] - ncore - ncas
        nmo = mo_init.shape[1]

        print("core core part")
        print(fock_mat[:ncore, :ncore])
        print("core act  part")
        print(fock_mat[:ncore, ncore : ncore + ncas])
        print("core virt part")
        print(fock_mat[:ncore, ncore + ncas : nmo])
        print("act  act  part")
        print(fock_mat[ncore : ncore + ncas, ncore : ncore + ncas])
        print("act  virt part")
        print(fock_mat[ncore : ncore + ncas, ncore + ncas : nmo])
        print("virt virt part")
        print(fock_mat[ncore + ncas : nmo, ncore + ncas : nmo])

        # note the generalized Fock is Hermitian but not block-diagonal #
        
        # 这里的定义和 Helgaker 书里头不一样 #
        
        # 与 Theor Chem Acc (1997) 97:88-95 一样，见 eqn 4.3 #