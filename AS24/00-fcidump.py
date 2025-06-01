from geometry import *

import pyscf
from pyscf import tools
from pyscf import symm
from pyscf.tools import fcidump
import pyscf.mcscf


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

    BASIS = ["aug-cc-pvdz", "aug-cc-pvtz"]

    for case_str in GEOMETRY.keys():

        for mole_str in GEOMETRY[case_str].keys():

            for basis in BASIS:

                # build mole #

                Mol = pyscf.gto.Mole()
                Mol.atom = GEOMETRY[case_str][mole_str]
                Mol.basis = basis
                Mol.symmetry = True
                Mol.spin = 0
                Mol.charge = 0
                Mol.verbose = 4
                Mol.unit = "angstorm"
                Mol.build()

                # run scf #

                SCF = pyscf.scf.RHF(Mol)
                SCF.max_cycle = 32
                SCF.conv_tol = 1e-9
                SCF.run()

                # dump fci #

                DumpFileName = "FCIDUMP" + "_" + case_str + "_" + mole_str + "_" + basis

                fcidump.from_scf(SCF, DumpFileName, tol=1e-11)
