from geometry import *

import pyscf
from pyscf import tools
from pyscf import symm
from pyscf.tools import fcidump
import pyscf.mcscf
from pyscf_util.iCIPT2.iCIPT2 import kernel


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

    CMIN = "1e-4 7e-5 5e-5 3e-5 1.5e-5 9e-6"

    # BASIS = ["aug-cc-pvdz", "aug-cc-pvtz"]

    BASIS = ["aug-cc-pvdz"]

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
                Mol.verbose = 2
                Mol.unit = "angstorm"
                Mol.build()

                # run scf #

                SCF = pyscf.scf.RHF(Mol)
                SCF.max_cycle = 32
                SCF.conv_tol = 1e-9
                # SCF.run()

                # dump fci #

                DumpFileName = "FCIDUMP" + "_" + case_str + "_" + mole_str + "_" + basis

                # fcidump.from_scf(SCF, DumpFileName, tol=1e-11)

                print(Mol.nelectron)
                nelec = int(Mol.nelectron)
                nfzc = CONFIG[case_str][mole_str]["nfzc"]
                nelec_corr = nelec - nfzc * 2
                print(nelec_corr)

                if nelec_corr <= 12:
                    nelec_val = nelec_corr
                    segment = "%d 0 %d %d %d 0" % (
                        nfzc,
                        nelec_val // 2,
                        12 - nelec_val // 2,
                        Mol.nao - nfzc - 12,
                    )
                else:
                    nelec_val = 12
                    ncor = (nelec_corr - nelec_val) // 2
                    segment = "%d %d %d %d %d 0" % (
                        nfzc,
                        ncor,
                        nelec_val // 2,
                        12 - nelec_val // 2,
                        Mol.nao - nfzc - ncor - 12,
                    )

                print(segment)

                task_name = case_str + "_" + mole_str + "_" + basis

                print(task_name)

                kernel(
                    IsCSF=True,
                    task_name=task_name,
                    fcidump="../" + DumpFileName,
                    segment=segment,
                    nelec_val=nelec_val,
                    rotatemo=1,
                    perturbation=1,
                    Task="0 0 1 1",
                    etol=1e-9,
                    cmin=CMIN,
                )
