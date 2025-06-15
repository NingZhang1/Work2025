from pyscf_util.misc._parse_bdf_optgeom import *
from pyscf_util.misc._parse_bdf_chkfil import *
from pyscf_util.misc._parse_bdf_orbfile import *
import os
from pyscf import gto, scf, mcscf, tools
from pyscf_util.misc.interface_BDF import *
from pyscf_util.File.file_cmoao import *

if __name__ == "__main__":

    sub_task_names = ["3E1", "3E2", "triplet", "quintet"]

    active_spaces = [(14, 18), (24, 28), (30, 34), (32, 35), (40, 42), (56, 58)]

    task_info = []

    for active_space in active_spaces:

        dir_name = f"CAS_{active_space[0]}_{active_space[1]}"

        for sub_task_name in sub_task_names:

            casorb_file = f"CAS-dz-{sub_task_name}.casorb"

            task_info.append((dir_name, sub_task_name, casorb_file))

    print(task_info)

    ### build molecule ###

    mol_geometry = parse_geometry("FeP.optgeom")
    mol = gto.M(
        atom=mol_geometry,
        basis="ccpvdz",
        verbose=4,
        symmetry="d2h",
        unit="bohr",
        spin=4,
    )
    mol.build()

    mf = pyscf.scf.RHF(mol)

    chkfil_file = "FeP.chkfil"

    ### for each task, read geometry ,ao2somat, casorb ###

    for dir_name, sub_task_name, casorb_file in task_info:

        tmp_dir = dir_name.split("_")

        nact = int(tmp_dir[2])
        nact_elec = int(tmp_dir[1])

        print(f"Processing task {dir_name}_{sub_task_name}...")

        output_fch = f"tmp.fch"
        output_fch_new = f"tmp_new.fch"
        output_scforb = f"{sub_task_name}_new.casorb"

        mo_coeffs_bdf = convert_bdf_to_pyscf(
            mol,
            mf,
            chkfil_file,
            f"FePOrb/{dir_name}/{casorb_file}",
            output_fch,
            output_fch_new,
            output_scforb,
            True,
            True,
        )

        ## check symmetry and orthogonality ##

        ovlp = mol.intor("int1e_ovlp")
        ovlp_mo = mo_coeffs_bdf.T @ ovlp @ mo_coeffs_bdf

        # print diagonal elements of ovlp_mo #

        print(np.diag(ovlp_mo))

        assert np.allclose(ovlp_mo, np.eye(ovlp_mo.shape[0]), atol=2e-7)

        print("Symmetry and orthogonality check passed.")

        orbsym = pyscf.symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mo_coeffs_bdf
        )

        print(orbsym)

        # ncore = (mol.nelectron - nact_elec) // 2
        # print(orbsym[:ncore])
        # print(orbsym[ncore : ncore + nact])
        # # print(orbsym[ncore + nact :])

        ## dump cmoao ##

        Dump_Cmoao(
            "cas_%d_%d_%s_cmoao" % (nact_elec, nact, sub_task_name), mo_coeffs_bdf
        )

        # tools.molden.from_mo(mol, "tmp.molden", mo_coeffs_bdf, symm=orbsym)
