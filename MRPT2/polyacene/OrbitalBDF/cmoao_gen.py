from pyscf_util.misc._parse_bdf_optgeom import *
from pyscf_util.misc._parse_bdf_chkfil import *
from pyscf_util.misc._parse_bdf_orbfile import *
import os
from pyscf import gto, scf, mcscf
from pyscf_util.misc.interface_BDF import *
from pyscf_util.File.file_cmoao import *

if __name__ == "__main__":

    tasks = []
    for task_type in range(2, 10):
        task_name = f"0{task_type}S"
        optgeom_file = f"{task_name}.optgeom"
        chkfil_file = f"{task_name}.chkfil"
        casorb_file = f"{task_name}.casorb"
        if (
            os.path.exists(optgeom_file)
            and os.path.exists(chkfil_file)
            and os.path.exists(casorb_file)
        ):
            tasks.append((task_name, optgeom_file, chkfil_file, casorb_file))
        task_name = f"0{task_type}T"
        optgeom_file = f"{task_name}.optgeom"
        chkfil_file = f"{task_name}.chkfil"
        casorb_file = f"{task_name}.casorb"
        if (
            os.path.exists(optgeom_file)
            and os.path.exists(chkfil_file)
            and os.path.exists(casorb_file)
        ):
            tasks.append((task_name, optgeom_file, chkfil_file, casorb_file))

    task_name = "10S"
    optgeom_file = f"{task_name}.optgeom"
    chkfil_file = f"{task_name}.chkfil"
    casorb_file = f"{task_name}.casorb"
    if (
        os.path.exists(optgeom_file)
        and os.path.exists(chkfil_file)
        and os.path.exists(casorb_file)
    ):
        tasks.append((task_name, optgeom_file, chkfil_file, casorb_file))

    task_name = "10T"
    optgeom_file = f"{task_name}.optgeom"
    chkfil_file = f"{task_name}.chkfil"
    casorb_file = f"{task_name}.casorb"
    if (
        os.path.exists(optgeom_file)
        and os.path.exists(chkfil_file)
        and os.path.exists(casorb_file)
    ):
        tasks.append((task_name, optgeom_file, chkfil_file, casorb_file))

    print(f"Found {len(tasks)} tasks.")

    ### for each task, read geometry ,ao2somat, casorb ###

    for task_name, optgeom_file, chkfil_file, casorb_file in tasks:

        print(f"Processing task {task_name}...")

        # Read molecular geometry from optgeom file
        mol_geometry = parse_geometry(optgeom_file)
        mol = gto.M(atom=mol_geometry, basis="ccpvtz", verbose=4, symmetry="d2h", unit="bohr")
        mol.build()

        mf = pyscf.scf.RHF(mol)

        output_fch = "tmp.fch"
        output_fch_new = "tmp_new.fch"
        output_scforb = "%s_new.scforb" % task_name

        mo_coeffs_bdf = convert_bdf_to_pyscf(
            mol,
            mf,
            chkfil_file,
            casorb_file,
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

        assert np.allclose(ovlp_mo, np.eye(ovlp_mo.shape[0]), atol=3e-7)

        print("Symmetry and orthogonality check passed.")

        orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeffs_bdf)

        # for irrep in range(len(mol.irrep_name)):
        #     print(f"Irrep {mol.irrep_name[irrep]} has {np.sum(orbsym == irrep)} orbitals.")

        print(orbsym)

        # os.system("rm %s" % output_fch)
        # os.system("rm %s" % output_fch_new)        

        ## dump cmoao ##

        Dump_Cmoao("%s_cmoao" % task_name, mo_coeffs_bdf)
