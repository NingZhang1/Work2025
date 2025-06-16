from pyscf_util.misc._parse_bdf_optgeom import *
import os
from pyscf import gto, scf, mcscf

# from pyscf_util.misc.interface_BDF import *
from pyscf_util.File.file_cmoao import *
from pyscf_util.Integrals.integral_MRPT2 import get_generalized_fock
from pyscf_util.Integrals.integral_MRPT2_incore_fast import *
from pyscf_util.Integrals.integral_CASCI import *
import numpy as np
from pyscf_util.File.file_cmoao import *
from pyscf_util.iCIPT2.iCIPT2 import kernel

if __name__ == "__main__":

    CMOAO_NAME_FORMAT = "../cas_%d_%d_%s_cmoao"

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

    mol_geometry = parse_geometry("../FeP.optgeom")
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

    chkfil_file = "../FeP.chkfil"

    task_str = {
        "3E1": "2 1 1 1",
        "3E2": "2 3 1 1",
        "triplet": "2 2 1 1",
        "quintet": "4 0 1 1",
    }

    ### first generate heff casci ###

    for dir_name, sub_task_name, casorb_file in task_info:

        tmp_dir = dir_name.split("_")

        nact = int(tmp_dir[2])
        nact_elec = int(tmp_dir[1])

        print(f"Processing task {dir_name}_{sub_task_name}...")

        cmoao_file = CMOAO_NAME_FORMAT % (nact_elec, nact, sub_task_name)
        cmoao = ReadIn_Cmoao(cmoao_file, mol.nao, mol.nao)
        
        ## build mcscf ##

        ncore = (mol.nelectron - nact_elec) // 2
        nvir = mol.nao - ncore - nact
        
        mf.mo_coeff = cmoao
        CASSCF_Driver = pyscf.mcscf.CASSCF(mf, nact, nact_elec)
        
        DumpFileName = f"../FCIDUMP_{dir_name}_{sub_task_name}_casci"

        segment = "0 %d 6 6 %d 0" % (
            (nact_elec - 12) // 2,
            nact - 12 - (nact_elec - 12) // 2,
        )
        nelec_val = 12

        CMIN = "1e-5"

        task = task_str[sub_task_name]

        ## canonicalize ##

        kernel(
            IsCSF=True,
            task_name=f"FeP_{dir_name}_{sub_task_name}",
            fcidump=DumpFileName,
            segment=segment,
            nelec_val=nelec_val,
            rotatemo=0,
            cmin=CMIN,
            perturbation=0,
            dumprdm=1,
            relative=0,
            Task=task,
            inputocfg=0,
            etol=1e-10,
            selection=1,
            doublegroup=None,
            direct=None,
            start_with=None,
            end_with=[".csv"],
        )

        os.system("mv rdm1.csv FeP_rdm1.csv")

        mo_coeff = CASSCF_Driver.mo_coeff
        rdm1 = file_rdm.ReadIn_rdm1("FeP_rdm1", nact, nact)

        # print(rdm1)
        gfock = get_generalized_fock(CASSCF_Driver, mo_coeff, rdm1)

        # Diagonalize gfock in core and virtual spaces
        core_fock = gfock[:ncore, :ncore]
        virtual_fock = gfock[ncore + nact :, ncore + nact :]

        # Get eigenvalues and eigenvectors for core and virtual spaces
        core_eigvals, core_eigvecs = np.linalg.eigh(core_fock)
        virtual_eigvals, virtual_eigvecs = np.linalg.eigh(virtual_fock)

        # Update mo_coeff with new eigenvectors
        # mo_coeff[:, :ncore] = core_eigvecs
        mo_coeff[:, :ncore] = mo_coeff[:, :ncore] @ core_eigvecs
        mo_coeff[:, ncore + nact :] = mo_coeff[:, ncore + nact :] @ virtual_eigvecs

        # Dump updated mo_coeff to file
        Dump_Cmoao(f"FeP_{dir_name}_{sub_task_name}_canonicalized_orb_cmoao", mo_coeff)

        # Transform gfock to a new basis
        # The new basis is the canonicalized MO coefficients

        trans_mat_old_2_new = np.eye(mol.nao)
        trans_mat_old_2_new[:ncore, :ncore] = core_eigvecs
        trans_mat_old_2_new[ncore + nact :, ncore + nact :] = virtual_eigvecs

        transformed_gfock = trans_mat_old_2_new.T @ gfock @ trans_mat_old_2_new

        # Save the transformed Fock matrix
        # np.savetxt(f"{task_name}_transformed_gfock.txt", transformed_gfock)

        Dump_Cmoao(f"FeP_{dir_name}_{sub_task_name}_gfock", transformed_gfock)

        # Analyze the transformed matrix
        # Get diagonal elements (orbital energies) and sort them
        orbital_energies = np.diag(transformed_gfock)
        sorted_indices = np.argsort(orbital_energies)
        sorted_energies = orbital_energies[sorted_indices]

        # Split orbital energies into core, active, and virtual orbitals
        core_orbitals = sorted_energies[:ncore]
        active_orbitals = sorted_energies[ncore : ncore + nact]
        virtual_orbitals = sorted_energies[ncore + nact :]

        # Print analysis results
        print(f"\nAnalysis for FeP_{dir_name}_{sub_task_name}:")
        print("=" * 50)
        print("Core orbital energies (eV):")
        print(core_orbitals * 27.2114)  # Convert to eV
        print("\nActive orbital energies (eV):")
        print(active_orbitals * 27.2114)
        print("\nVirtual orbital energies (eV):")
        print(virtual_orbitals * 27.2114)
