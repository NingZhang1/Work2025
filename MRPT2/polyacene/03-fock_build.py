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

    CMOAO_NAME_FORMAT = "../%s_cmoao"

    tasks = [
        "02S",
        "02T",
        "03S",
        "03T",
        "04S",
        "04T",
        "05S",
        "05T",
        "06S",
        "06T",
        "07S",
        "07T",
        "08S",
        "08T",
        "09S",
        "09T",
        "10S",
        "10T",
    ]

    ### first generate heff casci ###

    for task_name in tasks:

        optgeom_file = f"../{task_name}.optgeom"
        mol_geometry = parse_geometry(optgeom_file)
        mol = gto.M(
            atom=mol_geometry, basis="ccpvtz", verbose=4, symmetry="d2h", unit="bohr"
        )
        mol.build()

        cmoao_file = CMOAO_NAME_FORMAT % task_name
        cmoao = ReadIn_Cmoao(cmoao_file, mol.nao, mol.nao)

        ## build mf ##

        mf = scf.RHF(mol)
        mf.mo_coeff = cmoao

        ## build mcscf ##

        ncas = int(task_name[:2]) * 4 + 2
        neleccas = ncas
        ncore = (mol.nelectron - neleccas) // 2
        nvir = mol.nao - ncore - ncas

        CASSCF_Driver = pyscf.mcscf.CASSCF(mf, ncas, neleccas)

        ## build fock ##

        ### read cascidump ###

        DumpFileName = f"../FCIDUMP_{task_name}_casci"

        segment = "0 %d 5 5 %d 0" % (
            (neleccas - 10) // 2,
            ncas - 10 - (neleccas - 10) // 2,
        )
        nelec_val = 10

        CMIN = "1e-5"

        if neleccas <= 14:
            CMIN = "0.0"  # FCI

        task_str = "0 0 1 1"
        if "T" in task_name:
            task_str = "2 5 1 1"

        ## canonicalize ##

        kernel(
            IsCSF=True,
            task_name="polyacene_rdm1",
            fcidump=DumpFileName,
            segment=segment,
            nelec_val=nelec_val,
            rotatemo=0,
            cmin=CMIN,
            perturbation=0,
            dumprdm=1,
            relative=0,
            Task=task_str,
            inputocfg=0,
            etol=1e-10,
            selection=1,
            doublegroup=None,
            direct=None,
            start_with=None,
            end_with=[".csv"],
        )

        os.system("mv rdm1.csv polyacene_rdm1.csv")

        mo_coeff = CASSCF_Driver.mo_coeff
        rdm1 = file_rdm.ReadIn_rdm1("polyacene_rdm1", ncas, ncas)

        # print(rdm1)
        gfock = get_generalized_fock(CASSCF_Driver, mo_coeff, rdm1)

        # Diagonalize gfock in core and virtual spaces
        core_fock = gfock[:ncore, :ncore]
        virtual_fock = gfock[ncore + ncas :, ncore + ncas :]

        # Get eigenvalues and eigenvectors for core and virtual spaces
        core_eigvals, core_eigvecs = np.linalg.eigh(core_fock)
        virtual_eigvals, virtual_eigvecs = np.linalg.eigh(virtual_fock)

        # Update mo_coeff with new eigenvectors
        # mo_coeff[:, :ncore] = core_eigvecs
        mo_coeff[:, :ncore] = mo_coeff[:, :ncore] @ core_eigvecs
        mo_coeff[:, ncore + ncas :] = mo_coeff[:, ncore + ncas :] @ virtual_eigvecs

        # Dump updated mo_coeff to file
        Dump_Cmoao(f"{task_name}_canonicalized_orb_cmoao", mo_coeff)

        # Transform gfock to a new basis
        # The new basis is the canonicalized MO coefficients

        trans_mat_old_2_new = np.eye(mol.nao)
        trans_mat_old_2_new[:ncore, :ncore] = core_eigvecs
        trans_mat_old_2_new[ncore + ncas :, ncore + ncas :] = virtual_eigvecs

        transformed_gfock = trans_mat_old_2_new.T @ gfock @ trans_mat_old_2_new

        # Save the transformed Fock matrix
        # np.savetxt(f"{task_name}_transformed_gfock.txt", transformed_gfock)

        Dump_Cmoao(f"{task_name}_gfock", transformed_gfock)

        # Analyze the transformed matrix
        # Get diagonal elements (orbital energies) and sort them
        orbital_energies = np.diag(transformed_gfock)
        sorted_indices = np.argsort(orbital_energies)
        sorted_energies = orbital_energies[sorted_indices]

        # Split orbital energies into core, active, and virtual orbitals
        core_orbitals = sorted_energies[:ncore]
        active_orbitals = sorted_energies[ncore : ncore + ncas]
        virtual_orbitals = sorted_energies[ncore + ncas :]

        # Print analysis results
        print(f"\nAnalysis for {task_name}:")
        print("=" * 50)
        print("Core orbital energies (eV):")
        print(core_orbitals * 27.2114)  # Convert to eV
        print("\nActive orbital energies (eV):")
        print(active_orbitals * 27.2114)
        print("\nVirtual orbital energies (eV):")
        print(virtual_orbitals * 27.2114)
