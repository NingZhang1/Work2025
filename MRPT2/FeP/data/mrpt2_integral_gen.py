from pyscf_util.misc._parse_bdf_optgeom import *
import os
from pyscf import gto, scf, mcscf
from pyscf_util.misc.interface_BDF import *
from pyscf_util.File.file_cmoao import *
from pyscf_util.Integrals.integral_MRPT2_incore_fast import *
from pyscf_util.Integrals.integral_CASCI import *

if __name__ == "__main__":

    CMOAO_NAME_FORMAT = "cas_%d_%d_%s_cmoao"

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

    ### first generate heff casci ###

    for dir_name, sub_task_name, casorb_file in task_info:

        tmp_dir = dir_name.split("_")

        nact = int(tmp_dir[2])
        nact_elec = int(tmp_dir[1])

        print(f"Processing task {dir_name}_{sub_task_name}...")

        cmoao_file = CMOAO_NAME_FORMAT % (nact_elec, nact, sub_task_name)
        cmoao = ReadIn_Cmoao(cmoao_file, mol.nao, mol.nao)

        ## build mf ##

        mf = scf.RHF(mol)
        
        ## build mcscf ##

        #nact = int(task_name[:2]) * 4 + 2
        #nact_elec = nact
        ncore = (mol.nelectron - nact_elec) // 2
        nvir = mol.nao - ncore - nact
        
        CASSCF_Driver = pyscf.mcscf.CASSCF(mf, nact, nact_elec)
        
        DumpFileName = f"FCIDUMP_{dir_name}_{sub_task_name}_casci"
        
        dump_heff_casci(
            mol,
            CASSCF_Driver,
            cmoao[:, :ncore],
            cmoao[:, ncore : ncore + nact],
            DumpFileName,
        )
        

    for dir_name, sub_task_name, casorb_file in task_info:

        # print(cmoao.shape)

        ## build mol ##

        tmp_dir = dir_name.split("_")

        nact = int(tmp_dir[2])
        nact_elec = int(tmp_dir[1])

        cmoao_file = CMOAO_NAME_FORMAT % (nact_elec, nact, sub_task_name)
        cmoao = ReadIn_Cmoao(cmoao_file, mol.nao, mol.nao)

        ## build mf ##

        mf = scf.RHF(mol)

        ## build mcscf ##

        #nact = int(task_name[:2]) * 4 + 2
        #nact_elec = nact
        ncore = (mol.nelectron - nact_elec) // 2
        nvir = mol.nao - ncore - nact

        ## build mrpt2 ##

        fcidump_mrpt2_incore_fast(
            mol,
            mf,
            cmoao,
            ncore,
            nact,
            nvir,
            filename=f"FCIDUMP_{dir_name}_{sub_task_name}_mrpt2",
            tol=1e-10,
        )
