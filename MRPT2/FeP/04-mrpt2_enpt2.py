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
from pyscf_util.misc.icipt2_inputfile_generator import _Generate_InputFile_iCI

MRPT2_ENPT2 = ""

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

    spintwo = {
        "3E1" : 2,
        "3E2" : 2,
        "triplet" : 2,
        "quintet" : 4,
    }

    irrep = {
        "3E1" : 1,
        "3E2" : 3,
        "triplet" : 2,
        "quintet" : 0,
    }

    ### first generate heff casci ###
    
    CMIN = [5e-5, 3e-5, 1.5e-5, 9e-6, 7e-6]

    for dir_name, sub_task_name, casorb_file in task_info:

        tmp_dir = dir_name.split("_")

        nact = int(tmp_dir[2])
        nact_elec = int(tmp_dir[1])
        
        ncore = (mol.nelectron - nact_elec) // 2
        nvir = mol.nao - nact - ncore

        print(f"Processing task {dir_name}_{sub_task_name}...")

        cmoao_file = CMOAO_NAME_FORMAT % (nact_elec, nact, sub_task_name)
        cmoao = ReadIn_Cmoao(cmoao_file, mol.nao, mol.nao)
        
        ## build mcscf ##

        ncore = (mol.nelectron - nact_elec) // 2
        nvir = mol.nao - ncore - nact
        
        mf.mo_coeff = cmoao
        CASSCF_Driver = pyscf.mcscf.CASSCF(mf, nact, nact_elec)
        
        DumpFileName = f"../FCIDUMP_{dir_name}_{sub_task_name}_mrpt2"

        segment = "%d %d 6 6 %d %d" % (
            ncore,
            (nact_elec - 12) // 2,
            nact - 12 - (nact_elec - 12) // 2,
            nvir
        )
        nelec_val = 12

        # CMIN = "1e-5"

        task = task_str[sub_task_name]

        ## canonicalize ##

        # kernel(
        #     IsCSF=True,
        #     task_name=f"FeP_{dir_name}_{sub_task_name}",
        #     fcidump=DumpFileName,
        #     segment=segment,
        #     nelec_val=nelec_val,
        #     rotatemo=0,
        #     cmin=CMIN,
        #     perturbation=0,
        #     dumprdm=1,
        #     relative=0,
        #     Task=task,
        #     inputocfg=0,
        #     etol=1e-10,
        #     selection=1,
        #     doublegroup=None,
        #     direct=None,
        #     start_with=None,
        #     end_with=[".csv"],
        # )

        for cmin in CMIN:

            filename = f"FeP_{dir_name}_{sub_task_name}_{cmin:.1e}"

            _Generate_InputFile_iCI(
                filename,
                Segment=segment,    
                nelec_val=nelec_val,
                rotatemo=0,
                Task=task,
                cmin=cmin,
                inputocfg=0,
                perturbation=0,
                etol=1e-10,
                selection=0,
                doublegroup=None,
                direct=None,
                dumprdm=0,
                relative=0
            )

            ### build primespace file ###

            prime_space_file = f"FeP_{dir_name}_{sub_task_name}.SpinTwo_{spintwo[sub_task_name]}_Irrep_{irrep[sub_task_name]}_{cmin:.3e}.PrimeSpace"

            print("processing prime space file: ", prime_space_file)

            # os.system(f"os {prime_space_file} PrimeSpace")

            ### read contents of prime_space_file ###

            with open(prime_space_file, "r") as f:
                contents = f.readlines()

            ### write contents to PrimeSpace ###

            with open("PrimeSpace", "w") as f:
                f.write(contents[0])
                # for the remianing lines, add ncore "2" before each line
                prefix = "2" * ncore
                for line in contents[1:]:
                    f.write(f"{prefix}{line}")

            ### run mrpt2_enpt2 ###

            os.system(f"{MRPT2_ENPT2} {filename} {DumpFileName} PrimeSpace 1> {filename}.out 2> {filename}.err")

            os.system(f"rm PrimeSpace")
            