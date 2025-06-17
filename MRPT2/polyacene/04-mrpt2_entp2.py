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
        irrep = 0
        spintwo = 0
        if "T" in task_name:
            task_str = "2 5 1 1"
            irrep = 5
            spintwo = 2

        ## canonicalize ##

        # kernel(
        #     IsCSF=True,
        #     task_name="polyacene_rdm1",
        #     fcidump=DumpFileName,
        #     segment=segment,
        #     nelec_val=nelec_val,
        #     rotatemo=0,
        #     cmin=CMIN,
        #     perturbation=0,
        #     dumprdm=1,
        #     relative=0,
        #     Task=task_str,
        #     inputocfg=0,
        #     etol=1e-10,
        #     selection=1,
        #     doublegroup=None,
        #     direct=None,
        #     start_with=None,
        #     end_with=[".csv"],
        # )
        
        CMIN = [5e-5, 3e-5, 1.5e-5, 9e-6, 7e-6]
        
        for cmin in CMIN:

            filename = f"{task_name}_{cmin:.1e}"

            _Generate_InputFile_iCI(
                filename,
                Segment=segment,    
                nelec_val=nelec_val,
                rotatemo=0,
                Task=task_str,
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

            prime_space_file = f"{task_name}.SpinTwo_{spintwo}_Irrep_{irrep}_{cmin:.3e}.PrimeSpace"

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
            