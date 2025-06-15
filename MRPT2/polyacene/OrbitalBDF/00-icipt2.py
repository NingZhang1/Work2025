from pyscf_util.misc._parse_bdf_optgeom import *
import os
from pyscf import gto, scf, mcscf
from pyscf_util.misc.interface_BDF import *
from pyscf_util.File.file_cmoao import *
from pyscf_util.Integrals.integral_MRPT2_incore_fast import *
from pyscf_util.Integrals.integral_CASCI import *
from pyscf_util.iCIPT2.iCIPT2 import kernel


if __name__ == "__main__":

    CMOAO_NAME_FORMAT = "%s_cmoao"

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

    CMIN = "5e-5 3e-5 1.5e-5 9e-6 7e-6"

    ### first generate heff casci ###
    
    for task_name in tasks:

        optgeom_file = f"../{task_name}.optgeom"
        mol_geometry = parse_geometry(optgeom_file)
        mol = gto.M(
            atom=mol_geometry, basis="ccpvtz", verbose=4, symmetry="d2h", unit="bohr"
        )
        mol.build()

        cmoao_file = CMOAO_NAME_FORMAT % task_name
        cmoao = ReadIn_Cmoao("../" + cmoao_file, mol.nao, mol.nao)

        ## build mf ##

        mf = scf.RHF(mol)
        
        ## build mcscf ##

        ncas = int(task_name[:2]) * 4 + 2
        neleccas = ncas
        ncore = (mol.nelectron - neleccas) // 2
        nvir = mol.nao - ncore - ncas
        
        CASSCF_Driver = pyscf.mcscf.CASSCF(mf, ncas, neleccas)
        
        DumpFileName = f"FCIDUMP_{task_name}_casci"

        # icipt2 #

        segment = "0 %d 5 5 %d 0" % (
            (neleccas - 10) // 2,
            ncas - 10 - (neleccas - 10) // 2,
        )
        nelec_val = 10

        task_str = "0 0 1 1"
        if "T" in task_name:
            task_str = "2 5 1 1"

        kernel(
            IsCSF=True,
            task_name=f"{task_name}_casci",
            fcidump="../" + DumpFileName,
            segment=segment,
            nelec_val=nelec_val,
            rotatemo=0,
            perturbation=1,
            Task=task_str,
            etol=1e-8,
            cmin=CMIN,
        )
