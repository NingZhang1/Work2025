from pyscf_util.misc._parse_bdf_optgeom import *
import os
from pyscf import gto, scf, mcscf
from pyscf_util.misc.interface_BDF import *
from pyscf_util.File.file_cmoao import *
from pyscf_util.Integrals.integral_MRPT2_incore_fast import *
from pyscf_util.Integrals.integral_CASCI import *

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

    ### first generate heff casci ###
    
    for task_name in tasks:

        optgeom_file = f"{task_name}.optgeom"
        mol_geometry = parse_geometry(optgeom_file)
        mol = gto.M(
            atom=mol_geometry, basis="ccpvtz", verbose=4, symmetry="d2h", unit="bohr"
        )
        mol.build()

        cmoao_file = CMOAO_NAME_FORMAT % task_name
        cmoao = ReadIn_Cmoao(cmoao_file, mol.nao, mol.nao)

        ## build mf ##

        mf = scf.RHF(mol)
        
        ## build mcscf ##

        ncas = int(task_name[:2]) * 4 + 2
        neleccas = ncas
        ncore = (mol.nelectron - neleccas) // 2
        nvir = mol.nao - ncore - ncas
        
        CASSCF_Driver = pyscf.mcscf.CASSCF(mf, ncas, neleccas)
        
        DumpFileName = f"FCIDUMP_{task_name}_casci"
        
        dump_heff_casci(
            mol,
            CASSCF_Driver,
            cmoao[:, :ncore],
            cmoao[:, ncore : ncore + ncas],
            DumpFileName,
        )
        

    for task_name in tasks:

        # print(cmoao.shape)

        ## build mol ##

        optgeom_file = f"{task_name}.optgeom"
        mol_geometry = parse_geometry(optgeom_file)
        mol = gto.M(
            atom=mol_geometry, basis="ccpvtz", verbose=4, symmetry="d2h", unit="bohr"
        )
        mol.build()

        cmoao_file = CMOAO_NAME_FORMAT % task_name
        cmoao = ReadIn_Cmoao(cmoao_file, mol.nao, mol.nao)

        ## build mf ##

        mf = scf.RHF(mol)
        # mf.kernel()

        ## build mcscf ##

        ncas = int(task_name[:2]) * 4 + 2
        neleccas = ncas
        ncore = (mol.nelectron - neleccas) // 2
        nvir = mol.nao - ncore - ncas

        ## build mrpt2 ##

        fcidump_mrpt2_incore_fast(
            mol,
            mf,
            cmoao,
            ncore,
            ncas,
            nvir,
            filename=f"FCIDUMP_{task_name}_mrpt2",
            tol=1e-10,
        )
