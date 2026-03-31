from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, Topology, ForceField
from openff.interchange import Interchange
import os
from pathlib import Path
from collections import defaultdict
from pprint import pprint
from shutil import which
import numpy as np
from openff.units import unit
from openff.interchange.drivers import get_gromacs_energies, get_lammps_energies


def deduplicate(name, counts):
    idx = counts[name]
    counts[name] += 1
    return f'{name}_{idx}' if idx else f'{name}'


def generate_3d(mol, name):
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    success = AllChem.EmbedMolecule(mol, params)
    if success == -1:
        print(f'Error: RDKit couldn\'t generate 3D for {name}')


def read_sdf(input_sdf):
    with Chem.SDMolSupplier(input_sdf) as suppl:
        for mol in suppl:
            if mol is None:
                continue
            yield mol, mol.GetProp('_Name')


def prepare_molecule(mol_from_sdf, name):
    mol_h = Chem.AddHs(mol_from_sdf)
    if mol_h.GetNumConformers() == 0:
        generate_3d(mol_h, name)
        AllChem.MMFFOptimizeMolecule(mol_h)

    ligand = Molecule.from_rdkit(mol_h)
    ligand.name = name
    print(f"Preparing {name}")

    force_field = ForceField("openff-2.1.0.offxml")
    try:
        interchange = force_field.create_interchange(ligand.to_topology())
        print(f'Interchange for {name} is created')
        interchange.box = [4, 4, 4] * unit.nanometer
        print(f'Box for {name} is created')
        os.makedirs(name, exist_ok=True)
        output_path = os.path.join(name, name)
        interchange.to_gromacs(prefix=str(output_path))
        
        old_top = os.path.join(name, f"{name}.top")
        new_top = os.path.join(name, "topol.top")
        
        if os.path.exists(old_top):
            os.rename(old_top, new_top)
        
        print(f'Ligand {name} is prepared for MD simulation!')

        if which("lmp_serial"):
            print(get_lammps_energies(interchange).energies)
            print(f'Lammps energies for {name}')

       # if which("gmx"):
       #     pprint(get_gromacs_energies(interchange).energies)

    except Exception as e:
        print(f'Error: {e} \nCouldn\'t prepare {name}')
        

import subprocess

def validate_with_grompp(name):
    mdp_path = os.path.join(f"{name}_pointenergy.mdp")
    output_dir = name
    
    result = subprocess.run(
        ["gmx", "grompp",
         "-f", mdp_path,
         "-c", f"{name}.gro",      
         "-p", "topol.top",         
         "-o", "validate.tpr",      
         "-maxwarn", "1"],         
        cwd=output_dir,  
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"Validation passed for {name}")
    else:
        print(f"Validation failed for {name}")
        print(result.stderr)


if __name__ == '__main__':
    input_dir = input()
    path = os.listdir(input_dir)
    for file in path:
        if '.sdf' in file:
            name_counts = defaultdict(int)
            for mol, name in read_sdf(Path(file)):
                mol_name = deduplicate(name, name_counts)
                os.makedirs(mol_name, exist_ok=True)
                prepare_molecule(mol, mol_name)
                # validate_with_grompp(name)
