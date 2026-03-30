from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, Topology, ForceField
from openff.interchange import Interchange
import os
from pathlib import Path
from collections import defaultdict
from pprint import pprint
from shutil import which
from openff.interchange.drivers import get_gromacs_energies, get_lammps_energies
from openff.interchange.drivers.all import get_summary_data


def deduplicate(name, counts):
    idx = counts[name]
    counts[name] += 1
    return f'{name}_{idx}'


def read_sdf(input_sdf):
    with Chem.SDMolSupplier(input_sdf) as suppl:
        for mol in suppl:
            if mol is None:
                print('Invalid SDF')
                continue
            yield mol, mol.GetProp('_Name')


def workflow(mol, name):
    os.makedirs(name, exist_ok=True)
    output_sdf = os.path.join(name, f"{name}.sdf")

    with Chem.SDWriter(output_sdf) as out:
        mol.SetProp('_Name', name)
        out.write(mol)
    prepare_molecule(Path(output_sdf))


def prepare_molecule(mol_sdf):
    ligand = Molecule.from_file(mol_sdf)
    ligand.generate_conformers(n_conformers=1)

    force_field = ForceField("openff-2.3.0.offxml")
    interchange = Interchange.from_smirnoff(force_field, [ligand])

    interchange.to_gromacs(prefix=f"{mol_sdf.stem}")

    if which("lmp_serial"):
        pprint(get_lammps_energies(interchange).energies)

    if which("gmx"):
        pprint(get_gromacs_energies(interchange).energies)

    get_summary_data(interchange)


if __name__ == '__main__':
    input_dir = input()
    path = os.listdir(input_dir)
    for file in path:
        if '.sdf' in file:
            name_counts = defaultdict(int)
            for mol, name in read_sdf(Path(file)):
                mol_name = deduplicate(name, name_counts)
                workflow(mol, mol_name)
