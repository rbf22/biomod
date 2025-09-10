"""Structure classes."""

import numpy as np
import rmsd
import math
import warnings
from scipy.spatial.distance import cdist
from collections import Counter, defaultdict
from .base import StructureClass, query, StructureSet
from .constants import (
 COVALENT_RADII, METALS, ATOMIC_NUMBER, PERIODIC_TABLE,
 CHI_ANGLES, CODES, FULL_NAMES
)

class AtomStructure:
    """A structure made of atoms. This contains various useful methods that rely
    on a ``atoms()`` method, which the inheriting object must supply itself. All
    atomic structures also have IDs and names.

    The class would never be instantiated directly."""

    def __init__(self, id=None, name=None):
        """Creates an AtomStructure.

        :param id: The structure's ID.
        :param str name: The structure's name."""

        self._id, self._name = id, name


    def __eq__(self, other):
        """Checks if this atomic structure is equal to another.

        Two atomic structures are equal if every pairwise atom in their pairing
        are equal.

        :param AtomStructure other: the structure to compare to.
        :rtype: ``bool``"""

        try:
            mapping = self.pairing_with(other)
            for atom1, atom2 in mapping.items():
                if not atom1 == atom2:
                    return False
            return True
        except Exception:
            return False


    def __hash__(self):
        """Returns the hash of the structure, which is its memory address."""

        return id(self)


    @property
    def id(self):
        """The structure's unique ID.

        :rtype: ``str``"""

        return self._id


    @property
    def name(self):
        """The structure's name.

        :rtype: ``str``"""

        return self._name


    @name.setter
    def name(self, name):
        self._name = name


    @property
    def mass(self):
        """The structure's mass - the sum of all its atoms' masses.

        :rtype: ``float``"""

        return round(sum([atom.mass for atom in self.atoms()]), 12)


    @property
    def charge(self):
        """The structure's charge - the sum of all its atoms' charges.

        :rtype: ``float``"""

        return round(sum([atom.charge for atom in self.atoms()]), 12)


    @property
    def formula(self):
        """The structure's formula as a ``Counter`` dictionary - the count of
        all its atoms' elements.

        :rtype: ``Counter``"""

        return Counter([atom.element for atom in self.atoms()])


    @property
    def center_of_mass(self):
        """Returns the center of mass of the structure. This is the average of
        all the atom coordinates, weighted by the mass of each atom.

        :rtype: ``tuple``"""

        mass = self.mass
        locations = np.array([a._location * a.mass for a in self.atoms()])
        return np.sum(locations, axis=0) / mass


    @property
    def radius_of_gyration(self):
        """The radius of gyration of a structure is a measure of how extended it
        is. It is the root mean square deviation of the atoms' distance from the
        structure's :py:meth:`.center_of_mass`.

        :rtype: ``float``"""

        center_of_mass = self.center_of_mass
        atoms = self.atoms()
        square_deviation = sum(
         [atom.distance_to(center_of_mass) ** 2 for atom in atoms]
        )
        mean_square_deviation = square_deviation / len(atoms)
        return np.sqrt(mean_square_deviation)


    def pairing_with(self, structure):
        """Takes another structure with the same number of atoms as this one,
        and attempts to find the nearest equivalent of every atom in this
        structure, in that structure.

        Atoms will be aligned first by ID (if equal), then element, then by
        name, and finally by memory address - this last metric is
        used to ensure that even when allocation is essentially random, it is at
        least the same every time two structures are aligned.

        :param AtomStructure structure: the structure to pair with.
        :raises ValueError: if the other structure has a different number of atoms.
        :rtype: ``dict``"""

        atoms = self.atoms()
        other_atoms = structure.atoms()
        if len(atoms) != len(other_atoms):
            raise ValueError("{} and {} have different numbers of atoms".format(
             self, structure
            ))
        pair = {}
        common_ids = set(a._id for a in atoms) & set(a._id for a in other_atoms)
        id_atoms = {a._id: a for a in atoms}
        id_other_atoms = {a._id: a for a in other_atoms}
        for id_ in common_ids:
            pair[id_atoms[id_]] = id_other_atoms[id_]
            atoms.remove(id_atoms[id_])
            other_atoms.remove(id_other_atoms[id_])
        atoms, other_atoms = list(atoms), list(other_atoms)
        for atom_list in atoms, other_atoms:
            atom_list.sort(key=lambda a: (
             a._id, a._element, a._name, id(a)
            ))
        return {**pair, **{a1: a2 for a1, a2 in zip(atoms, other_atoms)}}


    def rmsd_with(self, structure):
        """Calculates the Root Mean Square Deviation between this structure and
        another.

        :param AtomStructure structure: the structure to check against.
        :raises ValueError: if the other structure has a different number of atoms.
        :rtype: ``float``"""

        pairing = self.pairing_with(structure)
        coords1, coords2 = [[a.location for a in atoms]
         for atoms in zip(*pairing.items())]
        c1, c2 = self.center_of_mass, structure.center_of_mass
        coords1 = [[x - c1[0], y - c1[1], z - c1[2]] for x, y, z in coords1]
        coords2 = [[x - c2[0], y - c2[1], z - c2[2]] for x, y, z in coords2]
        return round(rmsd.kabsch_rmsd(coords1, coords2), 12)


    def create_grid(self, size=1, margin=0):
        """A generator which models a grid around the structure and returns the
        coordinates of all the points in that grid. The origin is always one of
        those points, and the grid will be a box.

        :param int size: The spacing between grid points. The default is 1.
        :param int margin: How far to extend the grid beyond the structure coordinates. The default is 0.
        :rtype: ``tuple``"""

        atom_locations = [atom.location for atom in self.atoms()]
        dimension_values = []
        for dimension in range(3):
            coordinates = [loc[dimension] for loc in atom_locations]
            min_, max_ = min(coordinates) - margin, max(coordinates) + margin
            values = [0]
            while values[0] > min_:
                values.insert(0, values[0] - size)
            while values[-1] < max_:
                values.append(values[-1] + size)
            dimension_values.append(values)
        for x in dimension_values[0]:
            for y in dimension_values[1]:
                for z in dimension_values[2]:
                    yield (x, y, z)


    def check_ids(self):
        """Looks through all the structure's sub-structures and raises a
        warning if they have duplicate ID."""

        for objects in ("chains", "ligands", "waters", "residues", "atoms"):
            try:
                ids = [obj.id for obj in getattr(self, objects)()]
                unique_ids = set(ids)
                if len(ids) != len(unique_ids):
                    warnings.warn(f"{objects} have duplicate IDs")
            except AttributeError:
                pass


    def save(self, path):
        """Saves the structure to file. The file extension given in the filename
        will be used to determine which file format to save in.

        If the structure you are saving has any duplicate IDs, a warning will be
        issued, as the file saved will likely be nonsensical.

        :param str path: the filename and location to save to."""

        from .utilities import save
        self.check_ids()
        ext = path.split(".")[-1]
        if ext == "cif":
            from .mmcif import structure_to_mmcif_string
            string = structure_to_mmcif_string(self)
        elif ext == "mmtf":
            from .mmtf import structure_to_mmtf_string
            string = structure_to_mmtf_string(self)
        elif ext == "pdb":
            from .pdb import structure_to_pdb_string
            string = structure_to_pdb_string(self)
        else:
            raise ValueError("Unsupported file extension: " + ext)
        save(string, path)


    def atoms_in_sphere(self, location, radius, *args, **kwargs):
        """Returns all the atoms in a given sphere within this structure. This
        will be a lot faster if the structure is a :py:class:`.Model` and if
        :py:meth:`.optimise_distances` has been called, as it won't have to
        search all atoms.

        :param tuple location: the centre of the sphere.
        :param float radius: the radius of the sphere.
        :rtype: ``set``"""

        if "_internal_grid" in self.__dict__ and self._internal_grid:
            r, atoms = math.ceil(radius / 10), set()
            x, y, z = [int(math.floor(n / 10)) * 10 for n in location]
            x_range, y_range, z_range = [
             [(val - (n * 10)) for n in range(1, r + 1)][::-1] + [val] + [
              (val + n * 10) for n in range(1, r + 1)
             ] for val in (x, y, z)
            ]
            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        atoms = atoms.union(self._internal_grid[x][y][z])
            atoms = StructureSet(*atoms)
            atoms = query(lambda self: atoms)(self, *args, **kwargs)
        else:
            atoms = self.atoms(*args, **kwargs)
        X = np.tile(location, [len(atoms), 1])
        Y = np.array([a.location for a in atoms])
        distances = cdist(X, Y)[0]
        return {a for index, a in enumerate(atoms) if distances[index] <= radius}


    def pairwise_atoms(self, *args, **kwargs):
        """A generator which yeilds all the pairwise atom combinations of the
        structure. There will be no duplicates in the returned generator, and
        the number of returned pairs will be a triangle number.

        :rtype: ``tuple``"""

        atoms = list(self.atoms(*args, **kwargs))
        for a_index in range(len(atoms) - 1):
            for o_index in range(a_index + 1, len(atoms)):
                yield {atoms[a_index], atoms[o_index]}


    def infer_bonds(self, tolerance=0.4):
        """Automatically identifies and creates bonds between atoms in the
        structure based on their distance and covalent radii.

        :param float tolerance: a tolerance to add to the covalent radii sum.
         The default is 0.4."""

        for atom1, atom2 in self.pairwise_atoms():
            max_dist = atom1.covalent_radius + atom2.covalent_radius + tolerance
            if atom1.distance_to(atom2) < max_dist:
                atom1.bond(atom2)


    def nearby_atoms(self, *args, **kwargs):
        """Returns all atoms within a given distance of this structure,
        excluding the structure's own atoms.

        This will be a lot faster if the model's
        :py:meth:`.optimise_distances` has been called, as it won't have to
        search all atoms.

        :param float cutoff: the distance cutoff to use.
        :rtype: ``set``"""

        atoms = set()
        for atom in self.atoms():
            atoms.update(atom.nearby_atoms(*args, **kwargs))
        return atoms - self.atoms()
    

    def nearby_hets(self, *args, **kwargs):
        """Returns all other het structures within a given distance of this
        structure, excluding itself.

        This will be a lot faster if the model's
        :py:meth:`.optimise_distances` has been called, as it won't have to
        search all atoms.

        :param float cutoff: the distance cutoff to use.
        :rtype: ``set``"""

        structures = set()
        hets = set()
        for atom in self.atoms():
            structures.update(atom.nearby_hets(*args, **kwargs))
            hets.add(atom.het)
        return structures - hets
    

    def nearby_chains(self, *args, **kwargs):
        """Returns all other chain structures within a given distance of this
        structure, excluding itself.

        :param float cutoff: the distance cutoff to use.
        :rtype: ``set``"""

        structures = set()
        chains = set()
        for atom in self.atoms():
            structures.update(atom.nearby_chains(*args, **kwargs))
            chains.add(atom.chain)
        return structures - chains


    def translate(self, dx=0, dy=0, dz=0, trim=12):
        """Translates the structure through space, updating all atom
        coordinates accordingly. You can provide three values, or a single
        vector.

        :param Number dx: The distance to move in the x direction.
        :param Number dy: The distance to move in the y direction.
        :param Number dz: The distance to move in the z direction.
        :param int trim: The amount of rounding to do to the atoms' coordinates after translating - the default is 12 decimal places but this can be set to ``None`` if no rounding is to be done."""

        try:
            _, _, _ = dx
            vector = dx
        except TypeError:
            vector = (dx, dy, dz)
        Atom.translate_atoms(vector, *self.atoms())
        self.trim(trim)


    def transform(self, matrix, trim=12):
        """Transforms the structure using a 3x3 matrix supplied. This is useful
        if the :py:meth:`.rotate` method isn't powerful enough for your needs.

        :param array matrix: A NumPy matrix representing the transformation. You can supply a list of lists if you like and it will be converted to a NumPy matrix.
        :param int trim: The amount of rounding to do to the atoms' coordinates after transforming - the default is 12 decimal places but this can be set to ``None`` if no rounding is to be done."""

        Atom.transform_atoms(matrix, *self.atoms())
        self.trim(trim)


    def rotate(self, angle, axis, point=(0, 0, 0), trim=12):
        """Rotates the structure about an axis, updating all atom coordinates
        accordingly.

        :param Number angle: The angle in radians.
        :param str axis: The axis to rotate around. Can only be 'x', 'y' or 'z'.
        :param point: a point on the axis of rotation.
        :param int trim: The amount of rounding to do to the atoms' coordinates after translating - the default is 12 decimal places but this can be set to ``None`` if no rounding is to be done."""

        Atom.rotate_atoms(angle, axis, *self.atoms(), point=point)
        self.trim(trim)


    def trim(self, places):
        """Rounds the coordinate values to a given number of decimal places.
        Useful for removing floating point rounding errors after transformation.

        :param int places: The number of places to round the coordinates to. If ``None``, no rounding will be done."""

        for atom in self.atoms():
            atom.trim(places)



class Molecule(AtomStructure):
    """A molecule is a top-level constituent of a :py:class:`.Model` - a chain,
    a ligand, or a water molecule. They can have internal IDs, separate from the
    standard ID."""

    def __init__(self, id, name, internal_id):
        """Creates a Molecule.

        :param id: The molecule's ID.
        :param str name: The molecule's name.
        :param str internal_id: The molecule's internal ID."""

        AtomStructure.__init__(self, id, name)
        self._internal_id = internal_id
        self._model = None


    @property
    def internal_id(self):
        """The molecule's internal ID - how it is refered to by atomium
        operations. This will be identical to regular IDs when the model comes
        from a .pdb file, but .cif and .mmtf files make this distinction.
        
        :rtype: ``str``"""

        return self._internal_id or self._id
       

    @property
    def model(self):
        """Returns the molecules :py:class:`.Model`.

        :rtype: ``Model``"""

        return self._model



class Het(AtomStructure):
    """A direct container of atoms, such as a residue or ligand. Though never
    instantiated directly, there is an initaliser method for setting up the
    atom dictionary."""

    def __init__(self, id, name, full_name, *atoms):
        """Creates a Het.

        :param id: The het's ID.
        :param str name: The het's name.
        :param str full_name: The het's full name.
        :param *atoms: The atoms that make up the het."""

        AtomStructure.__init__(self, id, name)
        self._full_name = full_name
        for atom in atoms:
            atom._het = self
        self._atoms = StructureSet(*atoms)


    def __contains__(self, atom):
        return atom in self._atoms.structures
    

    @property
    def full_name(self):
        """Returns the residue's full name, based on its three letter name - or
        just the three letter name if it doesn't match anything. Or you can just
        supply a full name when you instantiate the Het.

        :rtype: ``str``"""

        if self._full_name:
            return self._full_name
        return FULL_NAMES.get(self._name, self._name)
    

    @full_name.setter
    def full_name(self, full_name):
        self._full_name = full_name
    

    @property
    def chain(self):
        """Returns the :py:class:`.Chain` the structure is part of (if a
        residue) or associated with (if a ligand).

        :rtype: ``Chain``"""

        return self._chain


    def atoms(self):
        """Returns the atoms in the ligand.

        :rtype: ``set``"""

        return self._atoms


    
class Model(AtomStructure, metaclass=StructureClass):
    """The universe in which all other molecules live, interact, and generally
    exist.

    It is a cotainer of its molecules, residues, and atoms."""

    def __init__(self, *molecules, file=None):
        """Creates a Model.

        :param *molecules: The chains, ligands, and waters that will inhabit the model.
        :param File file: The file the model came from."""

        AtomStructure.__init__(self, None, None)
        self._chains = set()
        self._ligands = set()
        self._waters = set()
        for mol in molecules:
            mol._model = self
            d = (self._chains if isinstance(mol, Chain) else self._waters
             if mol._water else self._ligands)
            d.add(mol)
        self._chains = StructureSet(*self._chains)
        self._ligands = StructureSet(*self._ligands)
        self._waters = StructureSet(*self._waters)
        self._file = file
        self._internal_grid = None


    def __repr__(self):
        chains = "{} chains".format(len(self._chains))
        if len(self._chains) == 1:
            chains = chains[:-1]
        ligands = "{} ligands".format(len(self._ligands))
        if len(self._ligands) == 1:
            ligands = ligands[:-1]
        return "<Model ({}, {})>".format(chains, ligands)


    def __contains__(self, obj):
        return (obj in self.molecules() or obj in self.residues()
         or obj in self.atoms())


    @property
    def file(self):
        """The :py:class:`.File` the model comes from."""

        return self._file


    def chains(self):
        """Returns the model's chains.

        :rtype: ``set``"""

        return self._chains


    def ligands(self):
        """Returns the model's ligands.

        :rtype: ``set``"""

        return self._ligands


    def waters(self):
        """Returns the model's water ligands.

        :rtype: ``set``"""

        return self._waters


    def molecules(self):
        """Returns all of the model's molecules (chains, ligands, waters).

        :rtype: ``set``"""

        return self._chains + self._ligands + self._waters


    def residues(self):
        """Returns all of the model's residues in all its chains.

        :rtype: ``set``"""

        res = []
        for chain in self._chains.structures:
            res += chain.residues()
        return StructureSet(*res)


    def atoms(self):
        """Returns all of the model's atoms in all its molecules.

        :rtype: ``set``"""

        atoms = set()
        for mol in self.molecules():
            try:
                atoms.update(mol._atoms.structures)
            except AttributeError:
                for res in mol._residues.structures:
                    atoms.update(res._atoms.structures)
        return StructureSet(*atoms)


    def dehydrate(self):
        """Removes all water ligands from the model."""

        self._waters = StructureSet()
    

    def optimise_distances(self):
        """Calling this method makes finding atoms within a sphere faster, and
        consequently makes all 'nearby' methods faster. It organises the atoms
        in the model into grids, so that only relevant atoms are checked for
        distances."""

        self._internal_grid = defaultdict(
         lambda: defaultdict(lambda: defaultdict(set))
        )
        for atom in self.atoms():
            x, y, z = [int(math.floor(n / 10)) * 10 for n in atom.location]
            self._internal_grid[x][y][z].add(atom)


    #TODO copy



class Chain(Molecule, metaclass=StructureClass):
    """A sequence of residues. Unlike other structures, they are iterable, and
    have a length.

    Residues can also be accessed using indexing."""

    def __init__(self, *residues, sequence="", helices=None, strands=None, information=None, **kwargs):
        """Creates a Chain.

        :param *residues: The residues that will make up the chain.
        :param str id: the chain's unique ID.
        :param str internal_id: the internal ID used for transformations.
        :param str sequence: the actual sequence the chain should have.
        :param list helices: the alpha helices within the chain.
        :param list strands: the beta strands within the chain."""

        Molecule.__init__(
         self, kwargs.get("id"), kwargs.get("name"), kwargs.get("internal_id")
        )
        self._sequence = sequence
        for res in residues:
            res._chain = self
        self._residues = StructureSet(*residues)
        self._model = None
        self._helices = helices or []
        self._strands = strands or []
        self._information = information or {}

    def __repr__(self):
        return "<Chain {} ({} residues)>".format(self._id, len(self._residues))


    def __len__(self):
        return len(self._residues)


    def __iter__(self):
        return iter(self._residues.structures)


    def __getitem__(self, key):
        return self.residues()[key]


    def __contains__(self, obj):
        return obj in self._residues.structures or obj in self.atoms()


    @property
    def sequence(self):
        """Returns the sequence associated with the chain. Note that this is the
        sequence that the molecule actually has in real life - some may be
        missing from the actual sequence of residues in the structure.

        :rtype: ``str``"""

        return self._sequence


    @sequence.setter
    def sequence(self, sequence):
        self._sequence = sequence


    @property
    def helices(self):
        """The alpha helix residues in the chain

        :rtype: ``tuple``"""

        return tuple(self._helices)
    

    @property
    def strands(self):
        """The beta strand residues in the chain

        :rtype: ``tuple``"""

        return tuple(self._strands)

    @property
    def information(self):
        """The source organism and other information related to the chain

        :rtype: ``dict``"""

        return self._information

    @property
    def length(self):
        """Returns the number of residues in the chain.

        :rtype: ``int``"""

        return len(self)
    

    @property
    def present_sequence(self):
        """The sequence of residues actually present in the atoms present.

        :rtype: ``str``"""

        return "".join(r.code for r in self.residues())
    

    def copy(self, id=None, residue_ids=None, atom_ids=None):
        """Creates a copy of the chain, with new atoms and residues.

        :param str id: if given, the ID of the new chain.
        :param function residue_ids: a callable which, if given, will generate new residue IDs.
        :param function atom_ids: a callable which, if given, will generate new atom IDs.
        :rtype: ``Chain``"""

        residue_ids = residue_ids or (lambda i: i)
        residues = {r: r.copy(
         id=residue_ids(r.id), atom_ids=atom_ids
        ) for r in self.residues()}
        for r in self.residues():
            residues[r].next = residues[r.next] if r.next else None
        return Chain(
         *residues.values(), id=id or self._id, internal_id=self._internal_id,
         name=self._name, sequence=self._sequence,
         helices=[tuple(residues[r] for r in h) for h in self._helices],
         strands=[tuple(residues[r] for r in s) for s in self._strands]
        )
    

    def residues(self):
        """Returns the residues in the chain.

        :rtype: ``tuple``"""

        return self._residues


    def ligands(self):
        """Returns all the ligands associated with the chain - but only if the
        chain is part of a model.

        :rtype: ``set``"""

        return StructureSet() if self._model is None else StructureSet(
         *[
          ligand for ligand in self._model._ligands.structures
          if ligand._chain is self
         ]
        )


    def atoms(self):
        """Returns all the atoms in with the chain.

        :rtype: ``set``"""

        atoms = set()
        for res in self._residues.structures:
            atoms.update(res._atoms.structures)
        return StructureSet(*atoms)



class Ligand(Molecule, Het, metaclass=StructureClass):
    """A small molecule, usually associated with a polymer chain."""

    def __init__(self, *atoms, chain=None, water=False, **kwargs):
        """Creates a Ligand.

        :param *atoms: The atoms that will make up the ligand.
        :param str id: the ligand's unique ID.
        :param str name: the ligand's name.
        :param str internal_id: the internal ID used for transformations.
        :param Chain chain: the chain the ligand is associated with.
        :param bool water: if ``True``, the ligand will be treated as water."""

        Het.__init__(
        self, kwargs.get("id"), kwargs.get("name"),
         kwargs.get("full_name"), *atoms)
        Molecule.__init__(self, kwargs.get("id"), kwargs.get("name"),
         kwargs.get("internal_id"))
        self._chain, self._water = chain, water


    def __repr__(self):
        return "<{} {} ({})>".format(
         "Water" if self._water else "Ligand", self._name, self._id
        )


    @property
    def is_water(self):
        """Returns ``True`` if the ligand is a water ligand.

        :rtype: ``bool``"""

        return self._water


    def copy(self, id=None, atom_ids=None):
        """Creates a copy of the ligand, with new atoms.

        :param str id: if given, the ID of the new ligand.
        :param function atom_ids: a callable which, if given, will generate new atom IDs.
        :rtype: ``Ligand``"""

        atoms = list(self.atoms())
        if atom_ids:
            new_ids = [atom_ids(a.id) for a in atoms]
            atoms = [a.copy(id=id) for a, id in zip(atoms, new_ids)]
        else:
            atoms = [a.copy() for a in self.atoms()]
        return self.__class__(*atoms, id=id or self._id,
         name=self._name, internal_id=self._internal_id, water=self._water)



class Residue(Het, metaclass=StructureClass):
    """A small subunit within a chain."""

    def __init__(self, *atoms, **kwargs):
        """Creates a Residue.

        :param *atoms: The atoms the residue is to be made of.
        :param str id: The residue's ID.
        :param str name: The residue's name."""

        Het.__init__(self, kwargs.get("id"), kwargs.get("name"),
         kwargs.get("full_name"), *atoms)
        self._next, self._previous = None, None
        self._chain = None


    def __repr__(self):
        return "<Residue {} ({})>".format(self._name, self._id)


    @property
    def next(self):
        """Residues can be linked to each other in a linear chain. This property
        returns the :py:class:`.Residue` downstream of this one. Alternatively,
        if you supply a residue, that residue will be assigned as the 'next' one
        downstream to this, and this residue will be upstream to that.
        Note that is a separate concept from bonds.

        :raises ValueError: if you try to connect a residue to itself.
        :rtype: ``Residue``"""

        return self._next


    @next.setter
    def next(self, next):
        if next is None:
            if self._next:
                self._next._previous = None
            self._next = None
        elif next is self:
            raise ValueError("Cannot link {} to itself".format(self))
        else:
            self._next = next
            next._previous = self


    @property
    def previous(self):
        """Residues can be linked to each other in a linear chain. This property
        returns the :py:class:`.Residue` upstream of this one. Alternatively,
        if you supply a residue, that residue will be assigned as the 'previous'
        one upstream to this, and this residue will be downstream to that.

        :raises ValueError: if you try to connect a residue to itself.
        :rtype: ``Residue``"""

        return self._previous


    @previous.setter
    def previous(self, previous):
        if previous is None:
            if self._previous:
                self._previous._next = None
            self._previous = None
        elif previous is self:
            raise ValueError("Cannot link {} to itself".format(self))
        else:
            self._previous = previous
            previous._next = self


    @property
    def code(self):
        """Returns the single letter code, based on its three letter name - or
        just 'X' if it doesn't match anything.

        :rtype: ``str``"""

        return CODES.get(self._name, "X")


    @property
    def helix(self):
        """Returns ``True`` if the residue is part of an alpha helix.

        :rtype: ``bool``"""

        if self.chain:
            for helix in self.chain.helices:
                if self in helix:
                    return True
        return False
    

    @property
    def strand(self):
        """Returns ``True`` if the residue is part of a beta strand.

        :rtype: ``bool``"""

        if self.chain:
            for strand in self.chain.strands:
                if self in strand:
                    return True
        return False


    @property
    def phi(self):
        """Returns the phi angle of the residue. This is the dihedral angle
        defined by the C atom of the previous residue, and the N, C-alpha, and
        C atoms of this residue. It will be ``None`` if the residue is the
        first in the chain.

        :rtype: ``float``"""

        if self.previous:
            try:
                c_prev = self.previous.atom(name="C")
                n, ca, c = self.atom(name="N"), self.atom(name="CA"), self.atom(name="C")
                return Atom.dihedral(c_prev, n, ca, c)
            except AttributeError:
                return None


    @property
    def psi(self):
        """Returns the psi angle of the residue. This is the dihedral angle
        defined by the N, C-alpha, and C atoms of this residue, and the N atom
        of the next residue. It will be ``None`` if the residue is the last
        in the chain.

        :rtype: ``float``"""

        if self.next:
            try:
                n_next = self.next.atom(name="N")
                n, ca, c = self.atom(name="N"), self.atom(name="CA"), self.atom(name="C")
                return Atom.dihedral(n, ca, c, n_next)
            except AttributeError:
                return None


    @property
    def omega(self):
        """Returns the omega angle of the residue. This is the dihedral angle
        defined by the C-alpha and C atoms of this residue, and the N and
        C-alpha atoms of the next residue. It will be ``None`` if the residue
        is the last in the chain.

        :rtype: ``float``"""

        if self.next:
            try:
                n_next, ca_next = self.next.atom(name="N"), self.next.atom(name="CA")
                ca, c = self.atom(name="CA"), self.atom(name="C")
                return Atom.dihedral(ca, c, n_next, ca_next)
            except AttributeError:
                return None


    def set_phi(self, angle):
        """Sets the phi angle of the residue to the given value, by rotating
        the relevant downstream atoms.

        :param float angle: The angle to set, in degrees."""

        from .utilities import find_downstream_atoms
        current_angle = self.phi
        if current_angle is None: return

        delta = math.radians(angle - current_angle)
        n = self.atom(name="N")
        ca = self.atom(name="CA")
        if n is None or ca is None: return

        axis = np.array(ca.location) - np.array(n.location)
        point = n.location
        atoms_to_rotate = find_downstream_atoms(ca, n)
        atoms_to_rotate = {a for a in atoms_to_rotate if a is not ca}
        Atom.rotate_atoms(delta, axis, *atoms_to_rotate, point=point)


    def set_psi(self, angle):
        """Sets the psi angle of the residue to the given value, by rotating
        the relevant downstream atoms.

        :param float angle: The angle to set, in degrees."""

        from .utilities import find_downstream_atoms
        current_angle = self.psi
        if current_angle is not None:
            delta = math.radians(angle - current_angle)
            ca = self.atom(name="CA")
            c = self.atom(name="C")
            if ca and c:
                axis = np.array(c.location) - np.array(ca.location)
                point = ca.location
                atoms_to_rotate = find_downstream_atoms(c, ca)
                atoms_to_rotate = {a for a in atoms_to_rotate if a is not c}
                Atom.rotate_atoms(delta, axis, *atoms_to_rotate, point=point)


    def set_omega(self, angle):
        """Sets the omega angle of the residue to the given value, by rotating
        the relevant downstream atoms.

        :param float angle: The angle to set, in degrees."""

        from .utilities import find_downstream_atoms
        current_angle = self.omega
        if current_angle is not None:
            delta = math.radians(angle - current_angle)
            c = self.atom(name="C")
            n_next = self.next.atom(name="N")
            if c and n_next:
                axis = np.array(n_next.location) - np.array(c.location)
                point = c.location
                atoms_to_rotate = find_downstream_atoms(n_next, c)
                atoms_to_rotate = {a for a in atoms_to_rotate if a is not n_next}
                Atom.rotate_atoms(delta, axis, *atoms_to_rotate, point=point)


    def set_chi(self, n, angle):
        """Sets the nth chi angle of the residue to the given value, by rotating
        the relevant downstream atoms.

        :param int n: The chi angle to set (1-5).
        :param float angle: The angle to set, in degrees."""

        from .utilities import find_downstream_atoms
        current_angle = self.chi(n)
        if current_angle is not None:
            delta = math.radians(angle - current_angle)

            atom_names = CHI_ANGLES[self.name][n - 1]
            b, c = self.atom(name=atom_names[1]), self.atom(name=atom_names[2])

            if b and c:
                axis = np.array(c.location) - np.array(b.location)
                point = b.location
                atoms_to_rotate = find_downstream_atoms(c, b)
                atoms_to_rotate = {a for a in atoms_to_rotate if a is not c}
                Atom.rotate_atoms(delta, axis, *atoms_to_rotate, point=point)


    def chi(self, n):
        """Returns the nth chi angle of the residue. This is the dihedral angle
        of the side chain. It will be ``None`` if the residue does not have that
        chi angle, or if any of the required atoms are missing.

        :param int n: The chi angle to get (1-5).
        :rtype: ``float``"""

        if self.name in CHI_ANGLES and 1 <= n <= len(CHI_ANGLES[self.name]):
            atom_names = CHI_ANGLES[self.name][n - 1]
            try:
                atoms = [self.atom(name=name) for name in atom_names]
                if None in atoms: return None
                return Atom.dihedral(*atoms)
            except AttributeError:
                return None


    def copy(self, id=None, atom_ids=None):
        """Creates a copy of the residue, with new atoms.

        :param str id: if given, the ID of the new residue.
        :param function atom_ids: a callable which, if given, will generate new atom IDs.
        :rtype: ``Residue``"""

        atoms = list(self.atoms())
        if atom_ids:
            new_ids = [atom_ids(a.id) for a in atoms]
            atoms = [a.copy(id=id) for a, id in zip(atoms, new_ids)]
        else:
            atoms = [a.copy() for a in self.atoms()]
        return self.__class__(*atoms, id=id or self._id, name=self._name)
    

    @property
    def model(self):
        """Returns the :py:class:`.Model` the residue is part of, via its
        chain.

        :rtype: ``Model``"""

        try:
            return self._chain._model
        except AttributeError:
            return None



class Atom:
    """An atom in space - a point particle with a location, element, charge etc.

    Atoms are the building blocks of all structures in atomium.

    Two atoms are equal if they have the same properties (not including ID)."""

    __slots__ = [
     "_element", "_location", "_id", "_name", "_charge",
     "_bvalue", "_anisotropy", "_het", "_bonded_atoms", "_is_hetatm"
    ]

    def __init__(self, element, x, y, z, id, name, charge, bvalue, anisotropy, is_hetatm=False):
        """Creates an Atom.

        :param str element: The atom's elemental symbol.
        :param number x: The atom's x coordinate.
        :param number y: The atom's y coordinate.
        :param number z: The atom's z coordinate.
        :param int id: An integer ID for the atom.
        :param str name: The atom's name.
        :param number charge: The charge of the atom.
        :param number bvalue: The B-value of the atom (its uncertainty).
        :param list anisotropy: The directional uncertainty of the atom."""

        self._location = np.array([x, y, z])
        self._element = element
        self._id, self._name, self._charge = id, name, charge
        self._bvalue, self._anisotropy = bvalue, anisotropy
        self._het, self._bonded_atoms, self._is_hetatm = None, set(), is_hetatm


    def __repr__(self):
        return "<Atom {} ({})>".format(self._id, self._name)


    def __iter__(self):
        return iter(self._location)


    def __eq__(self, other):
        """Checks if this atom is equal to another.

        Two atoms are equal if they have the same properties (not including ID).

        :param Atom other: the atom to compare to.
        :rtype: ``bool``"""

        if not isinstance(other, Atom):
            return False
        for attr in self.__slots__:
            if attr not in ("_id", "_het", "_bonded_atoms", "_location"):
                if getattr(self, attr) != getattr(other, attr):
                    return False
            if list(self._location) != list(other._location):
                return False
        return True


    def __hash__(self):
        return id(self)


    @staticmethod
    def translate_atoms(vector, *atoms):
        """Translates multiple atoms using some vector.

        :param vector: the three values representing the delta position.
        :param *atoms: the atoms to translate."""

        for atom in atoms:
            atom._location += np.array(vector)


    @staticmethod
    def transform_atoms(matrix, *atoms):
        """Transforms multiple atoms using some matrix.

        :param matrix: the transformation matrix.
        :param *atoms: the atoms to transform."""

        locations = [list(a) for a in atoms]
        output = np.dot(np.array(matrix), np.array(locations).transpose())
        for atom, location in zip(atoms, output.transpose()):
            atom._location = location


    @staticmethod
    def rotate_atoms(angle, axis, *atoms, point=(0, 0, 0), **kwargs):
        """Rotates multiple atoms using an axis and an angle.

        :param float angle: the angle to rotate by in radians.
        :param str or vector axis: the axis to rotate around. Can be 'x', 'y',
         'z', or a vector.
        :param *atoms: the atoms to rotate.
        :param point: a point on the axis of rotation."""

        try:
            axis = [1 if i == "xyz".index(axis) else 0 for i in range(3)]
        except (ValueError, TypeError):
            try:
                x, y, z = axis
            except:
                raise ValueError("'{}' is not a valid axis".format(axis))

        point = np.array(point)
        Atom.translate_atoms(point * -1, *atoms)

        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(angle / 2)
        b, c, d = -axis * np.sin(angle / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        matrix = np.array([
         [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
        ])
        Atom.transform_atoms(matrix, *atoms, **kwargs)
        Atom.translate_atoms(point, *atoms)


    @property
    def element(self):
        """The atom's element symbol. This is used to calculate its mass using a
        Periodic Table.

        :rtype: ``str``"""

        return self._element


    @property
    def location(self):
        """The atom's location.

        :rtype: ``tuple``"""

        return tuple(self._location)


    @property
    def id(self):
        """The atom's unique integer ID. It cannot be updated - the ID the atom
        is created with is its ID forever.

        :rtype: ``int``"""

        return self._id


    @property
    def name(self):
        """The atom's name. This is often used to determine what 'kind' of atom
        it is.

        :rtype: ``str``"""

        return self._name


    @name.setter
    def name(self, name):
        self._name = name


    @property
    def charge(self):
        """The atom's charge - usually just zero, or 'neutral'.

        :rtype: ``float``"""

        return self._charge


    @charge.setter
    def charge(self, charge):
        self._charge = charge


    @property
    def bvalue(self):
        """The atom's B-value - the uncertainty in its position in all
        directions.

        :rtype: ``float``"""

        return self._bvalue


    @bvalue.setter
    def bvalue(self, bvalue):
        self._bvalue = bvalue


    @property
    def anisotropy(self):
        """The atom's directional uncertainty, represented by a list of six
        numbers.

        :rtype: ``list``"""

        return self._anisotropy


    @property
    def bonded_atoms(self):
        """Returns the atoms this atom is bonded to.

        :rtype: ``set```"""

        return self._bonded_atoms


    @property
    def mass(self):
        """The atom's molar mass according to the Periodic Table, based on the
        atom's :py:meth:`element`. If the element doesn't match any symbol on
        the Periodic Table, a mass of 0 will be returned.

        The element lookup is case-insensitive.

        :rtype: ``float``"""

        return PERIODIC_TABLE.get(self._element.upper(), 0)
    

    @property
    def atomic_number(self):
        """The atom's nuclear charge or atomic number, based on the
        atom's :py:meth:`element`. If the element doesn't match any symbol on
        the Periodic Table, a mass of 0 will be returned.

        The element lookup is case-insensitive.

        :rtype: ``int``"""

        return ATOMIC_NUMBER.get(self._element.upper(), 0)


    @property
    def covalent_radius(self):
        """The atom's covalent radius, based on the atom's :py:meth:`element`.
        If the element doesn't match any symbol on the Periodic Table, a radius
        of 0 will be returned.

        The element lookup is case-insensitive.

        :rtype: ``float``"""

        return COVALENT_RADII.get(self._element.upper(), 0)


    @property
    def is_metal(self):
        """Checks whether the atom's element matches a metal element.

        The element lookup is case-insensitive.

        :rtype: ``bool``"""

        return self._element.upper() in METALS


    @property
    def is_backbone(self):
        """Returns ``True`` if the atom has a backbone atom name.

        :rtype: ``bool``"""

        return isinstance(self._het, Residue) and \
         self._name in ["CA", "C", "N", "O"]


    @property
    def is_side_chain(self):
        """Returns ``True`` if the atom has a side chain atom name.

        :rtype: ``bool``"""

        return isinstance(self._het, Residue) and not self.is_backbone


    def distance_to(self, other):
        """Returns the distance (in whatever units the coordinates are defined
        in) between this atom and another. You can also give a (x, y, z) tuple
        instead of another atom if you so wish.

        :param Atom other: The other atom (or location tuple).
        :rtype: ``float``"""

        return np.linalg.norm(self._location - np.array(list(other)))


    def angle(self, atom1, atom2):
        """Gets the angle between two atom vectors with this atom as the origin.

        :param Atom atom1: The first atom.
        :param Atom atom2: Thne second atom."""

        vectors = [
         [v1 - v2 for v1, v2 in zip(atom.location, self.location)
        ] for atom in (atom1, atom2)]
        normalized = [np.linalg.norm(v) for v in vectors]
        if 0 in normalized:
            return 0
        vectors = [v / n for v, n in zip(vectors, normalized)]
        return np.arccos(np.clip(np.dot(vectors[0], vectors[1]), -1.0, 1.0))


    @staticmethod
    def dihedral(p1, p2, p3, p4):
        """Returns the dihedral angle in degrees between four points.

        :param Atom p1: The first point.
        :param Atom p2: The second point.
        :param Atom p3: The third point.
        :param Atom p4: The fourth point.
        :rtype: ``float``"""

        p1 = np.array(p1.location)
        p2 = np.array(p2.location)
        p3 = np.array(p3.location)
        p4 = np.array(p4.location)
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        angle = np.degrees(np.arctan2(np.dot(v1, n2), np.dot(n1, n2)))
        return angle
    

    def copy(self, id=None):
        """Returns a copy of the atom. The new atom will have the same element,
        location, name, charge, ID, bvalue etc. as the original, but will not
        be part of any model or other molecule.

        :rtype: ``Atom``"""

        return Atom(
         self._element, *self._location, id or self._id, self._name,
         self._charge, self._bvalue, self._anisotropy
        )
    

    @property
    def het(self):
        """Returns the :py:class:`.Residue` or :py:class:`.Ligand` the atom is
        part of, or ``None`` if it is not part of one.

        :rtype: ``Het```"""

        return self._het
    

    @property
    def chain(self):
        """Returns the :py:class:`.Chain` the atom is part of, or ``None`` if
        it is not part of one.

        :rtype: ``Chain``"""

        if self._het:
            return self._het.chain


    @property
    def model(self):
        """Returns the :py:class:`.Model` the atom is part of, or ``None`` if
        it is not part of one.

        :rtype: ``Model``"""

        if self.chain:
            return self.chain.model


    def nearby_atoms(self, cutoff, *args, **kwargs):
        """Returns all atoms in the associated :py:class:`.Model` that are
        within a given distance (in the units of the atom coordinates) of this
        atom. If the atom is not part of a model, no atoms will be returned.

        :param float cutoff: The radius to search within.
        :rtype: ``set``"""

        if self.model:
            atoms =  self.model.atoms_in_sphere(
             self.location, cutoff, *args, **kwargs
            )
            try:
                atoms.remove(self)
            except KeyError:
                pass
            return atoms
        return set()


    def nearby_hets(self, *args, residues=True, ligands=True, **kwargs):
        """Returns all residues and ligands in the associated :py:class:`.Model`
        that are within a given distance (in the units of the atom coordinates)
        of this atom. If the atom is not part of a model, no residues will be
        returned.

        :param float cutoff: the distance cutoff to use.
        :param bool residues: if ``False``, residues will not be returned.
        :param bool ligands: if ``False``, ligands will not be returned.
        :rtype: ``set``"""

        atoms = self.nearby_atoms(*args, **kwargs)
        structures = set()
        for atom in atoms:
            if atom.het is not None:
                structures.add(atom.het)
        try:
            structures.remove(self.het)
        except KeyError:
            pass
        if not residues:
            structures = {s for s in structures if not isinstance(s, Residue)}
        if not ligands:
            structures = {s for s in structures if not (isinstance(s, Ligand))}
        return structures
    

    def nearby_chains(self, *args, **kwargs):
        """Returns all chain structures in the associated :py:class:`.Model`
        that are within a given distance (in the units of the atom coordinates)
        of this atom. If the atom is not part of a model, no chains will be
        returned.

        :param float cutoff: the distance cutoff to use.
        :rtype: ``set``"""

        atoms = self.nearby_atoms(*args, **kwargs)
        chains = set()
        for atom in atoms:
            if atom.chain is not None:
                chains.add(atom.chain)
        try:
            chains.remove(self.chain)
        except KeyError:
            pass
        return chains


    def translate(self, dx=0, dy=0, dz=0, trim=12):
        """Translates an atom in 3D space. You can provide three values, or a
        single vector.

        :param float dx: The distance to move in the x direction.
        :param float dy: The distance to move in the y direction.
        :param float dz: The distance to move in the z direction.
        :param int trim: The amount of rounding to do to the atom's coordinates after translating - the default is 12 decimal places but this can be set to ``None`` if no rounding is to be done."""

        try:
            _, _, _ = dx
            vector = dx
        except TypeError:
            vector = (dx, dy, dz)
        Atom.translate_atoms(vector, self)
        self.trim(trim)


    def transform(self, matrix, trim=12):
        """Transforms the atom using a 3x3 matrix supplied. This is useful if
        the :py:meth:`.rotate` method isn't powerful enough for your needs.

        :param array matrix: A NumPy matrix representing the transformation. You can supply a list of lists if you like and it will be converted to a NumPy matrix.
        :param int trim: The amount of rounding to do to the atom's coordinates after transforming - the default is 12 decimal places but this can be set to ``None`` if no rounding is to be done."""

        Atom.transform_atoms(matrix, self)
        self.trim(trim)


    def rotate(self, angle, axis, point=(0, 0, 0), trim=12):
        """Rotates the atom by an angle in radians, around one of the the three
        axes.

        :param float angle: The angle to rotate by in radians.
        :param str axis: the axis to rotate around.
        :param point: a point on the axis of rotation.
        :param int trim: The amount of rounding to do to the atom's coordinates after rotating - the default is 12 decimal places but this can be set to ``None`` if no rounding is to be done."""

        Atom.rotate_atoms(angle, axis, self, point=point)
        self.trim(trim)


    def move_to(self, x, y, z):
        """Moves the atom to the coordinates given.

        :param number x: The atom's new x coordinate.
        :param number y: The atom's new y coordinate.
        :param number z: The atom's new z coordinate."""

        self._location[0], self._location[1], self._location[2] = x, y, z


    def trim(self, places):
        """Rounds the coordinate values to a given number of decimal places.
        Useful for removing floating point rounding errors after transformation.

        :param int places: The number of places to round the coordinates to. If ``None``, no rounding will be done."""

        if places is not None:
            self._location = np.round(self._location, places)


    def bond(self, other):
        """Bonds the atom to some other atom.

        :param Atom other: the other atom to bond to."""
        
        self._bonded_atoms.add(other)
        other._bonded_atoms.add(self)
