"""Decorators and metaclasses used by atomium structures."""

import re
import numpy as np
import math
import warnings
from collections import Counter, defaultdict
from .rmsd import kabsch_rmsd

def get_object_from_filter(obj, components):
    """Gets the object whose attributes are actually being queried, which may be
    a different object if there is a chain.

    :param obj: the intial object.
    :param list components: the components of the original key.
    :returns: the relevant object"""

    components = components[:]
    while len(components) > 2:
        obj = getattr(obj, components.pop(0))
    if len(components) == 2:
        if components[-1] != "regex":
            if not hasattr(obj, f"__{components[-1]}__"):
                obj = getattr(obj, components[0])
    return obj


def get_object_attribute_from_filter(obj, components):
    """Gets the object's value of some attribute based on a list of key
    components.

    :param obj: the object with attributes.
    :param list components: the components of the original key.
    :returns: the value"""

    try:
        return getattr(
         obj, components[-1] if hasattr(obj, components[-1]) else components[-2]
        )
    except Exception:
        return None


def attribute_matches_value(attribute, value, components):
    """Checks if an attribute value matches a given value. The components given
    will determine whether an exact match is sought, or whether a more complex
    criterion is used.
    
    :param attribute: the value of an object's attribute.
    :param value: the value to match against.
    :param list components: the components of the original key.
    :rtype: ``bool``"""

    if components[-1] == "regex":
        return re.match(value, str(attribute))
    possible_magic = f"__{components[-1]}__"
    if hasattr(attribute, possible_magic):
        result = getattr(attribute, possible_magic)(value)
    else:
        result = getattr(attribute, "__eq__")(value)
    return True if result is NotImplemented else result


def filter_objects(objects, key, value):
    """Takes a :py:class:`.StructureSet` of objects, and filters them on object
    properties.

    They key can be an attribute of the object, or a complex double-underscore
    separated chain of attributes.

    :param StructreSet objects: the dictionary of objects - the keys are\
    unimportant.
    :param str key: the attribute to search. This can be an attribute of the object, or attr__regex, or attr__gt etc.
    :param value: the value that the attribute must have.
    :rtype: ``dict``"""

    components = key.split("__")
    matching_objects = []
    for structure in objects.structures:
        obj = get_object_from_filter(structure, components)
        attr = get_object_attribute_from_filter(obj, components)
        if attribute_matches_value(attr, value, components):
            matching_objects.append(structure)
    return StructureSet(*matching_objects)


def query(func, tuple_=False):
    """A decorator which can be applied to any function which returns a
    :py:class:`.StructureSet` and which takes no other parameters other than
    ``self``. It will query the returned objects by any keyword argument, or
    use a positional argument to search by ID.

    :param func: the function to modify.
    :param bool tuple_: if ``True``, objects will be returned in a tuple not a set.
    :rtype: ``function``"""

    def structures(self, *args, **kwargs):
        objects = func(self)
        original = {s: n for n, s in enumerate(objects.structures)}
        if len(args) == 1:
            return {objects.get(args[0])} if args[0] in objects.ids else set()
        for k, v in kwargs.items():
            objects = filter_objects(objects, k, v)
        if tuple_:
            return tuple(sorted(
             objects.structures, key=lambda s: original[s]
            ))
        else:
            return set(objects.structures)
    return structures


def getone(func):
    """A decorator which can be applied to any function which returns an
    iterable. It produces a function which just gets the first item in that
    iterable.

    In atomium, various classes define methods like atoms, residues, etc. - this
    decorator can make a function like atom, residue which takes all the same
    params but just returns one object.

    :param func: the function to modify.
    :rtype: ``function``"""

    def structure(self, *args, **kwargs):
        for obj in func(self, *args, **kwargs):
            return obj
    return structure



class StructureClass(type):
    """A metaclass which can be applied to structure class. It will override
    the instantation behaviour so that all methods that belong to a preset
    list ('atoms', 'chains' etc.) will have the :py:func:`.query` decorator
    applied and a copy with the :py:func:`.getone` decorator applied."""

    METHODS = ["chains", "residues", "ligands", "waters", "molecules", "atoms"]

    def __new__(self, *args, **kwargs):
        cls = type.__new__(self, *args, **kwargs)
        for attribute in dir(cls):
            if attribute in cls.METHODS:
                setattr(cls, attribute, query(
                 getattr(cls, attribute),
                 tuple_=(attribute == "residues" and cls.__name__ == "Chain")
                ))
                setattr(cls, attribute[:-1], getone(getattr(cls, attribute)))
        return cls



class StructureSet:
    """A data structure for holding structures. It stores them internally
    as a dictionary where they keys are IDs (to allow rapid lookup by ID) and
    the values are all structures with that ID (to allow for duplicate IDs).

    Two structure sets can be added together, but they are immutable - the
    structures they have when they are made is the structures they will always
    have.

    They're basically sets optimised to lookup things by ID.

    :param *args: the structures that will make up the StructureSet."""

    def __init__(self, *args):
        self._d = {}
        for obj in args:
            if obj._id in self._d:
                self._d[obj._id].add(obj)
            else:
                self._d[obj._id] = {obj}


    def __add__(self, other):
        new = StructureSet()
        for s in (self, other):
            for key, value in s._d.items():
                if key in new._d:
                    new._d[key].update(value)
                else:
                    new._d[key] = value
        return new


    def __len__(self):
        return len(self.structures)


    @property
    def ids(self):
        """Returns the IDs of the StructureSet.

        :rtype: ``set``"""

        return self._d.keys()


    @property
    def structures(self):
        """Returns the structures of the StructureSet.

        :rtype: ``list``"""

        structures = []
        for s in self._d.values():
            structures += s
        return structures


    def get(self, id):
        """Gets a structure by ID. If an ID points to multiple structures, just
        one will be returned.

        :returns: some structure."""

        matches = self._d.get(id, set())
        for match in matches:
            return match

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
        return round(kabsch_rmsd(np.array(coords1), np.array(coords2)), 12)


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

        from ..io.utils import save as save_file
        self.check_ids()
        ext = path.split(".")[-1]
        if ext == "cif":
            from ..io.mmcif import structure_to_mmcif_string
            string = structure_to_mmcif_string(self)
        elif ext == "mmtf":
            from ..io.mmtf import structure_to_mmtf_string
            string = structure_to_mmtf_string(self)
        elif ext == "pdb":
            from ..io.pdb import structure_to_pdb_string
            string = structure_to_pdb_string(self)
        else:
            raise ValueError("Unsupported file extension: " + ext)
        save_file(string, path)


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
        distances = np.linalg.norm(Y - X, axis=1)
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
        from .atoms import Atom
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
        from .atoms import Atom
        Atom.transform_atoms(matrix, *self.atoms())
        self.trim(trim)


    def rotate(self, angle, axis, point=(0, 0, 0), trim=12):
        """Rotates the structure about an axis, updating all atom coordinates
        accordingly.

        :param Number angle: The angle in radians.
        :param str axis: The axis to rotate around. Can only be 'x', 'y' or 'z'.
        :param point: a point on the axis of rotation.
        :param int trim: The amount of rounding to do to the atoms' coordinates after translating - the default is 12 decimal places but this can be set to ``None`` if no rounding is to be done."""
        from .atoms import Atom
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
