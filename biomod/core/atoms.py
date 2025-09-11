import numpy as np
from .constants import PERIODIC_TABLE, ATOMIC_NUMBER, COVALENT_RADII, METALS

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
        from .residues import Residue
        return isinstance(self._het, Residue) and \
         self._name in ["CA", "C", "N", "O"]


    @property
    def is_side_chain(self):
        """Returns ``True`` if the atom has a side chain atom name.

        :rtype: ``bool``"""
        from .residues import Residue
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
        from .residues import Residue, Ligand
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
