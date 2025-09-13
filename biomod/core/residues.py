import math
import numpy as np
from .constants import FULL_NAMES, CODES, CHI_ANGLES
from .base import AtomStructure, Molecule, StructureSet, StructureClass
from .atoms import Atom

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
    def number(self):
        return int(self._id.split('.')[-1])


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


    @chain.setter
    def chain(self, value):
        self._chain = value


    def atoms(self):
        """Returns the atoms in the ligand.

        :rtype: ``set``"""

        return self._atoms


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
        from ..utilities.secondary_structure.data_structures import HBond
        self._next, self._previous = None, None
        self._chain = None
        self.hbond_acceptor = [HBond(None, 0.0), HBond(None, 0.0)]
        self.hbond_donor = [HBond(None, 0.0), HBond(None, 0.0)]
        self.h_coord = None


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

        from ..utilities.utils import find_downstream_atoms_in_residue
        current_angle = self.phi
        if current_angle is None: return

        delta = math.radians(angle - current_angle)
        n = self.atom(name="N")
        ca = self.atom(name="CA")
        if n is None or ca is None: return

        axis = np.array(ca.location) - np.array(n.location)
        point = n.location
        atoms_to_rotate = find_downstream_atoms_in_residue(ca, n)
        atoms_to_rotate = {a for a in atoms_to_rotate if a is not ca}
        Atom.rotate_atoms(delta, axis, *atoms_to_rotate, point=point)


    def set_psi(self, angle):
        """Sets the psi angle of the residue to the given value, by rotating
        the relevant downstream atoms.

        :param float angle: The angle to set, in degrees."""

        from ..utilities.utils import find_downstream_atoms_in_chain
        current_angle = self.psi
        if current_angle is not None:
            delta = math.radians(angle - current_angle)
            ca = self.atom(name="CA")
            c = self.atom(name="C")
            if ca and c and self.next:
                axis = np.array(c.location) - np.array(ca.location)
                point = ca.location
                atoms_to_rotate = self.next.atoms()
                Atom.rotate_atoms(delta, axis, *atoms_to_rotate, point=point)


    def set_omega(self, angle):
        """Sets the omega angle of the residue to the given value, by rotating
        the relevant downstream atoms.

        :param float angle: The angle to set, in degrees."""

        from ..utilities.utils import find_downstream_atoms_in_residue
        current_angle = self.omega
        if current_angle is not None:
            delta = math.radians(angle - current_angle)
            c = self.atom(name="C")
            n_next = self.next.atom(name="N")
            ca_next = self.next.atom(name="CA")
            if c and n_next and ca_next:
                axis = np.array(n_next.location) - np.array(c.location)
                point = c.location
                atoms_to_rotate = find_downstream_atoms_in_residue(ca_next, n_next)
                Atom.rotate_atoms(delta, axis, *atoms_to_rotate, point=point)


    def set_chi(self, n, angle):
        """Sets the nth chi angle of the residue to the given value, by rotating
        the relevant downstream atoms.

        :param int n: The chi angle to set (1-5).
        :param float angle: The angle to set, in degrees."""

        from ..utilities.utils import find_downstream_atoms_in_residue
        current_angle = self.chi(n)
        if current_angle is not None:
            delta = math.radians(angle - current_angle)

            atom_names = CHI_ANGLES[self.name][n - 1]
            atom2 = self.atom(name=atom_names[1])
            atom3 = self.atom(name=atom_names[2])

            if atom2 and atom3:
                axis = np.array(atom3.location) - np.array(atom2.location)
                point = atom2.location
                atoms_to_rotate = find_downstream_atoms_in_residue(atom3, atom2)
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
