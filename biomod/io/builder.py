"""Contains logic for turning data dictionaies into a parsed Python objects."""

from ..core.atoms import Atom
from ..core.residues import Residue, Ligand
from ..core.base import Chain, Model
from ..core.constants import *

class File:
    """When a file is parsed, the result is a ``File``. It contains the
    structure of interest, as well as meta information.

    :param str filetype: the type of file that was parsed to make this."""

    def __init__(self, filetype):
        self._filetype = filetype
        self._models = []


    def __repr__(self):
        return "<{}.{} File>".format(self._code or "", self._filetype)


    @property
    def filetype(self):
        """The filetype that this File was created from, such as .pdb or
        .cif.

        :rtype: ``str``"""

        return self._filetype


    @property
    def code(self):
        """The unique database identifer for this structure.

        :rtype: ``str``"""

        return self._code


    @property
    def title(self):
        """The structure's text description.

        :rtype: ``str``"""

        return self._title


    @property
    def deposition_date(self):
        """The date the structure was submitted for publication.

        :rtype: ``datetime.date``"""

        return self._deposition_date


    @property
    def classification(self):
        """The structure's formal classification.

        :rtype: ``str``"""

        return self._classification


    @property
    def keywords(self):
        """The structure's keyword descriptors.

        :rtype: ``list``"""

        return self._keywords


    @property
    def authors(self):
        """The structure's authors.

        :rtype: ``list``"""

        return self._authors


    @property
    def technique(self):
        """The structure's experimental technique.

        :rtype: ``str``"""

        return self._technique


    @property
    def source_organism(self):
        """The structure's original organism.

        :rtype: ``float``"""

        return self._source_organism


    @property
    def expression_system(self):
        """The organism the structure was expressed in.

        :rtype: ``float``"""

        return self._expression_system


    @property
    def missing_residues(self):
        """The residues that should be in the model but aren't.

        :rtype: ``list``"""

        return self._missing_residues


    @property
    def resolution(self):
        """The structure's resolution.

        :rtype: ``float``"""

        return self._resolution


    @property
    def rvalue(self):
        """The structure's R-value.

        :rtype: ``float``"""

        return self._rvalue


    @property
    def rfree(self):
        """The structure's R-free value.

        :rtype: ``float``"""

        return self._rfree


    @property
    def assemblies(self):
        """The structure's biological assembly instructions.

        :rtype: ``list``"""

        return self._assemblies


    @property
    def models(self):
        """The structure's models.

        :rtype: ``list``"""

        return self._models


    @property
    def model(self):
        """The structure's first model (and only model if it has only one).

        :rtype: ``Model``"""

        return self._models[0]


    def generate_assembly(self, id):
        """Generates a new model from the existing model using one of the file's
        set of assembly instructions (for which you provide the ID).

        For example:

            >>> import atomium
            >>> pdb = atomium.fetch('1xda')
            >>> pdb.model
            <Model (8 chains, 16 ligands)>
            >>> pdb.generate_assembly(1)
            <Model (2 chains, 4 ligands)>
            >>> pdb.generate_assembly(5)
            <Model (12 chains, 24 ligands)>

        :param int id: the ID of the assembly to generate.
        :rtype: ``Model``"""
        
        m = self._models[0]
        for assembly in self._assemblies:
            if assembly["id"] == id:
                break
        else:
            raise ValueError(f"No assembly with ID {id}")
        all_structures = []
        for t in assembly["transformations"]:
            structures = {}
            for chain_id in t["chains"]:
                for obj in list(m.chains()) + list(m.ligands() | m.waters()):
                    if obj._internal_id == chain_id:
                        copy = obj.copy()
                        if isinstance(copy, Ligand):
                            copy._chain = structures.get(obj.chain)
                        structures[obj] = copy
            atoms = set()
            for s in structures.values():
                atoms.update(s.atoms())
            Atom.transform_atoms(t["matrix"], *atoms)
            Atom.translate_atoms(t["vector"], *atoms)
            all_structures += structures.values()
        return Model(*all_structures)


def data_dict_to_file(data_dict, filetype):
    """Turns an atomium data dictionary into a :py:class:`.File`.

    :param dict data_dict: the data dictionary to parse.
    :param str filetype: the file type that is being converted.
    :rtype: ``File``"""

    print("Converting data_dict to file")
    f = File(filetype)
    for key in data_dict.keys():
        if key != "models":
            for subkey, value in data_dict[key].items():
                setattr(f, "_" + subkey, value)
    f._models = [model_dict_to_model(m) for m in data_dict["models"]]
    return f


def model_dict_to_model(model_dict):
    """Takes a model dictionary and turns it into a fully processed
    :py:class:`.Model` object.

    :param dict model_dict: the model dictionary.
    :rtype: ``Model``"""


    print("Converting model_dict to model")
    chains = create_chains(model_dict)
    ligands = create_ligands(model_dict, chains)
    waters = create_ligands(model_dict, chains, water=True)
    model = Model(*(chains + ligands + waters))
    return model


def create_chains(model_dict):
    """Creates a list of :py:class:`.Chain` objects from a model dictionary.

    :param dict model_dict: the model dictionary.
    :rtype: ``list``"""

    chains = []
    for chain_id, chain in model_dict["polymer"].items():
        res = [create_het(r, i) for i, r in sorted(
         chain["residues"].items(), key=lambda x: x[1]["number"]
        )]
        res_by_id = {r.id: r for r in res}
        for res1, res2 in zip(res[:-1], res[1:]):
            res1._next, res2._previous = res2, res1
        chains.append(
            Chain(
                *res,
                id=chain_id,
                helices=[[res_by_id[r] for r in h] for h in chain["helices"]],
                strands=[[res_by_id[r] for r in s] for s in chain["strands"]],
                information=chain["information"] if "information" in chain else [],
                internal_id=chain["internal_id"],
                sequence=chain["sequence"],
            )
        )
    return chains


def create_ligands(model_dict, chains, water=False):
    """Creates a list of :py:class:`.Ligand` objects from a model dictionary.

    :param dict model_dict: the model dictionary.
    :param list chains: a list of :py:class:`.Chain` objects to assign by ID.
    :param bool water: if `True``, water ligands will be made.
    :rtype: ``list``"""

    ligands = []
    for lig_id, lig in model_dict["water" if water else "non-polymer"].items():
        chain = None
        for c in chains:
            if c._id == lig["polymer"]:
                chain = c
                break
        ligands.append(
         create_het(lig, lig_id, ligand=True, chain=chain, water=water)
        )
    return ligands


def create_het(d, id, ligand=False, chain=None, water=False):
    """Creates a :py:class:`.Residue` or :py:class:`.Ligand` from some
    atom-containing dictionary.

    If there is multiple occupancy, only one position will be used.

    :param dict d: the dictionary to parse.
    :param str id: the ID of the structure to make.
    :param bool ligand: if ``True`` a ligand will be made, not a residue.
    :param Chain chain: the :py:class:`.Chain` to assign if a ligand.
    :param bool water: if ``True``, the ligand will be a water ligand.
    :rtype: ``Residue`` or ``Ligand``"""

    alt_loc = None
    if any([atom["occupancy"] < 1 for atom in d["atoms"].values()]):
        if any([atom["alt_loc"] for atom in d["atoms"].values()]):
            alt_loc = sorted([atom["alt_loc"] for atom in d["atoms"].values()
             if atom["alt_loc"]])[0]
    atoms = [atom_dict_to_atom(a, i) for i, a in d["atoms"].items()
     if a["occupancy"] == 1 or a["alt_loc"] is None or a["alt_loc"] == alt_loc]
    if ligand:
        return Ligand(*atoms, id=id, name=d["name"], chain=chain,
         internal_id=d["internal_id"], water=water, full_name=d["full_name"])
    else:
        return Residue(*atoms, id=id, name=d["name"], full_name=d["full_name"])


def atom_dict_to_atom(d, atom_id):
    """Creates an :py:class:`.Atom` from an atom dictionary.

    :param dict d: the atom dictionary.
    :param int id: the atom's ID.
    :rtype: ``Atom``"""

    return Atom(
     d["element"], d["x"], d["y"], d["z"], atom_id,
     d["name"], d["charge"], d["bvalue"], d["anisotropy"], d["is_hetatm"]
    )
