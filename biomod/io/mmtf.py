"""Contains functions for dealing with the .mmtf file format with cycle protection and debug logging."""

import msgpack
import struct
from datetime import datetime
from .mmcif import get_structure_from_atom, create_entities, split_residue_id
from ..core.base import Chain
from ..core.residues import Ligand
import logging
import time

# Configure logging for debugging cycle detection
logger = logging.getLogger("mmtf_debug")
logging.basicConfig(level=logging.DEBUG)


class CycleDetectionError(Exception):
    """Raised when a cycle is detected during MMTF serialization."""
    pass


class MMTFSerializationContext:
    """Context manager for tracking visited objects during MMTF serialization with debug logging."""

    def __init__(self, max_depth=1000):
        self.visited_objects = set()
        self.current_depth = 0
        self.max_depth = max_depth
        self.path_stack = []

    def enter_object(self, obj, obj_type=None):
        obj_id = id(obj)
        obj_desc = f"{obj_type}:{obj_id}" if obj_type else str(obj_id)
        print(f"[DEBUG] Entering object {obj_desc}, depth {self.current_depth}, visited {len(self.visited_objects)}")

        if obj_id in self.visited_objects:
            logger.warning(f"Cycle detected at {obj_desc}, path: {' -> '.join(self.path_stack)}")
            raise CycleDetectionError(f"Cycle detected while processing {obj_desc}")

        if self.current_depth >= self.max_depth:
            logger.warning(f"Maximum recursion depth {self.max_depth} exceeded at {obj_desc}")
            raise CycleDetectionError(f"Maximum recursion depth exceeded at {obj_desc}")

        self.visited_objects.add(obj_id)
        self.path_stack.append(obj_desc)
        self.current_depth += 1
        return True

    def exit_object(self, obj):
        obj_id = id(obj)
        if obj_id in self.visited_objects:
            self.visited_objects.remove(obj_id)
        if self.path_stack:
            self.path_stack.pop()
        self.current_depth = max(0, self.current_depth - 1)
        print(f"[DEBUG] Exiting object {obj_id}, current depth {self.current_depth}")


def mmtf_bytes_to_mmtf_dict(bytestring):
    raw = msgpack.unpackb(bytestring)
    context = MMTFSerializationContext()
    return decode_dict(raw, context)


def decode_dict(d, context=None):
    if context is None:
        context = MMTFSerializationContext()

    context.enter_object(d, "dict")
    try:
        print(f"[DEBUG] Decoding dictionary with {len(d)} keys")
        new = {}
        for key, value in d.items():
            try:
                new_value = value.decode()
            except Exception:
                new_value = value

            if isinstance(new_value, str) and new_value and new_value[0] == "\x00":
                new_value = new_value.encode()

            if isinstance(new_value, bytes):
                new_value = parse_binary_field(new_value)

            if isinstance(new_value, list) and new_value:
                if isinstance(new_value[0], dict):
                    new_value = [decode_dict(x, context) for x in new_value]
                elif isinstance(new_value[0], bytes):
                    new_value = [x.decode() for x in new_value]

            new[key.decode() if isinstance(key, bytes) else key] = new_value
        return new
    finally:
        context.exit_object(d)


def parse_binary_field(b):
    codec, length, params = struct.unpack(">iii", b[:12])

    def len4(b):
        return int(len(b[12:]) / 4)

    if codec == 1:
        return struct.unpack("f" * length, b[12:])
    elif codec == 2:
        return struct.unpack("b" * length, b[12:])
    elif codec == 3:
        return struct.unpack(">" + "h" * length, b[12:])
    elif codec == 4:
        return struct.unpack(">" + "i" * length, b[12:])
    elif codec == 5:
        chars = struct.unpack("c" * (length * 4), b[12:])
        return [b"".join([c for c in chars[i*4:(i+1)*4] if c != b"\x00"]).decode() for i in range(length)]
    elif codec == 6:
        integers = struct.unpack(">" + "i" * len4(b), b[12:])
        return [chr(c) if c != 0 else "" for c in run_length_decode(integers)]
    elif codec == 7:
        integers = struct.unpack(">" + "i" * len4(b), b[12:])
        return run_length_decode(integers)
    elif codec == 8:
        integers = struct.unpack(">" + "i" * len4(b), b[12:])
        return delta_decode(run_length_decode(integers))
    elif codec == 9:
        integers = struct.unpack(">" + "i" * len4(b), b[12:])
        return [n / params for n in run_length_decode(integers)]
    elif codec == 10:
        integers = struct.unpack(">" + "h" * int(len(b[12:])/2), b[12:])
        return [n / params for n in delta_decode(recursive_decode(integers))]
    else:
        raise ValueError(f".mmtf error: {codec} is invalid codec")


def run_length_decode(integers):
    x = []
    for index, val in enumerate(integers[::2]):
        x += [val] * integers[1::2][index]
    return x


def delta_decode(integers):
    array, last = [], 0
    for i in integers:
        last += i
        array.append(last)
    return array


def recursive_decode(integers, bits=16):
    new = []
    power = 2 ** (bits - 1)
    cutoff = [power - 1, 0 - power]
    index = 0
    while index < len(integers):
        value = 0
        while integers[index] in cutoff:
            value += integers[index]
            index += 1
            if index >= len(integers) or integers[index] == 0:
                break
        if index < len(integers):
            value += integers[index]
            index += 1
        new.append(value)
    return new


def mmtf_dict_to_data_dict(mmtf_dict):
    data_dict = {
        "description": {"code": None, "title": None, "deposition_date": None, "classification": None, "keywords": [], "authors": []},
        "experiment": {"technique": None, "source_organism": None, "expression_system": None, "missing_residues": []},
        "quality": {"resolution": None, "rvalue": None, "rfree": None},
        "geometry": {"assemblies": [], "crystallography": {}},
        "models": []
    }

    mmtf_to_data_transfer(mmtf_dict, data_dict, "description", "code", "structureId")
    mmtf_to_data_transfer(mmtf_dict, data_dict, "description", "title", "title")
    mmtf_to_data_transfer(mmtf_dict, data_dict, "description", "deposition_date", "depositionDate", date=True)
    mmtf_to_data_transfer(mmtf_dict, data_dict, "experiment", "technique", "experimentalMethods", first=True)
    mmtf_to_data_transfer(mmtf_dict, data_dict, "quality", "resolution", "resolution", trim=3)
    mmtf_to_data_transfer(mmtf_dict, data_dict, "quality", "rvalue", "rWork", trim=3)
    mmtf_to_data_transfer(mmtf_dict, data_dict, "quality", "rfree", "rFree", trim=3)
    mmtf_to_data_transfer(mmtf_dict, data_dict["geometry"], "crystallography", "space_group", "spaceGroup")
    mmtf_to_data_transfer(mmtf_dict, data_dict["geometry"], "crystallography", "unit_cell", "unitCell", trim=3)

    if data_dict["geometry"]["crystallography"].get("space_group") == "NA":
        data_dict["geometry"]["crystallography"] = {}

    data_dict["geometry"]["assemblies"] = [{
        "id": int(a["name"]),
        "software": None,
        "delta_energy": None,
        "buried_surface_area": None,
        "surface_area": None,
        "transformations": [{
            "chains": [mmtf_dict["chainIdList"][i] for i in t["chainIndexList"]],
            "matrix": [t["matrix"][n*4:(n*4)+3] for n in range(3)],
            "vector": t["matrix"][3:-4:4]
        } for t in a.get("transformList", [])]
    } for a in mmtf_dict.get("bioAssemblyList", [])]

    context = MMTFSerializationContext()
    update_models_list(mmtf_dict, data_dict, context)
    print("[DEBUG] Completed mmtf_dict_to_data_dict")
    return data_dict


def update_models_list(mmtf_dict, data_dict, context=None):
    if context is None:
        context = MMTFSerializationContext()

    atoms = get_atoms_list(mmtf_dict)
    group_definitions = get_group_definitions_list(mmtf_dict)
    groups = get_groups_list(mmtf_dict, group_definitions)
    chains = get_chains_list(mmtf_dict, groups)

    for model_num in range(mmtf_dict["numModels"]):
        model = {"polymer": {}, "non-polymer": {}, "water": {}, "branched": {}}
        for chain_num in range(mmtf_dict["chainsPerModel"][model_num]):
            if chain_num < len(chains):
                chain = chains[chain_num]
                print(f"[DEBUG] Adding chain {chain['id']} to model {model_num}")
                add_chain_to_model(chain, model, atoms, context)
        data_dict["models"].append(model)

def get_atoms_list(mmtf_dict):
    """Creates a list of atom dictionaries from a .mmtf dictionary with debug."""
    print(f"[DEBUG] Generating atoms list, total atoms: {len(mmtf_dict.get('xCoordList', []))}")
    return [{
        "x": x, "y": y, "z": z, "alt_loc": a or None, "bvalue": b, "occupancy": o,
        "id": i, "is_hetatm": False
    } for x, y, z, a, b, i, o in zip(
        mmtf_dict["xCoordList"], mmtf_dict["yCoordList"], mmtf_dict["zCoordList"],
        mmtf_dict["altLocList"], mmtf_dict["bFactorList"], mmtf_dict["atomIdList"],
        mmtf_dict["occupancyList"]
    )]


def get_group_definitions_list(mmtf_dict):
    """Returns group definitions with atom dictionaries."""
    group_definitions = []
    for group in mmtf_dict["groupList"]:
        atoms = [{"name": name, "element": element.upper(), "charge": charge}
                 for name, element, charge in zip(group["atomNameList"], group["elementList"], group["formalChargeList"])]
        group_definitions.append({"name": group["groupName"], "atoms": atoms})
    print(f"[DEBUG] Created {len(group_definitions)} group definitions")
    return group_definitions


def get_groups_list(mmtf_dict, group_definitions):
    """Zips group IDs and types into group dictionaries."""
    sec_struct = ["helices", None, "helices", "strands", "helices", "strands", None, None]
    groups = [{
        "number": id_, "insert": insert, "secondary_structure": sec_struct[ss] if ss >= 0 else None, **group_definitions[type_]
    } for id_, insert, ss, type_ in zip(
        mmtf_dict["groupIdList"], mmtf_dict["insCodeList"],
        mmtf_dict.get("secStructList", [-1] * len(mmtf_dict["groupIdList"])),
        mmtf_dict["groupTypeList"]
    )]
    print(f"[DEBUG] Generated {len(groups)} groups")
    return groups


def get_chains_list(mmtf_dict, groups):
    """Generates chain dictionaries from groups."""
    chains = []
    for i_id, id_, group_num in zip(mmtf_dict["chainIdList"], mmtf_dict["chainNameList"], mmtf_dict["groupsPerChain"]):
        chain = {"id": id_, "internal_id": i_id, "groups": groups[:group_num]}
        del groups[:group_num]
        for entity in mmtf_dict["entityList"]:
            if len(chains) in entity["chainIndexList"]:
                chain["type"] = entity["type"]
                chain["sequence"] = entity.get("sequence", "")
                chain["full_name"] = entity.get("description", None)
                break
        chains.append(chain)
    print(f"[DEBUG] Generated {len(chains)} chains")
    return chains


def add_chain_to_model(chain, model, atoms, context):
    context.enter_object(chain, f"chain_{chain['id']}")
    try:
        print(f"[DEBUG] Adding chain {chain['id']} of type {chain.get('type')} to model")
        if chain["type"] in ("polymer", "branched"):
            polymer = {"internal_id": chain["internal_id"], "sequence": chain["sequence"], "helices": [], "strands": [], "residues": {}}
            for i, group in enumerate(chain["groups"], start=1):
                add_het_to_dict(group, chain, atoms, polymer["residues"], number=i, context=context)
            add_ss_to_chain(polymer)
            model["polymer"][chain["id"]] = polymer
        else:
            for group in chain["groups"]:
                add_het_to_dict(group, chain, atoms, model[chain["type"]], context=context)
    finally:
        context.exit_object(chain)


def add_het_to_dict(group, chain, atoms, d, number=None, context=None):
    if context is None:
        context = MMTFSerializationContext()
    context.enter_object(group, f"group_{group.get('name', 'unknown')}")
    try:
        het_id = f"{chain['id']}.{group['number']}{group['insert']}"
        atom_count = len(group.get("atoms", []))
        if len(atoms) < atom_count:
            logger.warning(f"Not enough atoms ({len(atoms)}) for group {het_id} requiring {atom_count}")
            het_atoms = atoms[:]
            atoms.clear()
        else:
            het_atoms = atoms[:atom_count]
            del atoms[:atom_count]
        het_atoms = {a["id"]: {"anisotropy": [0]*6, **a, **g_a} for a, g_a in zip(het_atoms, group.get("atoms", []))}
        for a in het_atoms.values():
            a.pop("id", None)
        het = {"name": group["name"], "atoms": het_atoms, "full_name": None, "secondary_structure": group["secondary_structure"]}
        if number is None:
            het["internal_id"] = chain["internal_id"]
            het["polymer"] = chain["id"]
            het["full_name"] = chain.get("full_name")
        else:
            het["number"] = number
        d[het_id] = het
        print(f"[DEBUG] Added het {het_id} with {len(het_atoms)} atoms")
    finally:
        context.exit_object(group)


def add_ss_to_chain(chain):
    """Updates polymer with secondary structure info."""
    in_ss = {"helices": False, "strands": False}
    for res_id, res in chain["residues"].items():
        ss = res.get("secondary_structure")
        if ss:
            if not in_ss[ss]:
                chain[ss].append([])
            in_ss[ss] = True
            chain[ss][-1].append(res_id)
        else:
            in_ss["helices"] = False
            in_ss["strands"] = False
        res.pop("secondary_structure", None)
    print(f"[DEBUG] Added secondary structure info to chain {chain.get('internal_id')}")


def mmtf_to_data_transfer(mmtf_dict, data_dict, d_cat, d_key, m_key, date=False, first=False, trim=False):
    try:
        value = mmtf_dict[m_key]
        if date:
            value = datetime.strptime(value, "%Y-%m-%d").date()
        if first and isinstance(value, (list, tuple)):
            value = value[0]
        if trim:
            try:
                value = [round(v, trim) for v in value]
            except Exception:
                value = round(value, trim)
        data_dict[d_cat][d_key] = value
        print(f"[DEBUG] Transferred {m_key} to {d_cat}.{d_key} => {value}")
    except Exception:
        print(f"[DEBUG] {m_key} not found in mmtf_dict")


def structure_to_mmtf_string(structure):
    """Converts a structure to an MMTF bytestring with cycle protection.

    Includes detailed debugging for atoms, chains, ligands, and waters.
    No compression is currently performed.

    :param AtomStructure structure: the structure to convert.
    :rtype: bytes
    """
    context = MMTFSerializationContext()
    print(f"[DEBUG] Entering object structure:{id(structure)}, depth {context.current_depth}, visited {len(context.visited_objects)}")

    try:
        # Extract chains, ligands, waters, atom properties, and entities
        chains, ligands, waters, properties, entities = get_structures(structure, context)
        print(f"[DEBUG] Extracted {len(properties)} atom properties")
        print(f"[DEBUG] Created {len(entities)} entities for MMTF")

        # Prepare MMTF lists
        entity_list = get_entity_list(entities, chains, ligands, waters)
        chain_ids, chain_names = get_chain_ids_and_names(chains, ligands, waters)
        groups_per_chain = get_groups_per_chain(chains, ligands, waters)
        print(f"[DEBUG] Prepared {len(chain_ids)} chain IDs and names")
        print(f"[DEBUG] Calculated groups per chain: {groups_per_chain}")

        # Debug first few waters
        print(f"[DEBUG] Number of waters: {len(waters)}")
        for w in waters[:5]:
            print(f"[DEBUG] Water {w._name} in chain {w.chain.id}, atoms: {[a.id for a in w.atoms()]}")

        # Prepare groups for MMTF
        group_types, group_ids, groups, ins = get_groups(chains, ligands, waters, context)

        if not properties:
            logger.warning("No atom properties found in structure")
            return msgpack.packb({})

        # Separate atom properties
        x, y, z, alt, bfactor, ids, occupancy = zip(*properties)
        chain_count = len(chains) + len(ligands) + len(set(w.chain for w in waters))

        # Final MMTF dictionary
        d = {
            "numModels": 1,
            "numChains": chain_count,
            "chainsPerModel": [chain_count],
            "xCoordList": x,
            "yCoordList": y,
            "zCoordList": z,
            "altLocList": alt,
            "bFactorList": bfactor,
            "atomIdList": ids,
            "occupancyList": occupancy,
            "entityList": entity_list,
            "chainIdList": chain_ids,
            "insCodeList": ins,
            "chainNameList": chain_names,
            "groupsPerChain": groups_per_chain,
            "groupList": groups,
            "groupIdList": group_ids,
            "groupTypeList": group_types
        }

        print("[DEBUG] Packed structure into MMTF dictionary")
        print("[DEBUG] About to pack MMTF dictionary into bytes...")

        start = time.time()
        try:
            packed = msgpack.packb(d)
        except Exception as e:
            print(f"[ERROR] msgpack.packb failed: {e}")
            raise
        end = time.time()
        print(f"[DEBUG] msgpack.packb completed in {end - start:.3f}s, size: {len(packed)} bytes")

        return packed

    except CycleDetectionError as e:
        logger.error(f"Cycle detected during MMTF serialization: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during MMTF serialization: {e}")
        raise
    finally:
        print(f"[DEBUG] Exiting object structure:{id(structure)}, current depth {context.current_depth}")

# --- Remaining helper functions for structures, entities, chain IDs, groups ---

def get_structures(structure, context):
    chains, ligands, waters, atom_properties = set(), set(), set(), []
    context.enter_object(structure, "structure")
    try:
        for atom in sorted(structure.atoms(), key=lambda a: a.id):
            get_structure_from_atom(atom, chains, ligands, waters)
            atom_properties.append(list(atom.location) + ["", atom.bvalue, atom.id, 1])
        chains = sorted(chains, key=lambda c: c._internal_id)
        ligands = sorted(ligands, key=lambda lig: lig._internal_id)
        waters = sorted(waters, key=lambda w: w._internal_id)
        entities = create_entities(chains, ligands, waters)
        print(f"[DEBUG] Extracted {len(atom_properties)} atom properties")
        return (chains, ligands, waters, atom_properties, entities)
    finally:
        context.exit_object(structure)


def get_entity_list(entities, chains, ligands, waters):
    entity_list = []
    for e in entities:
        if isinstance(e, Chain):
            entity_list.append({"type": "polymer", "chainIndexList": [i for i, c in enumerate(chains) if c.sequence == e.sequence], "sequence": e.sequence})
        elif isinstance(e, Ligand) and not e.is_water:
            entity_list.append({"type": "non-polymer", "chainIndexList": [i + len(chains) for i, lig in enumerate(ligands) if lig._name == e._name]})
        else:
            water_chains = set(w.chain for w in waters)
            entity_list.append({"type": "water", "chainIndexList": [i + len(chains) + len(ligands) for i in range(len(water_chains))]})
    print(f"[DEBUG] Created {len(entity_list)} entities for MMTF")
    return entity_list


def get_chain_ids_and_names(chains, ligands, waters):
    chain_ids, chain_names = [], []
    for chain in chains:
        chain_ids.append(chain._internal_id)
        chain_names.append(chain.id)
    for ligand in ligands:
        chain_ids.append(ligand._internal_id)
        chain_names.append(ligand.chain.id)
    for water in waters:
        if water._internal_id not in chain_ids:
            chain_ids.append(water._internal_id)
            chain_names.append(water.chain.id)
    print(f"[DEBUG] Prepared {len(chain_ids)} chain IDs and names")
    return (chain_ids, chain_names)

def get_groups(chains, ligands, waters, context):
    """Creates the relevant lists of group information from chains, ligands, and waters.

    :param list chains: the chains to pack.
    :param list ligands: the ligands to pack.
    :param list waters: the waters to pack.
    :param MMTFSerializationContext context: context for cycle detection.
    :rtype: tuple
    """
    group_types, group_ids, groups, inserts = [], [], [], []
    for chain in chains:
        for res in chain.residues():
            add_het_to_groups(res, group_types, group_ids, groups, inserts, context)
    for het in ligands + waters:
        add_het_to_groups(het, group_types, group_ids, groups, inserts, context)
    return group_types, group_ids, groups, inserts


def add_het_to_groups(het, group_type_list, group_id_list, group_list, ins_list, context):
    """Updates group lists with information from a single Het.

    :param Het het: the Het to pack.
    :param list group_type_list: the list of group types.
    :param list group_id_list: the list of group IDs.
    :param list group_list: the list of groups.
    :param list ins_list: the list of insertion codes.
    :param MMTFSerializationContext context: context for cycle detection.
    """
    context.enter_object(het, f"het_{het._name}")
    try:
        atoms = sorted(het.atoms(), key=lambda a: a.id)
        group = {
            "groupName": het._name,
            "atomNameList": [a._name for a in atoms],
            "elementList": [a.element for a in atoms],
            "formalChargeList": [a.charge for a in atoms],
        }

        # Use existing group if present
        for i, g in enumerate(group_list):
            if g == group:
                group_type_list.append(i)
                break
        else:
            group_list.append(group)
            group_type_list.append(len(group_list) - 1)

        if atoms:
            id_, insert = split_residue_id(atoms[0])
            group_id_list.append(id_)
            ins_list.append(insert if insert != "?" else "")
        else:
            group_id_list.append(0)
            ins_list.append("")
    finally:
        context.exit_object(het)

def get_groups_per_chain(chains, ligands, waters):
    groups_per_chain = []
    for chain in chains:
        groups_per_chain.append(len(chain.residues()))
    for ligand in ligands:
        groups_per_chain.append(1)
    water_chains = sorted(set(w._internal_id for w in waters))
    for wc in water_chains:
        groups_per_chain.append(len([w for w in waters if w._internal_id == wc]))
    print(f"[DEBUG] Calculated groups per chain: {groups_per_chain}")
    return