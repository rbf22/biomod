from . import mmcif
from . import mmtf
from . import pdb
from . import builder

def open(path, *args, **kwargs):
    try:
        with open(path) as f:
            content = f.read()
    except Exception:
        with open(path, "rb") as f:
            content = f.read()
    return parse_string(content, path, *args, **kwargs)

def parse_string(string, filename, *args, **kwargs):
    file_dict = kwargs.get("file_dict")
    data_dict = kwargs.get("data_dict")
    to_file_dict, to_data_dict = get_parse_functions(string, filename)
    file_dict_result = to_file_dict(string)
    if file_dict:
        return file_dict_result
    data_dict_result = to_data_dict(file_dict_result)
    if data_dict:
        return data_dict_result
    return builder.data_dict_to_file(data_dict_result, filename.split(".")[-1])


def get_parse_functions(string, filename):
    if isinstance(string, bytes) or filename.endswith(".mmtf"):
        return mmtf.mmtf_bytes_to_mmtf_dict, mmtf.mmtf_dict_to_data_dict
    if "atom_site" in string or filename.endswith(".cif"):
        return mmcif.mmcif_string_to_mmcif_dict, mmcif.mmcif_dict_to_data_dict
    return pdb.pdb_string_to_pdb_dict, pdb.pdb_dict_to_data_dict

def fetch(*args, **kwargs):
    pass

def fetch_over_ssh(*args, **kwargs):
    pass

def save(string, path):
    if isinstance(string, bytes):
        with open(path, "wb") as f:
            f.write(string)
    else:
        with open(path, "w") as f:
            f.write(string)
