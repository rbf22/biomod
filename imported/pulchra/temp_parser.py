# temp_parser.py
import re
import numpy as np

def parse_nco_data(file_path, var_name):
    with open(file_path, 'r') as f:
        content = f.read()

    # Isolate the specific array we want to parse
    start_str = f"{var_name}[] =  {{ {{"
    start_idx = content.find(start_str)
    if start_idx == -1:
        print(f"# WARNING: Variable {var_name} not found in {file_path}.")
        return []

    # Find the closing "}};" of the array
    end_idx = content.find("}};", start_idx)
    if end_idx == -1:
        print(f"# WARNING: Could not find end of array for {var_name}.")
        return []

    array_content = content[start_idx + len(start_str) - 4 : end_idx + 1]

    # Use regex to find all the struct blocks. This is a bit brittle.
    # It assumes the format is always { {b,b,b}, { {d,d,d}, ... } },
    struct_pattern = re.compile(r"\{\s*\{\s*([^{}]+?)\s*\},\s*\{([^{}]+?)\}\s*\},", re.DOTALL)

    records = []
    for match in struct_pattern.finditer(array_content):
        bins_str = match.group(1)
        coords_str = match.group(2)

        try:
            bins = [int(x) for x in bins_str.replace(' ', '').split(',') if x]

            coords_flat_str = coords_str.replace('\n', ' ').replace(',', ' ')
            coords_flat = [float(x) for x in coords_flat_str.split() if x]

            # Reshape into 8x3 array
            coords = np.array(coords_flat).reshape(8, 3)

            records.append((bins, coords.tolist()))
        except (ValueError, IndexError) as e:
            print(f"# WARNING: Failed to parse a record for {var_name}. Error: {e}")
            continue

    return records

def print_numpy_array(records, var_name):
    if not records:
        return
    # Print the python code to create the numpy array
    print(f"# Data ported from nco_data.h for {var_name}")
    print(f"import numpy as np")
    print("")
    print(f"{var_name.upper()} = np.array([")
    for bins, coords in records:
        # Manually format the output to be somewhat readable
        print(f"    ({bins}, [", end="")
        for i, row in enumerate(coords):
            if i > 0:
                print(" " * (len(str(bins)) + 6), end="")
            print(f"[{row[0]:9.3f}, {row[1]:9.3f}, {row[2]:9.3f}]", end="")
            if i < len(coords) - 1:
                print(",")
            else:
                print("]),")
    print(f"""], dtype=[('bins', 'i4', (3,)), ('data', 'f8', (8, 3))])""")
    print("")


if __name__ == '__main__':
    nco_records = parse_nco_data('c_legacy/src/nco_data.h', 'nco_struct nco_stat')
    print_numpy_array(nco_records, 'NCO_STAT')

    nco_pro_records = parse_nco_data('c_legacy/src/nco_data.h', 'nco_struct nco_stat_pro')
    print_numpy_array(nco_pro_records, 'NCO_STAT_PRO')
