import re
import numpy as np

def parse_nco_data(file_path, var_name):
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex to find the specified array variable
    var_regex = re.compile(r"nco_struct\s+" + re.escape(var_name) + r"\[\]\s*=\s*\{(.*?)\};", re.DOTALL)
    match = var_regex.search(content)

    if not match:
        print(f"Variable '{var_name}' not found.")
        return None

    data_str = match.group(1)

    # Regex to parse each struct element
    struct_regex = re.compile(r"\{\s*\{\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\},\s*\{(.*?)\}\s*\}", re.DOTALL)

    parsed_data = []
    for struct_match in struct_regex.finditer(data_str):
        bins = [int(b) for b in struct_match.groups()[:3]]
        floats_str = struct_match.group(4)

        # Clean up and split the float values
        floats_str = re.sub(r'\s+', ' ', floats_str).strip()
        floats = [float(x) for x in floats_str.split(',') if x.strip()]

        # Reshape the floats into a 8x3 matrix
        if len(floats) == 24:
            float_matrix = np.array(floats).reshape((8, 3)).tolist()
            parsed_data.append((bins, float_matrix))

    # Define the dtype for the structured array
    dtype = [('bins', 'i4', 3), ('data', 'f8', (8, 3))]
    structured_array = np.array([tuple(x) for x in parsed_data], dtype=dtype)

    return structured_array.tolist()

def main():
    file_path = 'c_legacy/src/nco_data.h'

    nco_stat = parse_nco_data(file_path, 'nco_stat')
    nco_stat_pro = parse_nco_data(file_path, 'nco_stat_pro')

    with open('pulchra/data.py', 'a') as f:
        f.write("\n\n")
        f.write("NCO_STAT = ")
        f.write(repr(nco_stat))
        f.write("\n\n")
        f.write("NCO_STAT_PRO = ")
        f.write(repr(nco_stat_pro))
        f.write("\n")

if __name__ == "__main__":
    main()
