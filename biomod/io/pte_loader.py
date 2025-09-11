from json import load
import os

def _read_pte(pte_file_name):
    pte = {}
    with open(pte_file_name, encoding="utf-8") as pte_file:
        data = load(pte_file)

    for row in data['Rows']:
        element = dict(zip(data['Columns'], row))
        pte[element['Symbol']] = element
        pte[element['AtomicNumber']] = element
    return pte

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
PTE_FILE_NAME = os.path.join(__location__, '../parameters/PTE.json')

PTE = _read_pte(PTE_FILE_NAME)
