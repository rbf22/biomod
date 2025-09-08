class Atom:
    def __init__(self, x, y, z, name, num, locnum, flag, cispro, res):
        self.x = x
        self.y = y
        self.z = z
        self.name = name
        self.num = num
        self.locnum = locnum
        self.flag = flag
        self.cispro = cispro
        self.res = res
        self.prev = None
        self.next = None

class Residue:
    def __init__(self, num, locnum, natoms, type, pdbsg, protein, name, chain):
        self.num = num
        self.locnum = locnum
        self.natoms = natoms
        self.type = type
        self.pdbsg = pdbsg
        self.protein = protein
        self.name = name
        self.chain = chain
        self.atoms = []
        self.sgx = 0.0
        self.sgy = 0.0
        self.sgz = 0.0
        self.cmx = 0.0
        self.cmy = 0.0
        self.cmz = 0.0
        self.prev = None
        self.next = None

    def add_or_replace_atom(self, name, x, y, z, flag):
        for atom in self.atoms:
            if atom.name == name:
                atom.x = x
                atom.y = y
                atom.z = z
                atom.flag |= flag
                return

        new_atom = Atom(x, y, z, name, 0, 0, flag, False, self)

        if name == "N":
            self.atoms.insert(0, new_atom)
        else:
            self.atoms.append(new_atom)
        self.natoms += 1

class Molecule:
    def __init__(self, name):
        self.name = name
        self.residues = []
        self.nres = 0
        self.r14 = None
        self.seq = None
        self.contacts = None
        self.cutoffs = None
        self.prev = None
        self.next = None
