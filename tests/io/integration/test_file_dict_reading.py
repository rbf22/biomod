import pytest
from biomod.io import io

class TestMmcifFileDictReading:

    def test_1lol_file_dict(self):
        d = io.open("tests/io/integration/files/1lol.cif", file_dict=True)
        assert d["entry"] == [{"id": "1LOL"}]
        assert d["audit_author"] == [
         {"name": "Wu, N.", "pdbx_ordinal": "1"},
         {"name": "Pai, E.F.", "pdbx_ordinal": "2"}
        ]
        assert d["struct"][0]["title"][:7] == "Crystal"
        entity = d["entity"]
        assert len(entity) == 4
        assert (
         entity[0]["pdbx_description"] == "orotidine 5'-monophosphate decarboxylase"
        )
        assert d["entity_poly"][0]["pdbx_seq_one_letter_code"].startswith("LRSRRVDVM")
        assert d["entity_poly"][0]["pdbx_seq_one_letter_code"].endswith("IESIKDLLIPE")
        assert entity[1]["type"] == "non-polymer"
        assert d["citation"][0]["title"].startswith("Crystal")
        assert d["citation"][0]["title"].endswith("decarboxylase.")



class TestMmtfFileDictReading:

    def test_1lol_file_dict(self):
        d = io.open("tests/io/integration/files/1lol.mmtf", file_dict=True)
        assert d["mmtfVersion"] == "1.0.0"
        assert len(d["unitCell"]) == 6
        assert d["unitCell"][0] == pytest.approx(57.57, abs=0.00005)
        assert d["resolution"] == pytest.approx(1.9, abs=0.00005)
        assert d["numAtoms"] == 3431
        assert len(d["secStructList"]) == 602
        assert d["secStructList"][:5] == (7, 4, 4, 4, 3)
        assert len(d["bondAtomList"]) == 828
        assert d["bondAtomList"][:6] == (7, 2, 15, 9, 23, 17)
        assert d["chainIdList"] == list("ABCDEFGH")
        assert d["insCodeList"] == [""] * 602
        assert d["sequenceIndexList"][:6] == [10, 11, 12, 13, 14, 15]
        assert d["occupancyList"] == [1.0] * 3431
        assert d["xCoordList"][:3] == [3.696, 3.198, 3.914]
        assert d["bFactorList"][:3] == [21.5, 19.76, 19.29]
        assert d["groupList"][0]["groupName"] == "ASN"
        assert d["groupList"][0]["atomNameList"][:3] == ["N", "CA", "C"]


    def test_1igt_file_dict(self):
        d = io.open("tests/io/integration/files/1igt.mmtf", file_dict=True)
        assert d["mmtfVersion"] == "1.0.0"
        assert d["insCodeList"][266] == "A"



class TestPdbFileDictReading:

    def test_1lol_file_dict(self):
        d = io.open("tests/io/integration/files/1lol.pdb", file_dict=True)
        assert d["HEADER"] == [
         "HEADER    LYASE                                   06-MAY-02   1LOL"
        ]
        assert d["HETNAM"] == [
         "HETNAM     BU2 1,3-BUTANEDIOL", "HETNAM     XMP XANTHOSINE-5'-MONOPHOSPHATE"
        ]
        assert d["CONECT"][0] == "CONECT 3194 3195 3196"
        assert len(d["REMARK"].keys()) == 16
        assert d["REMARK"]["2"][1] == "REMARK   2 RESOLUTION.    1.90 ANGSTROMS."
        assert len(d["MODEL"]) == 1
        assert len(d["MODEL"][0]) == 3433
        assert (
         d["MODEL"][0][0]
         == "ATOM      1  N   VAL A  11       3.696  33.898  63.219  1.00 21.50           N"
        )


    def test_5xme_file_dict(self):
        d = io.open("tests/io/integration/files/5xme.pdb", file_dict=True)
        assert d["HEADER"] == [
         "HEADER    APOPTOSIS                               15-MAY-17   5XME"
        ]
        assert len(d["MODEL"]) == 10
        assert len(d["MODEL"][0]) == 1828
        assert (
         d["MODEL"][1][4]
         == "ATOM      5  CB  ALA A 199      36.093  -8.556  -1.452  1.00  0.00           C"
        )