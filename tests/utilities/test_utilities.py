import pytest
from unittest.mock import Mock, patch, MagicMock
from biomod.core.atoms import Atom
from biomod.io.utils import (
 open, fetch, fetch_over_ssh, parse_string, get_parse_functions, save
)
from biomod.io.mmcif import mmcif_string_to_mmcif_dict, mmcif_dict_to_data_dict
from biomod.io.mmtf import mmtf_bytes_to_mmtf_dict, mmtf_dict_to_data_dict
from biomod.io.pdb import pdb_string_to_pdb_dict, pdb_dict_to_data_dict
from biomod.utilities.utils import find_downstream_atoms_in_residue

class TestDownstreamAtomFinding:

    def test_can_find_downstream_atoms(self):
        a1 = Atom("C", 0, 0, 0, 1, "C1", 0, 0, [])
        a2 = Atom("C", 0, 0, 0, 2, "C2", 0, 0, [])
        a3 = Atom("C", 0, 0, 0, 3, "C3", 0, 0, [])
        a4 = Atom("C", 0, 0, 0, 4, "C4", 0, 0, [])
        a5 = Atom("C", 0, 0, 0, 5, "C5", 0, 0, [])
        a1.bond(a2)
        a2.bond(a3)
        a3.bond(a4)
        a3.bond(a5)

        downstream = find_downstream_atoms_in_residue(a3, a2)
        assert downstream == {a3, a4, a5}


class TestOpening:

    @pytest.fixture
    def open_fixture(self):
        with patch("builtins.open") as mock_open, \
             patch("biomod.io.utils.parse_string") as mock_parse:
            open_return = MagicMock()
            mock_file = Mock()
            open_return.__enter__.return_value = mock_file
            mock_file.read.return_value = "returnstring"
            mock_open.return_value = open_return
            yield mock_open, mock_parse


    def test_can_open_string(self, open_fixture):
        mock_open, mock_parse = open_fixture
        assert open("path/to/file", 1, a=2) == mock_parse.return_value
        mock_open.assert_called_with("path/to/file")
        mock_parse.assert_called_with("returnstring", "path/to/file", 1, a=2)


    def test_can_open_bytestring(self, open_fixture):
        mock_open, mock_parse = open_fixture
        mock_open.side_effect = [Exception, mock_open.return_value]
        assert open("path/to/file", 1, a=2) == mock_parse.return_value
        mock_open.assert_called_with("path/to/file", "rb")
        mock_parse.assert_called_with("returnstring", "path/to/file", 1, a=2)


class TestFetching:

    @pytest.fixture
    def fetch_fixture(self):
        with patch("requests.get") as mock_get, \
             patch("biomod.io.utils.parse_string") as mock_parse:
            mock_get.return_value = Mock(status_code=200, text="ABC")
            yield mock_get, mock_parse


    def test_can_fetch_cif(self, fetch_fixture):
        mock_get, mock_parse = fetch_fixture
        f = fetch("1ABC", 1, b=2)
        mock_get.assert_called_with("https://files.rcsb.org/view/1abc.cif", stream=True)
        mock_parse.assert_called_with("ABC", "1ABC.cif", 1, b=2)
        assert f == mock_parse.return_value


    def test_can_fetch_pdb(self, fetch_fixture):
        mock_get, mock_parse = fetch_fixture
        f = fetch("1ABC.pdb", 1, b=2)
        mock_get.assert_called_with("https://files.rcsb.org/view/1abc.pdb", stream=True)
        mock_parse.assert_called_with("ABC", "1ABC.pdb", 1, b=2)
        assert f == mock_parse.return_value


    def test_can_fetch_mmtf(self, fetch_fixture):
        mock_get, mock_parse = fetch_fixture
        mock_get.return_value.content = b"ABC"
        f = fetch("1ABC.mmtf", 1, b=2)
        mock_get.assert_called_with("https://mmtf.rcsb.org/v1.0/full/1abc", stream=True)
        mock_parse.assert_called_with(b"ABC", "1ABC.mmtf", 1, b=2)
        assert f == mock_parse.return_value


    def test_can_fetch_by_url(self, fetch_fixture):
        mock_get, mock_parse = fetch_fixture
        f = fetch("https://website.com/1ABC", 1, b=2)
        mock_get.assert_called_with("https://website.com/1ABC", stream=True)
        mock_parse.assert_called_with("ABC", "https://website.com/1ABC", 1, b=2)
        assert f == mock_parse.return_value


    def test_can_handle_no_results(self, fetch_fixture):
        mock_get, mock_parse = fetch_fixture
        mock_get.return_value.status_code = 400
        with pytest.raises(ValueError):
            fetch("1ABC", 1, b=2)


class TestFetchingOverSsh:

    @pytest.fixture
    def ssh_fixture(self):
        with patch("paramiko.SSHClient") as mock_ssh, \
             patch("paramiko.AutoAddPolicy") as mock_policy, \
             patch("biomod.io.utils.parse_string") as mock_parse:
            mock_client = Mock()
            mock_ssh.return_value = mock_client
            mock_client.exec_command.return_value = (Mock(), Mock(), Mock())
            mock_client.exec_command.return_value[1].read.return_value = b"STRING"
            mock_policy.return_value = "POLICY"
            yield mock_client, mock_parse


    def test_can_get_filestring_over_ssh_with_keys(self, ssh_fixture):
        mock_client, mock_parse = ssh_fixture
        f = fetch_over_ssh("HOST", "USER", "/path/", 1, a=2)
        mock_client.set_missing_host_key_policy.assert_called_with("POLICY")
        mock_client.load_system_host_keys.assert_called_with()
        mock_client.connect.assert_called_with(hostname="HOST", username="USER")
        mock_client.exec_command.assert_called_with("less /path/")
        mock_client.close.assert_called_with()
        mock_parse.assert_called_with("STRING", "/path/", 1, a=2)
        assert f is mock_parse.return_value


    def test_can_get_filestring_over_ssh_with_password(self, ssh_fixture):
        mock_client, mock_parse = ssh_fixture
        f = fetch_over_ssh("HOST", "USER", "/path/", 1, password="xxx", a=2)
        mock_client.set_missing_host_key_policy.assert_called_with("POLICY")
        assert not mock_client.load_system_host_keys.called
        mock_client.connect.assert_called_with(
         hostname="HOST", username="USER", password="xxx"
        )
        mock_client.exec_command.assert_called_with("less /path/")
        mock_client.close.assert_called_with()
        mock_parse.assert_called_with("STRING", "/path/", 1, a=2)
        assert f is mock_parse.return_value


    def test_connection_is_always_closed(self, ssh_fixture):
        mock_client, mock_parse = ssh_fixture
        mock_client.set_missing_host_key_policy.side_effect = Exception
        try:
            fetch_over_ssh("HOST", "USER", "/path/")
        except Exception:
            pass
        mock_client.close.assert_called_with()


class TestStringParsing:

    @patch("biomod.io.utils.get_parse_functions")
    def test_can_get_file_dict(self, mock_get):
        mock_get.return_value = [MagicMock(), MagicMock()]
        f = parse_string("ABCD", "file.xyz", file_dict=True)
        mock_get.assert_called_with("ABCD", "file.xyz")
        mock_get.return_value[0].assert_called_with("ABCD")
        assert f == mock_get.return_value[0].return_value


    @patch("biomod.io.utils.get_parse_functions")
    def test_can_get_data_dict(self, mock_get):
        mock_get.return_value = [MagicMock(), MagicMock()]
        f = parse_string("ABCD", "file.xyz", data_dict=True)
        mock_get.assert_called_with("ABCD", "file.xyz")
        mock_get.return_value[0].assert_called_with("ABCD")
        mock_get.return_value[1].assert_called_with(mock_get.return_value[0].return_value)
        assert f == mock_get.return_value[1].return_value


    @patch("biomod.io.utils.get_parse_functions")
    @patch("biomod.io.builder.data_dict_to_file")
    def test_can_get_file(self, mock_data, mock_get):
        mock_get.return_value = [MagicMock(), MagicMock()]
        mock_get.return_value[1].__name__ = "mmcif_Z"
        f = parse_string("ABCD", "file.cif")
        mock_get.assert_called_with("ABCD", "file.cif")
        mock_get.return_value[0].assert_called_with("ABCD")
        mock_get.return_value[1].assert_called_with(mock_get.return_value[0].return_value)
        mock_data.assert_called_with(mock_get.return_value[1].return_value, "cif")
        assert f == mock_data.return_value



class TestParseFunctionGetting:

    def test_can_get_cif_functions(self):
        f1, f2 = get_parse_functions("ABC", "x.cif")
        assert f1 is mmcif_string_to_mmcif_dict
        assert f2 is mmcif_dict_to_data_dict


    def test_can_get_mmtf_functions(self):
        f1, f2 = get_parse_functions("ABC", "x.mmtf")
        assert f1 is mmtf_bytes_to_mmtf_dict
        assert f2 is mmtf_dict_to_data_dict


    def test_can_get_pdb_functions(self):
        f1, f2 = get_parse_functions("ABC", "x.pdb")
        assert f1 is pdb_string_to_pdb_dict
        assert f2 is pdb_dict_to_data_dict


    def test_bytes_mean_mmtf(self):
        f1, f2 = get_parse_functions(b"ABC", "x.xxx")
        assert f1 is mmtf_bytes_to_mmtf_dict
        assert f2 is mmtf_dict_to_data_dict


    def test_can_identify_cif(self):
        f1, f2 = get_parse_functions("ABC_atom_sites", "x.xxx")
        assert f1 is mmcif_string_to_mmcif_dict
        assert f2 is mmcif_dict_to_data_dict


    def test_can_identify_pdb(self):
        f1, f2 = get_parse_functions("ABC", "x.xxx")
        assert f1 is pdb_string_to_pdb_dict
        assert f2 is pdb_dict_to_data_dict


class TestSaving:

    @patch("builtins.open")
    def test_saves_string_to_file(self, mock_open):
        open_return = MagicMock()
        mock_file = Mock()
        mock_write = MagicMock()
        mock_file.write = mock_write
        open_return.__enter__.return_value = mock_file
        mock_open.return_value = open_return
        save("filestring", "filename")
        mock_open.assert_called_once_with("filename", "w")
        mock_write.assert_called_once_with("filestring")


    @patch("builtins.open")
    def test_saves_bytestring_to_file(self, mock_open):
        open_return = MagicMock()
        mock_file = Mock()
        mock_write = MagicMock()
        mock_file.write = mock_write
        open_return.__enter__.return_value = mock_file
        mock_open.return_value = open_return
        mock_write.side_effect = [Exception, None]
        save(b"filestring", "filename")
        mock_open.assert_called_with("filename", "wb")
        mock_write.assert_called_with(b"filestring")
