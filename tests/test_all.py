import unittest

from damask_local import main as dsk


class TestFull(unittest.TestCase):

    def test_get_lattice_structure(self):
        self.assertEqual(dsk._get_lattice_structure(chemical_symbol="Al"), "cF")
        self.assertEqual(dsk._get_lattice_structure(chemical_symbol="Aluminum"), "cF")


if __name__ == '__main__':
    unittest.main()
