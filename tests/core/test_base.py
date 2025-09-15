from unittest.mock import Mock, patch, MagicMock
from biomod.core.base import (
 get_object_from_filter, get_object_attribute_from_filter,
 attribute_matches_value, filter_objects, query, getone, StructureClass,
 StructureSet
)

class TestObjectFromFilter:

    def test_can_get_same_object(self):
        obj = Mock()
        obj.__lt__ = MagicMock()
        obj2 = get_object_from_filter(obj, ["height"])
        assert obj is obj2
        obj2 = get_object_from_filter(obj, ["height", "regex"])
        assert obj is obj2
        obj2 = get_object_from_filter(obj, ["height", "lt"])
        assert obj is obj2


    def test_can_get_chained_object(self):
        obj = Mock()
        obj2 = get_object_from_filter(obj, ["o1", "o2", "o3", "height", "regex"])
        assert obj2 is obj.o1.o2.o3
        obj2 = get_object_from_filter(obj, ["o1", "o2", "o3", "height"])
        assert obj2 is obj.o1.o2.o3



class TestAttributeGetting:

    def test_can_get_basic_attribute(self):
        obj = Mock(x=10)
        assert get_object_attribute_from_filter(obj, ["x"]) == 10
        assert get_object_attribute_from_filter(obj, ["y", "x"]) == 10
    

    def test_can_get_attribute_from_early_chain(self):
        obj = Mock(x=10)
        del obj.regex
        assert get_object_attribute_from_filter(obj, ["x", "regex"]) == 10
    

    def test_can_get_no_attribute(self):
        obj = Mock(x=10)
        del obj.y
        assert get_object_attribute_from_filter(obj, ["y"]) is None



class TestAttributeMatching:

    def test_exact_match(self):
        assert attribute_matches_value(10, 10, ["height", "xy"])
        assert not attribute_matches_value(10, 11, ["height", "xy"])
    

    def test_regex_match(self):
        assert attribute_matches_value("jon", "jon|joe", ["name", "regex"])
        assert not attribute_matches_value("jon", "jon|joe", ["name", "rogox"])


    def test_magic_method_match(self):
        assert attribute_matches_value(12, 10, ["height", "gt"])
        assert not attribute_matches_value(10, 10, ["height", "gt"])
        assert attribute_matches_value(10, 10, ["height", "gte"])



class TestObjectFiltering:

    @patch("biomod.core.base.get_object_from_filter")
    @patch("biomod.core.base.get_object_attribute_from_filter")
    @patch("biomod.core.base.attribute_matches_value")
    @patch("biomod.core.base.StructureSet")
    def test_can_filter_objects(self, mock_s, mock_match, mock_getat, mock_getob):
        structures=[
         Mock(x="A", y=1), Mock(x="B", y=3), Mock(x="B", y=3),
         Mock(x="C", y=2), Mock(x="D", y=4), Mock(x="D", y=4)
        ]
        objects = Mock(structures=structures)
        mock_getob.side_effect = lambda s, c: s
        mock_getat.side_effect = lambda s, c: c[0]
        mock_match.side_effect = [False, True, False, True, False, False]
        filter_objects(objects, "key__key2__key_3", "value")
        for structure in structures:
            mock_getob.assert_any_call(structure, ["key", "key2", "key_3"])
            mock_getat.assert_any_call(structure, ["key", "key2", "key_3"])
            mock_match.assert_any_call("key", "value", ["key", "key2", "key_3"])
        mock_s.assert_called_with(structures[1], structures[3])



class TestQueryDecorator:

    def setup_method(self, method):
        self.s = Mock(structures={2, 4, 6}, ids={1, 3, 5})
        self.f = lambda s: self.s


    def test_can_get_unfiltered_objects(self):
        f = query(self.f)
        assert f(self) == {2, 4, 6}


    @patch("biomod.core.base.filter_objects")
    def test_can_get_filtered_objects(self, mock_filter):
        mock_filter.side_effect = [Mock(structures={20}, ids={10})]
        f = query(self.f)
        assert f(self, a=1) == {20}
        mock_filter.assert_any_call(self.s, "a", 1)


    @patch("biomod.core.base.filter_objects")
    def test_can_get_filtered_objects_as_tuple(self, mock_filter):
        mock_filter.side_effect = [Mock(structures={2}, ids={1})]
        f = query(self.f, tuple_=True)
        assert f(self, a=1) == (2,)
        mock_filter.assert_any_call(self.s, "a", 1)


    def test_can_get_objects_by_id(self):
        f = query(self.f)
        assert f(self, 3) == {self.s.get.return_value}
        self.s.get.assert_called_with(3)
        assert f(self, 8) == set()



class TestGetOneDecorator:

    def test_can_get_one(self):
        def f(s):
            return [4, 6, 7]
        f = getone(f)
        assert f(self) == 4


    def test_can_get_mone(self):
        def f(s):
            return []
        f = getone(f)
        assert f(self) is None



class TestStructureClassMetaclass:

    @patch("biomod.core.base.query")
    @patch("biomod.core.base.getone")
    def test_structure_class_metaclass(self, mock_getone, mock_query):
        class TestClass(metaclass=StructureClass):
            def a(self): return 1000
            def chains(self): return {1: 2, 3: 4}
            def residues(self): return {10: 2, 30: 4}
            def ligands(self): return {11: 2, 31: 4}
            def waters(self): return {12: 2, 32: 4}
            def molecules(self): return {13: 2, 33: 4}
            def atoms(self): return {14: 2, 34: 4}
            def b(self): return 2000
        obj = TestClass()
        assert obj.chains is mock_query.return_value
        assert obj.chain is mock_getone.return_value
        assert obj.residues is mock_query.return_value
        assert obj.residue is mock_getone.return_value
        assert obj.ligands is mock_query.return_value
        assert obj.ligand is mock_getone.return_value
        assert obj.waters is mock_query.return_value
        assert obj.water is mock_getone.return_value
        assert obj.molecules is mock_query.return_value
        assert obj.molecule is mock_getone.return_value
        assert obj.atoms is mock_query.return_value
        assert obj.atom is mock_getone.return_value
        assert obj.a() == 1000
        assert obj.b() == 2000



class TestStructureSet:

    def test_can_make_structure_set(self):
        objects = [Mock(_id=n) for n in range(5)]
        s = StructureSet(*objects)
        assert s._d == {
         0: {objects[0]}, 1: {objects[1]}, 2: {objects[2]},
         3: {objects[3]}, 4: {objects[4]}
        }
        objects[2]._id = 0
        s = StructureSet(*objects)
        assert s._d == {
         0: {objects[0], objects[2]}, 1: {objects[1]},
         3: {objects[3]}, 4: {objects[4]}
        }
    

    def test_can_add_two_structure_sets(self):
        objects = [Mock(_id=n) for n in range(5)]
        objects[2]._id = 0
        s1 = StructureSet(*objects[:3])
        s2 = StructureSet(*objects[3:])
        assert s1._d == {
         0: {objects[0], objects[2]}, 1: {objects[1]},
        }
        assert s2._d == {3: {objects[3]}, 4: {objects[4]}}
        s3 = s1 + s2
        assert s3._d == {
         0: {objects[0], objects[2]}, 1: {objects[1]},
         3: {objects[3]}, 4: {objects[4]}
        }
    

    def test_can_get_length_of_structure_sets(self):
        objects = [Mock(_id=n) for n in range(5)]
        s = StructureSet(*objects)
        assert len(s) == 5
        objects[2]._id = 0
        s = StructureSet(*objects)
        assert len(s) == 5
    

    def test_can_get_structure_set_ids(self):
        objects = [Mock(_id=n) for n in range(5)]
        s = StructureSet(*objects)
        assert s.ids == {0, 1, 2, 3, 4}
    

    def test_can_get_structure_set_structures(self):
        objects = [Mock(_id=n) for n in range(5)]
        s = StructureSet(*objects)
        assert s.structures == objects
        objects[2]._id = 0
        s = StructureSet(*objects)
        assert set(s.structures) == set(objects)
    

    def test_can_get_structures_by_id(self):
        objects = [Mock(_id=n) for n in range(5)]
        s = StructureSet(*objects)
        assert s.get(0) == objects[0]
        assert s.get(4) == objects[4]
        assert s.get(5) is None
        objects[2]._id = 0
        s = StructureSet(*objects)
        assert s.get(0) in (objects[0], objects[2])
        assert s.get(4) == objects[4]
        assert s.get(2) is None