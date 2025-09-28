import inspect
import re

import numpy as np
import pytest

import _tskit
import tests.tsutil as tsutil
import tskit


pytestmark = pytest.mark.skipif(
    not getattr(_tskit, "HAS_NUMPY_2", False),
    reason="ImmutableTableCollection requires NumPy 2 runtime",
)


def get_mutable_and_immutable(ts):
    mutable = ts.dump_tables()
    immutable = ts.tables
    assert isinstance(immutable, tskit.tables.ImmutableTableCollection)
    return mutable, immutable


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
class TestCollectionParity:
    def test_basic_properties_match(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        assert mutable.sequence_length == immutable.sequence_length
        assert mutable.time_units == immutable.time_units
        assert mutable.file_uuid == immutable.file_uuid
        assert mutable.metadata_schema == immutable.metadata_schema
        assert mutable.metadata == immutable.metadata
        assert mutable.metadata_schema.encode_row(mutable.metadata) == bytes(
            immutable.metadata_bytes
        )

    def test_asdict_equals(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)

        d_mut = mutable.asdict()
        d_imm = immutable.asdict()
        assert set(d_mut.keys()) == set(d_imm.keys())
        for key, val in d_mut.items():
            if isinstance(val, dict):
                for col, arr in val.items():
                    arr2 = d_imm[key][col]
                    assert np.array_equal(arr, arr2) or (
                        np.all(map(tskit.is_unknown_time, arr))
                        and np.all(map(tskit.is_unknown_time, arr2))
                    )
            else:
                assert d_imm[key] == val

    def test_equals_bidirectional(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        assert mutable.equals(mutable)
        assert mutable.equals(immutable)
        assert immutable.equals(mutable)
        assert immutable.equals(immutable)
        # assert_equals should not raise
        mutable.assert_equals(mutable)
        mutable.assert_equals(immutable)
        immutable.assert_equals(mutable)
        immutable.assert_equals(immutable)

    def test_equals_ignore_flags(self, ts):
        # Create two mutable copies and an immutable baseline
        m1, imm = get_mutable_and_immutable(ts)
        m2 = m1.copy()
        # Diverge TS-level metadata
        m1.metadata_schema = tskit.MetadataSchema({"codec": "json", "type": "object"})
        m1.metadata = {"x": 1}
        assert not imm.equals(m1)
        assert imm.equals(m1, ignore_ts_metadata=True)
        # Diverge provenance
        m1.provenances.add_row(record="random stuff")
        assert not imm.equals(m1)
        assert imm.equals(m1, ignore_ts_metadata=True, ignore_provenance=True)
        # Reset to identical and verify equals again
        m1 = m2
        assert imm.equals(m1)

    def test_nbytes_parity(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        assert mutable.nbytes == immutable.nbytes

    def test_reference_sequence_and_index_flags(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)

        assert bool(mutable.has_reference_sequence()) == bool(
            immutable.has_reference_sequence()
        )
        if mutable.has_reference_sequence():
            mutable.reference_sequence.assert_equals(immutable.reference_sequence)

        assert mutable.has_index() == immutable.has_index()
        if mutable.has_index():
            assert np.array_equal(
                mutable.indexes.edge_insertion_order,
                immutable.indexes.edge_insertion_order,
            )
            assert np.array_equal(
                mutable.indexes.edge_removal_order, immutable.indexes.edge_removal_order
            )

    def test_copy_dump_tree_sequence_roundtrip(self, tmp_path, ts):
        mutable, immutable = get_mutable_and_immutable(ts)

        # copy() returns a mutable TableCollection equal to both
        copy_tc = immutable.copy()
        assert isinstance(copy_tc, tskit.TableCollection)
        copy_tc.assert_equals(mutable)
        copy_tc.assert_equals(immutable)

        # dump() uses the mutable copy under the hood
        out = tmp_path / "tables"
        immutable.dump(out)
        loaded = tskit.load(out)
        ts.tables.assert_equals(loaded.tables)

        # tree_sequence() identical to original
        ts2 = immutable.tree_sequence()
        ts.tables.assert_equals(ts2.tables)

    def test_str_contains_identifier(self, ts):
        _, immutable = get_mutable_and_immutable(ts)
        s = str(immutable)
        assert "ImmutableTableCollection" in s


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
class TestTablesParity:
    def test_table_name_map_and_lengths(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        assert set(mutable.table_name_map.keys()) == set(
            immutable.table_name_map.keys()
        )

        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]
            assert len(mt) == len(it)

    def test_columns_and_rows_equal(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]

            for col_name in mt.column_names:
                a = getattr(mt, col_name)
                b = getattr(it, col_name)
                assert np.array_equal(a, b) or (
                    np.all(map(tskit.is_unknown_time, a))
                    and np.all(map(tskit.is_unknown_time, b))
                )

            # Row object equality
            if len(mt) > 0:
                for idx in [0, len(mt) - 1]:
                    assert mt[idx] == it[idx]

        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]
            for col_name in mt.column_names:
                a = getattr(mt, col_name)
                b = getattr(it, col_name)
                assert np.array_equal(a, b) or (
                    np.all(map(tskit.is_unknown_time, a))
                    and np.all(map(tskit.is_unknown_time, b))
                )

    def test_slicing_and_boolean_index(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]
            # Slice view
            sl = slice(0, max(0, len(mt)))
            it_view = it[sl]
            mt_view = mt[sl]
            it_view.assert_equals(mt_view)

            mask = np.zeros(len(it), dtype=bool)
            if len(it) > 0:
                mask[0] = True
            it_view2 = it[mask]
            mt_view2 = mt[mask]
            it_view2.assert_equals(mt_view2)

    def test_mask_then_slice(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]

            if len(it) < 3:
                continue

            mask = np.zeros(len(it), dtype=bool)
            mask[0] = True
            mask[-1] = True
            mask[len(it) // 2] = True

            it_mask = it[mask]
            mt_mask = mt[mask]

            if len(it_mask) <= 1:
                continue

            it_slice = it_mask[1:]
            mt_slice = mt_mask[1:]
            it_slice.assert_equals(mt_slice)

    def test_slice_then_mask(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]

            if len(it) < 4:
                continue

            it_slice = it[1:-1]
            mt_slice = mt[1:-1]

            if len(it_slice) == 0:
                continue

            mask = np.zeros(len(it_slice), dtype=bool)
            mask[0] = True
            mask[-1] = True

            it_mask = it_slice[mask]
            mt_mask = mt_slice[mask]
            it_mask.assert_equals(mt_mask)

    def test_slice_view_indexing(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]

            if len(it) == 0:
                continue

            if len(it) >= 10:
                mt_view = mt[-10:]
                it_view = it[-10:]
                for i in [0, 5, -1]:
                    mt_row = mt_view[i]
                    it_row = it_view[i]
                    assert mt_row == it_row
                    if i == 0:
                        assert it_row == it[-10]
                        assert mt_row == mt[-10]
                    elif i == 5:
                        assert it_row == it[-5]
                        assert mt_row == mt[-5]
                    elif i == -1:
                        assert it_row == it[-1]
                        assert mt_row == mt[-1]

            if len(it) >= 20:
                mt_view = mt[5:15]
                it_view = it[5:15]
                for i in [0, 4, -1]:
                    mt_row = mt_view[i]
                    it_row = it_view[i]
                    assert mt_row == it_row
                    if i == 0:
                        assert it_row == it[5]
                        assert mt_row == mt[5]
                    elif i == 4:
                        assert it_row == it[9]
                        assert mt_row == mt[9]
                    elif i == -1:
                        assert it_row == it[14]
                        assert mt_row == mt[14]

    def test_slice_view_iteration(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]

            if len(it) < 10:
                continue

            mt_view = mt[-10:]
            it_view = it[-10:]

            mt_rows = list(mt_view)
            it_rows = list(it_view)

            assert len(mt_rows) == len(it_rows)
            for mt_row, it_row in zip(mt_rows, it_rows):
                assert mt_row == it_row

    def test_slice_view_ragged_column_access(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)

        if ts.num_mutations >= 100:
            mt = mutable.mutations
            it = immutable.mutations

            mt_slice = mt[-100:]
            it_slice = it[-100:]

            for idx, (mt_mut, it_mut) in enumerate(zip(mt_slice, it_slice)):
                assert mt_mut.derived_state == it_mut.derived_state
                assert mt_mut.metadata == it_mut.metadata
                original_idx = len(mt) - 100 + idx
                assert it_mut.derived_state == it[original_idx].derived_state
                assert mt_mut.derived_state == mt[original_idx].derived_state

        if ts.num_sites >= 50:
            mt = mutable.sites
            it = immutable.sites

            mt_slice = mt[-50:]
            it_slice = it[-50:]

            for idx, (mt_site, it_site) in enumerate(zip(mt_slice, it_slice)):
                assert mt_site.ancestral_state == it_site.ancestral_state
                assert mt_site.metadata == it_site.metadata
                original_idx = len(mt) - 50 + idx
                assert it_site.ancestral_state == it[original_idx].ancestral_state
                assert mt_site.ancestral_state == mt[original_idx].ancestral_state

    def test_nested_slicing(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]

            if len(it) < 50:
                continue

            # Create nested slices
            mt_slice1 = mt[-50:]
            it_slice1 = it[-50:]

            mt_slice2 = mt_slice1[10:30]
            it_slice2 = it_slice1[10:30]

            mt_row = mt_slice2[5]
            it_row = it_slice2[5]
            assert mt_row == it_row

            for mt_r, it_r in zip(mt_slice2, it_slice2):
                assert mt_r == it_r

            for col_name in mt.column_names:
                a = getattr(mt_slice2, col_name)
                b = getattr(it_slice2, col_name)
                assert np.array_equal(a, b) or (
                    np.all(map(tskit.is_unknown_time, a))
                    and np.all(map(tskit.is_unknown_time, b))
                )

    def test_random_access_on_slice(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]

            if len(it) < 100:
                continue

            mt_slice = mt[-100:]
            it_slice = it[-100:]

            indices = [0, 10, 50, 75, 99, -1, -10]
            for idx in indices:
                mt_row = mt_slice[idx]
                it_row = it_slice[idx]
                assert mt_row == it_row

            for col_name in mt.column_names:
                a = getattr(mt_slice, col_name)
                b = getattr(it_slice, col_name)
                assert np.array_equal(a, b) or (
                    np.all(map(tskit.is_unknown_time, a))
                    and np.all(map(tskit.is_unknown_time, b))
                )

    def test_table_equals_bidirectional(self, ts):
        mutable, immutable = get_mutable_and_immutable(ts)
        for name in mutable.table_name_map.keys():
            mt = mutable.table_name_map[name]
            it = immutable.table_name_map[name]
            assert mt.equals(it)
            assert it.equals(mt)
            mt.assert_equals(it)
            it.assert_equals(mt)
            if isinstance(mt, tskit.tables.MutableMetadataTable) or isinstance(
                it, tskit.tables.ImmutableMetadataTable
            ):
                mt.assert_equals(it, ignore_metadata=True)
                it.assert_equals(mt, ignore_metadata=True)


@pytest.mark.parametrize("ts", tsutil.get_example_tree_sequences())
class TestImmutableErrors:
    def test_collection_mutators_raise(self, ts):
        _, immutable = get_mutable_and_immutable(ts)
        for name in type(immutable)._MUTATOR_METHODS:
            with pytest.raises(tskit.ImmutableTableError):
                getattr(immutable, name)

    def test_collection_property_setters_raise(self, ts):
        _, immutable = get_mutable_and_immutable(ts)
        with pytest.raises(tskit.ImmutableTableError):
            immutable.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        with pytest.raises(tskit.ImmutableTableError):
            immutable.metadata = {}
        with pytest.raises(tskit.ImmutableTableError):
            immutable.metadata_bytes = b""

    def test_table_mutators_raise(self, ts):
        _, immutable = get_mutable_and_immutable(ts)
        for it in immutable.table_name_map.values():
            for name in tskit.tables.ImmutableBaseTable._MUTATION_METHODS:
                with pytest.raises(tskit.ImmutableTableError):
                    getattr(it, name)

    def test_table_metadata_schema_setter_raises(self, ts):
        _, immutable = get_mutable_and_immutable(ts)
        for _, itab in immutable.table_name_map.items():
            if isinstance(itab, tskit.tables.ImmutableMetadataTable):
                with pytest.raises(tskit.ImmutableTableError):
                    itab.metadata_schema = tskit.MetadataSchema(None)

    def test_table_attribute_assignment_raises(self, ts):
        _, immutable = get_mutable_and_immutable(ts)
        for _, itab in immutable.table_name_map.items():
            # Try setting a known column if any exist
            col_names = [
                c
                for c in getattr(itab, "column_names", [])
                if not c.endswith("_schema")
            ]
            if col_names:
                col = col_names[0]
                current = getattr(itab, col)
                with pytest.raises(tskit.ImmutableTableError):
                    setattr(itab, col, current)
            # Setting an unknown public attribute should also raise
            with pytest.raises(tskit.ImmutableTableError):
                itab.not_a_real_column = 123

    def test_collection_attribute_assignment_raises(self, ts):
        _, immutable = get_mutable_and_immutable(ts)
        # Try to set table attributes
        with pytest.raises(tskit.ImmutableTableError):
            immutable.nodes = immutable.nodes
        with pytest.raises(tskit.ImmutableTableError):
            immutable.edges = immutable.edges
        with pytest.raises(tskit.ImmutableTableError):
            immutable.sites = immutable.sites
        # Try to set arbitrary public attributes
        with pytest.raises(tskit.ImmutableTableError):
            immutable.new_attribute = 123
        with pytest.raises(tskit.ImmutableTableError):
            immutable.sequence_length = 100


class TestMethodParity:
    def test_immutable_has_method_or_mutator(self, ts_fixture):
        tc = ts_fixture.dump_tables()
        it = ts_fixture.tables
        # Collect instance-bound public methods of TableCollection
        tc_methods = []
        for name in dir(tc):
            if name.startswith("_"):
                continue
            attr = getattr(tc, name)
            if inspect.ismethod(attr) and getattr(attr, "__self__", None) is tc:
                tc_methods.append(name)

        missing = []
        for name in tc_methods:
            try:
                inspect.getattr_static(it, name)
                present = True
            except AttributeError:
                present = False
            if present:
                continue
            if name in type(it)._MUTATOR_METHODS:
                continue
            missing.append(name)

        assert (
            missing == []
        ), f"ImmutableTableCollection missing non-mutator methods: {missing}"

    def test_immutable_tables_have_method_or_mutator(self, ts_fixture):
        tc = ts_fixture.dump_tables()
        itc = ts_fixture.tables

        for table_name, mt in tc.table_name_map.items():
            it = itc.table_name_map[table_name]
            # Collect instance-bound public methods on the mutable table
            mt_methods = []
            for name in dir(mt):
                if name.startswith("_"):
                    continue
                attr = getattr(mt, name)
                if inspect.ismethod(attr) and getattr(attr, "__self__", None) is mt:
                    mt_methods.append(name)

            missing = []
            for name in mt_methods:
                # Use getattr_static to avoid triggering __getattr__ on immutable tables
                try:
                    inspect.getattr_static(it, name)
                    present = True
                except AttributeError:
                    present = False
                if present:
                    continue
                if name in tskit.tables.ImmutableBaseTable._MUTATION_METHODS:
                    continue
                missing.append(name)

            assert (
                missing == []
            ), f"Immutable {table_name} table missing non-mutator methods: {missing}"


class TestImmutableTimestampHandling:
    def test_assert_equals_ignore_timestamps_roundtrip(self):
        tables = tskit.TableCollection(sequence_length=1.0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.provenances.add_row(record="{}", timestamp="2024-01-01T00:00:00Z")

        ts = tables.tree_sequence()
        immutable_prov = ts.tables.provenances

        mutable_tables = ts.dump_tables()
        mutable_tables.provenances.clear()
        mutable_tables.provenances.add_row(
            record="{}", timestamp="2024-02-01T00:00:00Z"
        )
        mutable_prov = mutable_tables.provenances

        with pytest.raises(AssertionError, match="timestamp"):
            immutable_prov.assert_equals(mutable_prov)
        immutable_prov.assert_equals(mutable_prov, ignore_timestamps=True)
        mutable_prov.assert_equals(immutable_prov, ignore_timestamps=True)

    def test_assert_equals_ignore_timestamps_guard(self, ts_fixture, monkeypatch):
        immutable_prov = ts_fixture.tables.provenances
        monkeypatch.setattr(immutable_prov.__class__, "table_name", "not_provenances")
        with pytest.raises(ValueError, match="only valid for Provenance tables"):
            immutable_prov.assert_equals(immutable_prov, ignore_timestamps=True)

    def test_assert_equals_ignore_timestamps_other_difference(self):
        tables = tskit.TableCollection(sequence_length=1.0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.provenances.add_row(record="{}", timestamp="2024-01-01T00:00:00Z")

        ts = tables.tree_sequence()
        immutable = ts.tables
        mutable = ts.dump_tables()
        mutable.provenances.clear()
        mutable.provenances.add_row(
            record="different", timestamp="2024-02-01T00:00:00Z"
        )

        immutable_msg = re.escape(
            "ImmutableProvenanceTable row 0 differs:\n"
            "self.record={} other.record=different"
        )

        with pytest.raises(AssertionError, match=immutable_msg):
            immutable.assert_equals(mutable, ignore_timestamps=True)
        with pytest.raises(AssertionError, match=immutable_msg):
            mutable.assert_equals(immutable, ignore_timestamps=True)


class TestImmutableIndexViews:
    def test_index_view_ragged_columns(self, ts_fixture):
        immutable_tables = ts_fixture.tables
        mutations = immutable_tables.mutations
        if mutations.num_rows < 3:
            pytest.skip("Need mutations with metadata for this test")

        indices = np.array([0, mutations.num_rows - 1, 1], dtype=np.int64)
        view = mutations[indices]

        expected_rows = [mutations[i] for i in indices]
        assert list(view) == expected_rows

        base_ds = mutations.derived_state
        base_ds_offset = mutations.derived_state_offset
        expected_ds = []
        expected_ds_lengths = []
        for idx in indices:
            start = base_ds_offset[idx]
            end = base_ds_offset[idx + 1]
            expected_ds.extend(base_ds[start:end])
            expected_ds_lengths.append(end - start)
        assert np.array_equal(view.derived_state, np.array(expected_ds, dtype=np.int8))
        derived_offsets = view.derived_state_offset
        assert list(derived_offsets[1:] - derived_offsets[:-1]) == expected_ds_lengths

        base_md = mutations.metadata
        base_md_offset = mutations.metadata_offset
        expected_md = []
        expected_md_lengths = []
        for idx in indices:
            start = base_md_offset[idx]
            end = base_md_offset[idx + 1]
            expected_md.extend(base_md[start:end])
            expected_md_lengths.append(end - start)
        assert np.array_equal(view.metadata, np.array(expected_md, dtype=np.int8))
        metadata_offsets = view.metadata_offset
        assert list(metadata_offsets[1:] - metadata_offsets[:-1]) == expected_md_lengths

    def test_index_view_offset_columns(self):
        tables = tskit.TableCollection(sequence_length=1.0)
        n0 = tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        n1 = tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        site = tables.sites.add_row(position=0.1, ancestral_state="A")
        tables.mutations.add_row(site=site, node=n0, derived_state="AA")
        tables.mutations.add_row(site=site, node=n1, derived_state="BBB")

        ts = tables.tree_sequence()
        mutations = ts.tables.mutations
        indices = np.array([1, 0], dtype=np.int64)
        view = mutations[indices]

        expected_offsets = np.array([0, 3, 5], dtype=np.uint32)
        assert np.array_equal(view.derived_state_offset, expected_offsets)
        assert view[0] == mutations[1]

    def test_index_view_empty_selection(self):
        tables = tskit.TableCollection(sequence_length=1.0)
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        site = tables.sites.add_row(position=0.1, ancestral_state="A")
        tables.mutations.add_row(site=site, node=0, derived_state="A")
        ts = tables.tree_sequence()

        mutations = ts.tables.mutations
        indices = np.array([], dtype=np.int64)
        view = mutations[indices]

        assert view.num_rows == 0
        assert np.array_equal(view.derived_state_offset, np.array([0], dtype=np.uint32))
        assert np.array_equal(view.metadata_offset, np.array([0], dtype=np.uint32))

    def test_index_out_of_bounds(self, ts_fixture):
        nodes = ts_fixture.tables.nodes
        with pytest.raises(IndexError, match="Index out of bounds"):
            nodes[nodes.num_rows]
        with pytest.raises(IndexError, match="Index out of bounds"):
            nodes[-nodes.num_rows - 1]

    def test_boolean_index_length_mismatch(self, ts_fixture):
        nodes = ts_fixture.tables.nodes
        mask = np.zeros(nodes.num_rows + 1, dtype=bool)
        with pytest.raises(
            IndexError, match="Boolean index must be same length as table"
        ):
            nodes[mask]


def test_immutable_table_metadata_schema_difference():
    tables = tskit.TableCollection(sequence_length=1.0)
    tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)

    tables_with_schema = tables.copy()
    tables_with_schema.nodes.metadata_schema = tskit.MetadataSchema(
        {"codec": "json", "type": "object"}
    )

    ts_plain = tables.tree_sequence()
    ts_schema = tables_with_schema.tree_sequence()

    plain_nodes = ts_plain.tables.nodes
    schema_nodes = ts_schema.tables.nodes

    assert not plain_nodes.equals(schema_nodes)
    assert not schema_nodes.equals(plain_nodes)
    with pytest.raises(AssertionError, match="metadata schemas differ"):
        plain_nodes.assert_equals(schema_nodes)
