"""Tiny CPU identities for cell-prefix subtraction with fixed projection."""
import pytest
import torch

from scripts.spatial_complement_cpu import CellIndex, centered_covariance, fixed_projection, stats_error, analytical_cell_h_costs


def test_full_rank_cell_cost_and_grouped_reuse_arithmetic():
    costs = analytical_cell_h_costs()['ranks']
    full, sketch = costs['128'], costs['8']
    assert full['cell_h_bytes_24_heads'] == 402653184
    assert full['prefix_h_bytes_24_heads'] == 454557696
    assert full['cell_h_plus_prefix_bytes_24_heads'] == 857210880
    assert full['uncached_per_query_logical_h_reads']['four_corners_bytes_all_queries_24_heads'] == 412316860416
    assert full['grouped_cell_support_reuse']['four_corner_reads_bytes_24_heads']*256 == 412316860416
    assert full['grouped_cell_support_reuse']['materialized_complement_h_bytes_24_heads'] == 402653184
    assert full['one_head_build_flops_selected_operations']['key_projection'] == 0
    for field in ('cell_h_bytes_24_heads', 'prefix_h_bytes_24_heads', 'h_times_projected_query_flops_all_queries_24_heads'):
        assert full[field] == 16*sketch[field]


def test_additive_sketch_has_no_directional_fidelity_guarantee():
    projection = fixed_projection(16, 8, 19)
    unit = torch.eye(16, dtype=torch.float64)[0]
    orthogonal = unit-projection@(projection.T@unit)
    orthogonal /= orthogonal.norm()
    keys = torch.stack((orthogonal, -orthogonal))
    values = torch.stack((unit, -unit))
    full_h = values.T@keys
    exact = full_h@orthogonal
    projected = (full_h@projection)@(projection.T@orthogonal)
    assert exact.norm().item() == pytest.approx(2.)
    assert projected.norm().item() < 1e-25


@pytest.fixture
def index():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(123)
    keys = torch.randn(64, 8, generator=generator, dtype=torch.float64)
    values = torch.randn(64, 6, generator=generator, dtype=torch.float64)
    return CellIndex(keys, values, fixed_projection(8, 3, 124), (8, 8), 2)


@pytest.mark.parametrize('rectangles,registers', [([], []), ([[0, 0, 8, 8]], []),
    ([[0, 0, 4, 4], [2, 2, 6, 6]], [0, 0, 63, 63]),
    ([[6, 6, 8, 8]], [0, 63, 63]), ([], [0, 0, 1, 63, 63]),
    ([[0, 0, 2, 8], [6, 0, 8, 8]], [])])
def test_cell_prefix_matches_direct_complement(index, rectangles, registers):
    actual, accounting = index.complement(rectangles, registers)
    direct, _ = index.direct_complement(rectangles, registers)
    assert actual.count == direct.count == accounting['complement_count']
    assert actual.count+accounting['retained_unique_count'] == 64
    for name in ('sum_k', 'sum_v', 'sum_vk_u'):
        torch.testing.assert_close(getattr(actual, name), getattr(direct, name), rtol=1e-12, atol=1e-12)


def test_overlap_and_register_dedup_counts(index):
    _, accounting = index.complement([[0, 0, 4, 4], [2, 2, 6, 6]], [0, 0, 63, 63])
    assert accounting == dict(rectangle_union_count=28, register_input_count=4,
        unique_register_count=2, outside_register_count=1, retained_unique_count=29, complement_count=35)


@pytest.mark.parametrize('rectangles', [[[0, 0, 3, 4]], [[1, 0, 4, 4]], [[0, 0, 10, 8]],
                                      [[0, 0, 2, 2]]*3])
def test_partial_outside_and_excess_rectangles_rejected(index, rectangles):
    with pytest.raises(ValueError):
        index.complement(rectangles)


@pytest.mark.parametrize('registers', [[-1], [64], [1.5], [True]])
def test_invalid_register_ids_rejected(index, registers):
    with pytest.raises(ValueError):
        index.complement([], registers)


def test_fixed_projection_commutes_with_raw_subtraction_and_centering(index):
    actual, _ = index.complement([[0, 0, 4, 4]], [63])
    _, mask = index.direct_complement([[0, 0, 4, 4]], [63])
    k, v = index.keys[mask], index.values[mask]
    full_cross = v.T @ k
    torch.testing.assert_close(actual.sum_vk_u, full_cross @ index.projection, rtol=1e-12, atol=1e-12)
    full_covariance = (v-v.mean(0)).T @ (k-k.mean(0))/len(k)
    torch.testing.assert_close(centered_covariance(actual, index.projection),
                               full_covariance @ index.projection, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(index.projection.T@index.projection, torch.eye(3, dtype=torch.float64),
                               rtol=1e-12, atol=1e-12)


def test_empty_complement_bypasses_covariance(index):
    stats, _ = index.complement([[0, 0, 8, 8]])
    assert stats.count == 0
    assert centered_covariance(stats, index.projection) is None


def test_near_full_complement_raw_errors_are_not_clipped(index):
    # A fixed one-token complement, using two aligned rectangles plus registers.
    rectangles = [[0, 0, 6, 8], [6, 0, 8, 6]]
    registers = [54, 55, 62]  # Leaves token63.
    index32 = CellIndex(index.keys.float(), index.values.float(), index.projection.float(), (8, 8), 2)
    actual, _ = index32.complement(rectangles, registers)
    direct, mask = index32.direct_complement(rectangles, registers)
    assert actual.count == direct.count == 1 and mask.nonzero().flatten().tolist() == [63]
    errors = stats_error(actual, direct)
    assert all(torch.isfinite(torch.tensor(errors[name]['l2'])) for name in ('sum_k', 'sum_v', 'sum_vk_u'))
    # The implementation exposes the raw formula rather than zeroing the result
    # just because a singleton has mathematically zero centered covariance.
    expected = actual.sum_vk_u-torch.outer(actual.sum_v, actual.sum_k@index32.projection)
    assert torch.equal(centered_covariance(actual, index32.projection), expected)


def test_storage_is_actual_cell_tensor_accounting(index):
    storage = index.storage()
    components = storage['resident_tensor_bytes_by_component']
    assert components['prefix_sum_vk_u'] == 5*5*6*3*8
    assert components['cell_sum_vk_u'] == 4*4*6*3*8
    assert storage['resident_tensor_bytes_total'] == sum(components.values())
    assert storage['centered_covariance_cache_bytes'] == 0
    assert storage['raw_cross_moment_cache_bytes_included_above'] == components['prefix_sum_vk_u']+components['cell_sum_vk_u']


@pytest.fixture
def h_index(index):
    return CellIndex(index.keys, index.values, index.projection, index.grid, index.cell_size, with_cell_h=True)


def test_cell_h_is_projected_within_cell_centered_moment(h_index):
    for row in range(4):
        for col in range(4):
            ids = [y*8+x for y in range(2*row, 2*row+2) for x in range(2*col, 2*col+2)]
            k, v = h_index.keys[ids], h_index.values[ids]
            full_h = (v-v.mean(0)).T @ (k-k.mean(0))
            torch.testing.assert_close(h_index.cell_h[row, col], full_h@h_index.projection, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('rectangles', [[], [[0, 0, 8, 8]], [[0, 0, 4, 4], [2, 2, 6, 6]],
                                       [[6, 6, 8, 8]], [[0, 0, 2, 8], [6, 0, 8, 8]]])
def test_complementary_cell_h_prefix_is_exact(h_index, rectangles):
    selected = h_index.complementary_cell_h(rectangles)
    mask = torch.ones(4, 4, dtype=torch.bool)
    for top, left, bottom, right in rectangles:
        mask[top//2:bottom//2, left//2:right//2] = False
    assert selected['cell_count'] == int(mask.sum())
    torch.testing.assert_close(selected['sum_h'], h_index.cell_h[mask].sum(0), rtol=1e-12, atol=1e-12)
    if mask.any():
        torch.testing.assert_close(selected['mean_h'], h_index.cell_h[mask].mean(0), rtol=1e-12, atol=1e-12)
    else:
        assert selected['mean_h'] is None


def test_cell_h_rejects_registers_partial_cells_and_missing_cache(index, h_index):
    with pytest.raises(ValueError, match='not built'):
        index.complementary_cell_h([])
    with pytest.raises(ValueError, match='register'):
        h_index.complementary_cell_h([], [63])
    with pytest.raises(ValueError, match='Partial cells'):
        h_index.complementary_cell_h([[0, 0, 3, 4]])


def test_whole_cloud_covariance_has_between_cell_term(h_index):
    whole, _ = h_index.complement([])
    cell_means_v = h_index.cell_sum_v.reshape(-1, 6)/4
    cell_means_z = (h_index.cell_sum_k@h_index.projection).reshape(-1, 3)/4
    between = (cell_means_v-cell_means_v.mean(0)).T@(cell_means_z-cell_means_z.mean(0))/16
    within = h_index.complementary_cell_h([])['mean_h']/4
    torch.testing.assert_close(centered_covariance(whole, h_index.projection), within+between, rtol=1e-12, atol=1e-12)
    assert between.norm() > 0


def test_cell_h_storage_is_counted_separately(h_index):
    storage = h_index.storage()
    assert storage['within_cell_h_cache_bytes_included_above'] == (4*4+5*5)*6*3*8
    assert storage['centered_covariance_cache_bytes'] == storage['within_cell_h_cache_bytes_included_above']
