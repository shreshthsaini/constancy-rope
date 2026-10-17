"""Cell-aligned raw complement moments with a fixed shared linear projection.

One-head synthetic CPU accounting/cost prototype. K is interpreted as already
post-RoPE; V is the corresponding value tensor. No attention approximation,
adaptive partition, pretrained integration or GPU speed claim is implemented.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import resource
import time

import torch


@dataclass
class Stats:
    count: int
    sum_k: torch.Tensor
    sum_v: torch.Tensor
    sum_vk_u: torch.Tensor


def add(a, b, sign=1):
    return Stats(a.count+sign*b.count, a.sum_k+sign*b.sum_k,
                 a.sum_v+sign*b.sum_v, a.sum_vk_u+sign*b.sum_vk_u)


def raw_stats(k, v, z):
    return Stats(len(k), k.sum(0), v.sum(0), v.T @ z)


def fixed_projection(dimension, rank, seed, dtype=torch.float64):
    if not 0 < rank <= dimension:
        raise ValueError('Projection rank must lie in [1, dimension]')
    random = torch.randn(dimension, rank, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    q, r = torch.linalg.qr(random, mode='reduced')
    # Fix column signs so the generated matrix has a deterministic convention.
    signs = torch.where(r.diag() < 0, -torch.ones(rank, dtype=torch.float64), torch.ones(rank, dtype=torch.float64))
    return (q*signs).to(dtype)


def cell_prefix(values):
    shape = (values.shape[0]+1, values.shape[1]+1, *values.shape[2:])
    result = torch.zeros(shape, dtype=values.dtype)
    result[1:, 1:] = values.cumsum(0).cumsum(1)
    return result


class CellIndex:
    def __init__(self, keys, values, projection, grid, cell_size=16):
        height, width = grid
        if (min(height, width, cell_size) <= 0 or height % cell_size or width % cell_size
                or keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != height*width
                or values.shape[0] != height*width or projection.ndim != 2
                or projection.shape[0] != keys.shape[1]):
            raise ValueError('Invalid grid, cell size or tensor shape')
        if not all(t.device.type == 'cpu' and t.dtype == keys.dtype and bool(torch.isfinite(t).all())
                   for t in (keys, values, projection)) or keys.dtype not in (torch.float32, torch.float64):
            raise ValueError('Finite CPU FP32/FP64 tensors with matching dtypes required')
        self.keys, self.values, self.projection = keys, values, projection
        self.grid, self.cell_size = tuple(grid), cell_size
        self.sketch = keys @ projection
        rows, cols = height//cell_size, width//cell_size
        self.cell_count = torch.full((rows, cols), cell_size**2, dtype=torch.int64)
        self.cell_sum_k = keys.reshape(rows, cell_size, cols, cell_size, -1).sum((1, 3))
        self.cell_sum_v = values.reshape(rows, cell_size, cols, cell_size, -1).sum((1, 3))
        self.cell_sum_vk_u = torch.empty(rows, cols, values.shape[1], projection.shape[1], dtype=keys.dtype)
        spatial_v = values.reshape(height, width, -1)
        spatial_z = self.sketch.reshape(height, width, -1)
        # At most one cell is packed for matmul. No N x Dv x rank outer-product tensor.
        for row in range(rows):
            for col in range(cols):
                tile_v = spatial_v[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size].reshape(-1, values.shape[1])
                tile_z = spatial_z[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size].reshape(-1, projection.shape[1])
                self.cell_sum_vk_u[row, col] = tile_v.T @ tile_z
        self.prefix_count = cell_prefix(self.cell_count)
        self.prefix_sum_k = cell_prefix(self.cell_sum_k)
        self.prefix_sum_v = cell_prefix(self.cell_sum_v)
        self.prefix_sum_vk_u = cell_prefix(self.cell_sum_vk_u)

    def validate_rect(self, rectangle):
        if len(rectangle) != 4 or any(isinstance(x, bool) or not isinstance(x, int) for x in rectangle):
            raise ValueError('Rectangles require integer [top,left,bottom,right] token coordinates')
        top, left, bottom, right = rectangle
        if not (0 <= top <= bottom <= self.grid[0] and 0 <= left <= right <= self.grid[1]):
            raise ValueError('Rectangle outside token grid')
        if any(value % self.cell_size for value in rectangle):
            raise ValueError('Partial cells are unsupported; every rectangle edge must be cell aligned')
        return tuple(rectangle)

    def rectangle_stats(self, rectangle):
        top, left, bottom, right = [x//self.cell_size for x in self.validate_rect(rectangle)]
        def lookup(prefix):
            return prefix[bottom, right]-prefix[top, right]-prefix[bottom, left]+prefix[top, left]
        return Stats(int(lookup(self.prefix_count)), lookup(self.prefix_sum_k),
                     lookup(self.prefix_sum_v), lookup(self.prefix_sum_vk_u))

    def zero_stats(self):
        return Stats(0, torch.zeros(self.keys.shape[1], dtype=self.keys.dtype),
                     torch.zeros(self.values.shape[1], dtype=self.keys.dtype),
                     torch.zeros(self.values.shape[1], self.projection.shape[1], dtype=self.keys.dtype))

    def parse_selection(self, rectangles, register_ids):
        if len(rectangles) > 2:
            raise ValueError('At most two cell-aligned rectangles are supported')
        rectangles = [self.validate_rect(rectangle) for rectangle in rectangles]
        for token in register_ids:
            if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < len(self.keys):
                raise ValueError('Register IDs must index original tokens')
        unique = sorted(set(register_ids))
        outside = []
        for token in unique:
            y, x = divmod(token, self.grid[1])
            if not any(top <= y < bottom and left <= x < right for top, left, bottom, right in rectangles):
                outside.append(token)
        return rectangles, unique, outside

    def complement(self, rectangles, register_ids=()):
        rectangles, unique, outside = self.parse_selection(rectangles, register_ids)
        retained = self.zero_stats()
        for rectangle in rectangles:
            retained = add(retained, self.rectangle_stats(rectangle))
        if len(rectangles) == 2:
            a, b = rectangles
            intersection = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
            if intersection[0] < intersection[2] and intersection[1] < intersection[3]:
                retained = add(retained, self.rectangle_stats(intersection), -1)
        rectangle_count = retained.count
        if outside:
            ids = torch.tensor(outside, dtype=torch.long)
            retained = add(retained, raw_stats(self.keys[ids], self.values[ids], self.sketch[ids]))
        total = self.rectangle_stats((0, 0, *self.grid))
        result = add(total, retained, -1)
        if not 0 <= result.count <= len(self.keys):
            raise ValueError('Union count outside valid range')
        return result, {'rectangle_union_count': rectangle_count, 'register_input_count': len(register_ids),
                        'unique_register_count': len(unique), 'outside_register_count': len(outside),
                        'retained_unique_count': retained.count, 'complement_count': result.count}

    def direct_complement(self, rectangles, register_ids=()):
        rectangles, _, outside = self.parse_selection(rectangles, register_ids)
        mask = torch.ones(self.grid, dtype=torch.bool)
        for top, left, bottom, right in rectangles:
            mask[top:bottom, left:right] = False
        flat = mask.flatten()
        if outside:
            flat[torch.tensor(outside, dtype=torch.long)] = False
        return raw_stats(self.keys[flat], self.values[flat], self.sketch[flat]), flat

    def storage(self):
        components = {name: tensor.numel()*tensor.element_size() for name, tensor in vars(self).items()
                      if isinstance(tensor, torch.Tensor)}
        count = sum(components.values())
        cross_cache = components['cell_sum_vk_u']+components['prefix_sum_vk_u']
        cell_tile = self.cell_size**2*(self.values.shape[1]+self.projection.shape[1])*self.keys.element_size()
        prefix_working = 2*components['cell_sum_vk_u']
        return {'resident_tensor_bytes_by_component': components, 'resident_tensor_bytes_total': count,
                'raw_cross_moment_cache_bytes_included_above': cross_cache,
                'centered_covariance_cache_bytes': 0,
                'single_query_centered_covariance_bytes': self.values.shape[1]*self.projection.shape[1]*self.keys.element_size(),
                'build_temporary_tensor_allowance_bytes': cell_tile+prefix_working,
                'build_peak_tensor_bytes_estimate': count+cell_tile+prefix_working,
                'estimate_limit': 'Tensor accounting plus explicit packed-cell/prefix temporaries; allocator/BLAS internal workspaces excluded. Direct enumeration validation has additional gathers.'}


def centered_covariance(stats, projection):
    if stats.count == 0:
        return None  # Explicit empty-complement bypass, without changing raw stats.
    return stats.sum_vk_u/stats.count - torch.outer(stats.sum_v/stats.count,
                                                   (stats.sum_k/stats.count) @ projection)


def tensor_error(actual, reference):
    delta = actual.double()-reference.double()
    norm = reference.double().norm().item()
    return {'max_abs': delta.abs().max().item(), 'l2': delta.norm().item(),
            'relative_l2': delta.norm().item()/norm if norm else None, 'reference_l2': norm}


def stats_error(actual, reference):
    return {'count_error': actual.count-reference.count,
            **{name: tensor_error(getattr(actual, name), getattr(reference, name))
               for name in ('sum_k', 'sum_v', 'sum_vk_u')}}


def declared_selections():
    width = 256
    final_cell = [y*width+x for y in range(240, 256) for x in range(240, 256)]
    return [
        ('full_complement', [], []),
        ('empty_complement', [[0, 0, 256, 256]], []),
        ('overlapping_rectangles', [[0, 0, 128, 128], [64, 64, 192, 192]], [0, 0, 65535, 65535, 51400]),
        ('duplicate_registers_only', [], [0, 1, 1, 512, 65535, 65535]),
        ('last_cell_boundary', [[240, 240, 256, 256]], [65535, 65535, 0, 0]),
        ('disjoint_edge_strips', [[0, 0, 16, 256], [240, 0, 256, 256]], []),
        ('near_full_single_token_complement', [[0, 0, 240, 256], [240, 0, 256, 240]], final_cell[:-1]),
    ]


def run_audit():
    seed = 20261018
    generator = torch.Generator(device='cpu').manual_seed(seed)
    tick = time.perf_counter()
    keys = torch.randn(65536, 128, generator=generator, dtype=torch.float64)
    values = .35*keys + .8*torch.randn(65536, 128, generator=generator, dtype=torch.float64)
    projection = fixed_projection(128, 8, seed+1)
    setup_seconds = time.perf_counter()-tick
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    tick = time.perf_counter()
    index = CellIndex(keys, values, projection, (256, 256), 16)
    build_seconds = time.perf_counter()-tick
    rss_after_build = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    results = []
    for name, rectangles, registers in declared_selections():
        tick = time.perf_counter()
        actual, accounting = index.complement(rectangles, registers)
        query_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        direct, mask = index.direct_complement(rectangles, registers)
        direct_seconds = time.perf_counter()-tick
        actual_cov, direct_cov = centered_covariance(actual, projection), centered_covariance(direct, projection)
        results.append({'name': name, 'rectangles_token_coordinates': rectangles,
                        'register_ids': registers, 'accounting': accounting,
                        'raw_statistics_error': stats_error(actual, direct),
                        'centered_covariance_vs_direct_moment_formula': tensor_error(actual_cov, direct_cov) if actual_cov is not None else None,
                        'query_seconds': query_seconds, 'direct_enumeration_seconds': direct_seconds})
    storage = index.storage()
    near_name, near_rectangles, near_registers = declared_selections()[-1]
    tick = time.perf_counter()
    index32 = CellIndex(keys.float(), values.float(), projection.float(), (256, 256), 16)
    build32_seconds = time.perf_counter()-tick
    approx32, accounting32 = index32.complement(near_rectangles, near_registers)
    direct32, mask32 = index32.direct_complement(near_rectangles, near_registers)
    # One residual token has exactly zero covariance when formed from centered
    # rows. Raw-moment recentering can leave error; preserve it without clipping.
    vc, zc = index32.values[mask32], index32.sketch[mask32]
    direct_centered = (vc-vc.mean(0)).T @ (zc-zc.mean(0)) / len(vc)
    covariance32 = centered_covariance(approx32, index32.projection)
    rss_final = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    return {'scope': 'One synthetic CPU head; cost/accounting only, no quality, novelty or GPU benchmark',
            'seed': seed, 'projection_seed': seed+1, 'grid_tokens': [256, 256], 'base_cell_tokens': [16, 16],
            'cell_grid': [16, 16], 'd': 128, 'dv': 128, 'rank': 8,
            'target_domain': 'K is interpreted as already post-RoPE, V as its corresponding values. No actual RoPE transform is simulated.',
            'projection': 'Fixed shared orthonormal U; z=K U. Projection remains identical for global/retained/complement statistics.',
            'orthonormality_max_abs_error': (projection.T@projection-torch.eye(8, dtype=torch.float64)).abs().max().item(),
            'moment_definition': 'n, sum(k), sum(v), sum[v (kU)^T]; raw moments are added/subtracted before single-cloud recentering.',
            'prefix_definition': '2D prefix over16x16 base CELLS, not per-token outer products.',
            'restrictions': ['At most two cell-aligned rectangles; partial cells rejected.',
                'Register IDs refer to original tokens; duplicates and IDs already inside rectangle union are removed.',
                'This yields a single complement-cloud covariance, not PISA mean within-cell covariance or per-cell centroid routing.',
                'No per-cell recentering after individual register deletion and no general AReA integration are implemented.',
                'PISA, MomentKV and Scatterbrain are prior-art boundaries; no novelty claim follows from these statistics.',
                'No adaptive clipping, error-based case selection or silent FP32 repair is performed.'],
            'versions': {'torch': torch.__version__}, 'cpu_threads': torch.get_num_threads(),
            'float64_cases': results, 'float64_storage': storage,
            'timing': {'synthetic_inputs_and_projection_setup_seconds': setup_seconds,
                       'one_head_float64_build_seconds': build_seconds,
                       'seven_declared_query_seconds_total': sum(item['query_seconds'] for item in results),
                       'query_scope': 'Raw moment retrieval includes rectangle inclusion-exclusion, register dedup/gathers and global subtraction; covariance recentering and direct validation are outside this timer.',
                       'scope': 'Single CPU pass, includes K U, base moments and cell-prefix build; setup and direct enumeration timed separately. Not a GPU or throughput benchmark.'},
            'linear_24_head_arithmetic_only': {
                'resident_bytes_same_float64_dtype': 24*storage['resident_tensor_bytes_total'],
                'serial_build_seconds_24_times_measured_one_head': 24*build_seconds,
                'query_seconds_24_times_measured_seven_queries': 24*sum(item['query_seconds'] for item in results),
                'interpretation': 'Scalar multiplication only; no24-head arrays, parallelism, GPU latency or model cost measured.'},
            'float32_near_full_cancellation': {'name': near_name, 'accounting': accounting32,
                'raw_statistics_error_against_direct_same_precision': stats_error(approx32, direct32),
                'centered_covariance_error_against_centered_rows': tensor_error(covariance32, direct_centered),
                'covariance_min': covariance32.min().item(), 'covariance_max': covariance32.max().item(),
                'direct_centered_covariance_l2': direct_centered.norm().item(),
                'storage': index32.storage(), 'build_seconds_including_float32_conversion': build32_seconds,
                'interpretation': 'Predetermined one-token complement. Reports FP32 aggregation/subtraction/recentering error, including projection arithmetic; no clipping or PSD repair.'},
            'process_peak_rss_bytes': {'before_build': rss_before, 'after_float64_build': rss_after_build, 'final': rss_final,
                'scope': 'Linux process high-water RSS includes imports, allocator and direct-enumeration validation; final includes both FP64 and FP32 indexes.'},
            'source_sha256': {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'novelty_established': False, 'efficacy_evaluated': False, 'gpu_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'timing': report['timing'], 'storage_bytes': report['float64_storage']['resident_tensor_bytes_total'],
                      'fp32_covariance_error': report['float32_near_full_cancellation']['centered_covariance_error_against_centered_rows']}, indent=2))


if __name__ == '__main__':
    main()
