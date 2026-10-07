import ast
import hashlib
import json
from pathlib import Path
import sys

import ncu_report

sys.path.insert(0, str(Path('tools').resolve()))
from csv_ratios import compare_references

OUT = Path('output')
backends = ['tilelang', 'triton', 'cutile']
selected = []
provenance = []
keys = {
    'device__attribute_display_name', 'device__attribute_multiprocessor_count',
    'device__attribute_compute_capability_major', 'device__attribute_compute_capability_minor',
    'device__attribute_max_warps_per_multiprocessor',
    'gpu__time_duration.sum', 'launch__block_size', 'launch__grid_size',
    'launch__grid_dim_x', 'launch__grid_dim_y', 'launch__grid_dim_z',
    'launch__registers_per_thread', 'launch__shared_mem_per_block',
    'launch__occupancy_limit_registers', 'launch__occupancy_limit_warps',
    'sm__warps_active.avg.pct_of_peak_sustained_active', 'smsp__inst_executed.sum',
    'dram__bytes_read.sum', 'dram__bytes_write.sum',
    'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    'l1tex__t_sector_hit_rate.pct', 'lts__t_sector_hit_rate.pct', 'lts__t_sectors.sum',
    'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
    'smsp__warps_eligible.avg.per_cycle_active',
    'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed',
    'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed',
    'sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed',
    'sm__inst_executed_pipe_tma.avg.pct_of_peak_sustained_elapsed',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum',
    'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
    'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
    'smsp__sass_inst_executed_op_global_ld.sum',
    'launch__waves_per_multiprocessor', 'profiler__replayer_passes',
}
for kind in ['long_scoreboard', 'short_scoreboard', 'math_pipe_throttle',
             'lg_throttle', 'mio_throttle', 'barrier', 'wait', 'not_selected']:
    keys.add('smsp__average_warps_issue_stalled_' + kind + '_per_issue_active.ratio')

manifest = {r['backend']: r for r in json.loads(Path('evidence/reports.json').read_text())}
for backend in backends:
    data = json.loads((OUT / (backend + '_ncu.json')).read_text())
    assert data['report_sha256'] == manifest[backend]['sha256']
    selected.extend({**m, 'backend': backend} for m in data['metrics'] if m['metric'] in keys)
    report = ncu_report.load_report(data['report'])
    action = report.range_by_idx(0).action_by_idx(0)
    identity = {'backend': backend, 'report': data['report'], 'range_index': 0,
                'action_index': 0, 'kernel': action.name()}
    sources = []
    for i, (name, contents) in enumerate(action.source_files().items()):
        item = {'path': name, 'characters': len(contents)}
        if contents:
            dest = OUT / (backend + '_source_' + str(i) + Path(name).suffix)
            dest.write_text(contents)
            item['saved'] = str(dest)
            current = Path('tilebench/benchmarks/operators/gaussian_blur') / ('impl_' + backend + '.py')
            item['exact_current_match'] = contents == current.read_text()
            def kernel_ast(text):
                return ast.dump(next(n for n in ast.parse(text).body
                                     if isinstance(n, ast.FunctionDef)
                                     and n.name == 'gaussian_blur_kernel'))
            item['kernel_ast_current_match'] = kernel_ast(contents) == kernel_ast(current.read_text())
        sources.append(item)
    for name in ['sass__inst_executed_per_opcode', 'sass__inst_executed_per_opcode_with_modifier_all']:
        metric = action[name]
        ids = metric.correlation_ids()
        for i in range(metric.num_instances()):
            selected.append({**identity, 'metric': name, 'value': metric.value(i),
                             'unit': metric.unit(), 'instance_index': i,
                             'correlation_id': ids.value(i)})
    pc_metric = action['smsp__pcsamp_warps_issue_stalled_long_scoreboard']
    ids = pc_metric.correlation_ids()
    pcs = sorted(set(ids.value(i) for i in range(pc_metric.num_instances())))
    lines = [f'{pc:#x}: {action.sass_by_pc(pc)}' for pc in pcs]
    (OUT / (backend + '_sass.txt')).write_text('\n'.join(lines) + '\n')
    provenance.append({**identity, 'sha256_verified': True, 'reader_version': report.get_version(),
                       'sources': sources, 'sass_addresses': len(pcs)})

(OUT / 'selected_records.json').write_text(json.dumps(selected, indent=2) + '\n')
(OUT / 'capture_checks.json').write_text(json.dumps(provenance, indent=2) + '\n')
ratios = compare_references('results/B200/csv/gaussian_blur_autotune.csv',
                            {'params': 'input_rows=10240', 'dtype': 'fp32'},
                            'tilelang_ms', ['triton_ms', 'cutile_ms'])
(OUT / 'csv_comparison.json').write_text(json.dumps(ratios, indent=2) + '\n')
print(json.dumps(provenance, indent=2))
for backend in backends:
    print(backend, [(m['correlation_id'], m['value']) for m in selected
                    if m['backend'] == backend and m['metric'] == 'sass__inst_executed_per_opcode'])
print(json.dumps(ratios, indent=2))

configs = {'tilelang': {'BLOCK_R': 4, 'BLOCK_C': 256, 'threads': 128},
           'triton': {'BLOCK_R': 8, 'BLOCK_C': 64, 'num_warps': 8},
           'cutile': {'tile_r': 2, 'tile_c': 128, 'occupancy': 4}}
latencies = {'tilelang': 1.1436, 'triton': 1.2487, 'cutile': 1.2713}
evidence = {
    'case': {'hardware': 'B200', 'operator': 'gaussian_blur', 'dtype': 'fp32',
             'shape': [10240, 10240], 'kernel_shape': [7, 7], 'tuning_mode': 'autotuned',
             'semantics': 'Direct zero-padded stencil, FP32 accumulation, one output per input pixel'},
    'implementations': [
        {'backend': b, 'benchmark_path': ratios['csv'], 'benchmark_csv_line': 41,
         'benchmark_value': latencies[b], 'benchmark_unit': 'ms',
         'selected_configuration': configs[b], 'profile': provenance[i],
         'source_path': 'tilebench/benchmarks/operators/gaussian_blur/impl_' + b + '.py',
         'source_revision_supplied': '17d2d4f6', 'winner_archive_revision_supplied': '9455c0bd',
         'profile_dataset_revision': manifest[b]['dataset_revision'],
         'provenance_limits': 'No capture command/stack manifest; TileLang CUDA source empty; cuTile occupancy hint only from winner log'}
        for i, b in enumerate(backends)],
    'timing': {'benchmark': ratios['comparisons'],
               'profiler_duration_ns': {r['backend']: r['value'] for r in selected
                                        if r['metric'] == 'gpu__time_duration.sum'},
               'profiler_ratio_triton_over_tilelang': 1274624 / 1134976,
               'profiler_ratio_cutile_over_tilelang': 1279904 / 1134976},
    'mechanisms': [
        {'claim': 'TileLang amortizes weights/indexing across more outputs per thread and contiguous lanes',
         'status': 'inferred', 'code_difference': '4x256/128 versus 8x64/256 and 2x128/128; 8 versus 2 outputs/thread',
         'expected_consequences': 'Fewer repeated coefficient loads and bounds/address instructions; less L1 load-sector work',
         'observations': ['selected_records.json: smsp__inst_executed.sum, LDG/ISETP/IMAD instances; L1 sectors/requests',
                          'tilelang_sass.txt prologue, triton_sass.txt lane mapping'],
         'alternative': 'TileLang large register state reduces latency hiding',
         'limits': 'No controlled config comparison; fastest non-TileLang CSV gap within 10%'},
        {'claim': 'Triton has cache-side transaction pressure despite high concurrency',
         'status': 'inferred', 'code_difference': 'Interleaved adjacent-column pairs, scalar and LDG.E.64 loads',
         'expected_consequences': 'Extra sectors and load-queue stalls without proportional extra HBM traffic',
         'observations': ['selected_records.json: L1 throughput, sectors, lg_throttle ratio, HBM bytes'],
         'alternative': 'Vector loads reduce input load instruction work; many ready warps hide dependencies',
         'limits': 'Stall ratios are per issue-active, not elapsed-time shares'},
        {'claim': 'cuTile gather bounds/index lowering and store layout conversion add overhead',
         'status': 'inferred', 'code_difference': 'Runtime row loop with repeated checks; FFMA2 compute layout converted via STSM/BAR/LDS',
         'expected_consequences': 'More bounds/index instructions, ALU/issue activity and synchronization despite efficient paired FMA',
         'observations': ['cutile_sass.txt loop/store epilogue',
                          'selected_records.json: ISETP/VIADD/FFMA2 instances, ALU activity, barrier ratio'],
         'alternative': 'FFMA2 halves FMA instruction count and limits register footprint',
         'limits': 'No measured component time shares or compiler defect attribution'}]}
(OUT / 'comparison_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
coverage = []
for b in backends:
    for lens, metrics, conclusion in [
        ('useful work', ['smsp__inst_executed.sum', 'sass__inst_executed_per_opcode'], 'Arithmetic matched; overhead differs'),
        ('residency', ['launch__registers_per_thread', 'launch__occupancy_limit_registers', 'sm__warps_active.avg.pct_of_peak_sustained_active'], 'TileLang register limit trades concurrency for work amortization'),
        ('memory', ['dram__bytes_read.sum', 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'], 'Near-equal HBM traffic; different cache-side transaction work'),
        ('issue/dependencies', ['smsp__warps_eligible.avg.per_cycle_active', 'smsp__issue_active.avg.pct_of_peak_sustained_active'], 'Issue pressure and concurrency support tradeoff, not timing shares'),
        ('compute', ['sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed', 'sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed'], 'Scalar FFMA versus paired FFMA2; indexing pressure differs'),
        ('balance', ['launch__grid_size'], 'Large grid; no saved timeline analyzed, tail timing unresolved')]:
        coverage.append({'backend': b, 'lens': lens, 'status': 'insufficient' if lens == 'balance' else 'supported',
                         'metrics': metrics, 'missing_metrics': [],
                         'evidence': {'report': manifest[b]['path'], 'range_index': 0, 'action_index': 0},
                         'conclusion': conclusion})
(OUT / 'diagnostic_coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')
