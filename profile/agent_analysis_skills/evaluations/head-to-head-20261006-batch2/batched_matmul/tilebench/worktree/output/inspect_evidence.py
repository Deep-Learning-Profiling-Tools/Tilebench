"""Saved-input inspection only; no benchmark or GPU imports."""
import ast
import hashlib
import json
import sys
from pathlib import Path

import ncu_report

sys.path.insert(0, str(Path('tools').resolve()))
from csv_ratios import compare_references
from sass_listing import parse_listing

OUT = Path('output')
selected = []
source_inventory = []
reports = json.loads(Path('evidence/reports.json').read_text())
wanted = {
    'device__attribute_display_name', 'device__attribute_compute_capability_major',
    'device__attribute_compute_capability_minor', 'device__attribute_multiprocessor_count',
    'gpu__time_duration.sum', 'smsp__inst_executed.sum',
    'launch__grid_size', 'launch__grid_dim_x', 'launch__grid_dim_y', 'launch__grid_dim_z',
    'launch__block_size', 'launch__registers_per_thread', 'launch__shared_mem_per_block',
    'launch__occupancy_limit_registers', 'launch__occupancy_limit_shared_mem',
    'launch__waves_per_multiprocessor', 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__warps_eligible.avg.per_cycle_active',
    'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'dram__bytes_read.sum', 'dram__bytes_write.sum',
    'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
    'smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio',
    'sm__ops_path_tensor_src_fp16_dst_fp32.sum', 'profiler__replayer_passes',
    'profiler__replayer_bytes_mem_backed_up.sum',
    'smsp__sass_inst_executed_op_tma_ld.sum', 'smsp__sass_inst_executed_op_tma_st.sum',
    'smsp__sass_inst_executed_op_utcmma.sum',
}
for spec in reports:
    backend = spec['backend']
    exported = json.loads((OUT / f'{backend}_ncu.json').read_text())
    assert exported['report_sha256'] == spec['sha256']
    selected.extend(dict(m, backend=backend) for m in exported['metrics'] if m['metric'] in wanted)
    report = ncu_report.load_report(spec['path'])
    for ri in range(report.num_ranges()):
        current = report.range_by_idx(ri)
        for ai in range(current.num_actions()):
            action = current.action_by_idx(ai)
            for index, (original, content) in enumerate(action.source_files().items()):
                item = dict(backend=backend, report=spec['path'], range_index=ri,
                            action_index=ai, kernel=action.name(), original_path=original,
                            characters=len(content), sha256=hashlib.sha256(content.encode()).hexdigest())
                if content:
                    dest = OUT / f'{backend}_source_{index}_{Path(original).name}'
                    dest.write_text(content)
                    item['extracted_path'] = str(dest)
                source_inventory.append(item)
    listing = parse_listing((OUT / f'{backend}_sass.txt').read_text())
    (OUT / f'{backend}_sass_inventory.json').write_text(json.dumps(listing, indent=2) + '\n')
    print(backend, 'SASS status', listing['status'], 'warnings', listing['warnings'])
    print('Relevant opcodes', {k:v for k,v in listing['kernels'][0]['opcodes'].items()
                             if k.startswith(('UTC', 'UTMA', 'LDGSTS', 'LDTM', 'STG', 'BAR', 'CPASYNC'))})

(OUT / 'selected_records.json').write_text(json.dumps(selected, indent=2) + '\n')
(OUT / 'source_inventory.json').write_text(json.dumps(source_inventory, indent=2) + '\n')
csv = compare_references('results/B200/csv/batched_matmul_autotune.csv',
                        {'params':'M=640', 'dtype':'fp16'}, 'tilelang_ms',
                        ['triton_ms', 'cutile_ms', 'torch_ms'])
(OUT / 'csv_comparison.json').write_text(json.dumps(csv, indent=2) + '\n')
print(json.dumps(csv, indent=2))

def kernel_ast(path):
    return ast.dump(next(node for node in ast.parse(Path(path).read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == 'bmm_kernel'))

correspondence = {}
for backend, saved in [('triton', 'output/triton_source_1_impl_triton.py'),
                       ('cutile', 'output/cutile_source_0_impl_cutile.py')]:
    current = f'tilebench/benchmarks/operators/batched_matmul/impl_{backend}.py'
    correspondence[backend] = {'embedded_source': saved, 'current_source': current,
                              'kernel_ast_equal': kernel_ast(saved) == kernel_ast(current)}
correspondence['tilelang'] = {
    'current_source': 'tilebench/benchmarks/operators/batched_matmul/impl_tilelang.py',
    'kernel_ast_equal': None, 'limit': 'Embedded tvm_kernels.cu is empty; SASS and headers recovered.'}
(OUT / 'source_correspondence.json').write_text(json.dumps(correspondence, indent=2) + '\n')

winners = json.loads(Path('evidence/batched_matmul_autotune.json').read_text())
winner = next(row for row in winners if row['params'] == {'BATCH':32, 'M':640}
              and row['dtype'] == 'fp16')
tl_winners = json.loads(Path('evidence/batched_matmul_tilelang_autotune.json').read_text())
tl_winner = next(row for row in tl_winners if row['params'] == {'BATCH':32, 'M':640}
                 and row['dtype'] == 'fp16')
assert winner['tilelang_autotune_cfg'] == tl_winner['tilelang_autotune_cfg']
mechanisms = [
    dict(claim='TileLang operand delivery and completion dependencies underfeed tensor compute',
         status='inferred',
         code_difference='Lane-issued LDGSTS copies, phase waits and block synchronization versus TMA transfers; current TileLang explicitly disables warp specialization and synchronizes each K tile.',
         expected_consequences='Lower tensor activity and fewer active warps despite comparable CTA residency ceilings.',
         observations=['TileLang SASS LDGSTS.E.BYPASS.128, UTCHMMA, UTCBAR, SYNCS.PHASECHK.TRANS64.TRYWAIT and BAR.SYNC.DEFER_BLOCKING',
                       'selected_records.json: tensor activity, active warps, launch residency limits; all report range 0/action 0'],
         alternative='Replay/cache differences or tensor-instruction granularity and epilogue costs also contribute; TileLang stall and eligible-warp metrics are uncollected.',
         limits='No elapsed-time attribution or quantitative share of benchmark gap; TileLang capture-time kernel source missing.'),
    dict(claim='cuTile and Triton are benchmark parity; counters show different scheduling tradeoffs',
         status='observed', code_difference='cuTile uses 256 threads and role-specific register deallocation; Triton uses 128 threads and cached B transpose.',
         expected_consequences='Higher cuTile active occupancy but lower warp instruction issue and eligibility; role-idle waits can be large.',
         observations=['selected_records.json: launch block size, registers, active warps, instructions, eligibility, long-scoreboard ratio; range 0/action 0'],
         alternative='Role-specialized waits prevent interpreting whole-kernel stall averages as useful-compute critical-path stalls.',
         limits='8.5% CSV latency difference is within 10% parity convention; no independent claimed cause of that small gap.'),
]
comparison = {
    'case': {'hardware':'B200', 'operator':'batched_matmul', 'dtype':'fp16',
             'shape':{'BATCH':32, 'M':640, 'N':640, 'K':640},
             'semantics':'C[b]=A[b]@B[b], FP32 accumulation and FP16 output', 'tuning_mode':'autotune'},
    'provenance': {'supplied_source_revision':'17d2d4f6', 'supplied_archive_revision':'9455c0bd',
                   'reports':reports, 'hashes_verified':True, 'reader':'NCU 2026.1.1 Python API',
                   'limits':'Dataset revision/hash identify bytes, not capture command/compiler versions; no capture manifest supplied.'},
    'implementations': [
        {'backend':b, 'benchmark':{'path':csv['csv'], 'csv_line':59,
          'value':float(csv['selected_row']['case'][b+'_ms']), 'unit':'ms'},
         'selected_configuration':winner[b+'_autotune_cfg'],
         'profile_actions':json.loads((OUT / f'{b}_ncu.json').read_text())['actions'][0] | {'metric_names':'see full export'},
         'source':correspondence[b],
         'configuration_correspondence':'cuTile signature explicitly matches K tiles, BM/BN/BK and grouping; Triton/TileLang launch/resources/SASS consistent, exact config-to-capture manifest absent.'}
        for b in ['tilelang','triton','cutile']],
    'timing':{'benchmark_comparisons':csv['comparisons'],
              'profiler_durations_ns':{b:next(r['value'] for r in selected if r['backend']==b and r['metric']=='gpu__time_duration.sum')
                                       for b in ['tilelang','triton','cutile']},
              'profiler_tilelang_over_triton':48576/32064,
              'profiler_tilelang_over_cutile':48576/34848},
    'mechanisms':mechanisms}
(OUT / 'comparison_evidence.json').write_text(json.dumps(comparison, indent=2) + '\n')
coverage = [
    {'lens':'useful work/instruction path', 'status':'supported',
     'metrics':['smsp__inst_executed.sum'], 'missing_metrics':[],
     'evidence':'All reports range 0/action 0; selected winner logs, source bodies and saved UTCHMMA SASS.',
     'conclusion':'Same mathematical work; instruction totals do not predict ordering.'},
    {'lens':'launch/residency', 'status':'supported',
     'metrics':['launch__grid_size','launch__block_size','launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem','sm__warps_active.avg.pct_of_peak_sustained_active'],
     'missing_metrics':[], 'evidence':'All reports range 0/action 0.',
     'conclusion':'TileLang and Triton share three-CTA limits; their achieved activity differs.'},
    {'lens':'memory movement/reuse', 'status':'insufficient',
     'metrics':['dram__bytes_read.sum','dram__bytes_write.sum'], 'missing_metrics':[],
     'evidence':'All reports range 0/action 0; TileLang LDGSTS, peers UTMALDG SASS.',
     'conclusion':'Delivery paths differ, but missing capture cache metadata and incomparable HBM writeback prevent a bandwidth explanation.'},
    {'lens':'issue/dependencies/synchronization', 'status':'insufficient',
     'metrics':['smsp__warps_eligible.avg.per_cycle_active','smsp__issue_active.avg.pct_of_peak_sustained_active'],
     'missing_metrics':['smsp__warps_eligible.avg.per_cycle_active','smsp__issue_active.avg.pct_of_peak_sustained_active'],
     'evidence':'Triton/cuTile range 0/action 0 collect these metrics; missing_metrics applies to TileLang range 0/action 0. TileLang completion waits visible in SASS.',
     'conclusion':'Leading synchronization inference is structurally supported, without TileLang stall localization.'},
    {'lens':'compute pipeline', 'status':'supported',
     'metrics':['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'], 'missing_metrics':[],
     'evidence':'All reports range 0/action 0; UTCHMMA and LDTM present in all.',
     'conclusion':'TileLang tensor activity lower; absence of tensor cores rejected.'},
    {'lens':'balance/temporal behavior', 'status':'insufficient',
     'metrics':['launch__waves_per_multiprocessor'], 'missing_metrics':[],
     'evidence':'All reports range 0/action 0; no saved timeline inspected.',
     'conclusion':'Few scheduling waves may matter; no measured tail duration or host launch cost.'}]
(OUT / 'diagnostic_coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')
