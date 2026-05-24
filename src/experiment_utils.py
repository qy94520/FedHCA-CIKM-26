import csv
import fcntl
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
DEFAULT_RESULT_ROOT = './outputs'
DEFAULT_OUTPUT_ROOT = os.path.join(DEFAULT_RESULT_ROOT, 'results')
DEFAULT_LOG_ROOT = os.path.join(DEFAULT_RESULT_ROOT, 'logs')
SUMMARY_FIELDNAMES = ['run_tag', 'dataset', 'method', 'exp_type', 'variant', 'async_timing_mode', 'timing_pattern_desc', 'timing_pattern_csv', 'auc_window', 'ACC_all', 'ACC_old', 'ACC_new(avg)', 'AUC_new', 'seed_count', 'seeds', 'acc_all_mean', 'acc_all_std', 'acc_old_mean', 'acc_old_std', 'acc_new_avg_mean', 'acc_new_avg_std', 'auc_new_mean', 'auc_new_std', 'comm_model_only', 'comm_extra_proto_meta', 'comm_total', 'comm_ratio_vs_fedavg', 'server_time_mean', 'server_time_ratio_vs_fedavg', 'per_round_csv', 'legacy_acc_global_csv', 'legacy_acc_taskwise_csv']

def sanitize_tag(value: object, default: str='default') -> str:
    text = str(value).strip()
    text = re.sub('[^0-9A-Za-z._-]+', '_', text)
    text = text.strip('._-')
    return text or default

def sigma_to_tag(sigma: float) -> str:
    return sanitize_tag(f'{float(sigma):.2f}'.replace('.', 'p'), default='0p00')

def detect_cli_flags(argv: Optional[Sequence[str]]=None) -> set:
    argv = list(argv) if argv is not None else []
    flags = set()
    for token in argv:
        if not str(token).startswith('--'):
            continue
        name = str(token).split('=', 1)[0]
        flags.add(name)
    return flags

def validate_cumulative_schedule(cumulative: Sequence[int], total_cap: int=100, num_stages: int=4) -> List[int]:
    vals = [int(x) for x in cumulative]
    if len(vals) != int(num_stages):
        raise ValueError(f'Invalid cumulative schedule length: expected {num_stages}, got {len(vals)}')
    if vals[-1] != int(total_cap):
        raise ValueError(f'Invalid cumulative schedule tail: expected {total_cap}, got {vals[-1]}')
    prev = 0
    for idx, value in enumerate(vals):
        if value <= prev:
            raise ValueError(f'Cumulative schedule must be strictly increasing: idx={idx}, prev={prev}, cur={value}')
        if value - prev < 1:
            raise ValueError(f'Each stage must add at least 1 sample: idx={idx}, prev={prev}, cur={value}')
        prev = value
    return vals

def build_random_cumulative_schedule(total_cap: int=100, num_stages: int=4, seed: int=0) -> List[int]:
    total_cap = int(total_cap)
    num_stages = int(num_stages)
    if total_cap < num_stages:
        raise ValueError(f'total_cap={total_cap} must be >= num_stages={num_stages}')
    rng = np.random.default_rng(int(seed))
    cut_pool = np.arange(1, total_cap, dtype=int)
    cuts = sorted((int(x) for x in rng.choice(cut_pool, size=num_stages - 1, replace=False).tolist()))
    return validate_cumulative_schedule(cuts + [total_cap], total_cap=total_cap, num_stages=num_stages)

def cumulative_to_increment(cumulative: Sequence[int]) -> List[int]:
    prev = 0
    increments = []
    for value in cumulative:
        cur = int(value)
        increments.append(cur - prev)
        prev = cur
    return increments

def resolve_exposure_cumulative(exposure_mode: str, seed: int, default_increment_splits: Optional[Sequence[int]]=None, total_cap: int=100, num_stages: int=4) -> List[int]:
    mode = str(exposure_mode).strip().lower()
    if mode == 'exponential':
        base = list(default_increment_splits or [3, 9, 24, 64])
        cumulative = np.cumsum(np.asarray(base, dtype=int)).tolist()
    elif mode == 'linear':
        if int(total_cap) == 100 and int(num_stages) == 4:
            cumulative = [25, 50, 75, 100]
        else:
            step = int(total_cap) // int(num_stages)
            cumulative = [step * (idx + 1) for idx in range(int(num_stages) - 1)] + [int(total_cap)]
    elif mode == 'random_cumulative':
        cumulative = build_random_cumulative_schedule(total_cap=total_cap, num_stages=num_stages, seed=seed)
    else:
        raise ValueError(f'Unsupported exposure_mode: {exposure_mode}')
    return validate_cumulative_schedule(cumulative, total_cap=total_cap, num_stages=num_stages)

def infer_experiment_identity(*, method: str, server_ablation: str='full', async_timing_mode: str='fixed_default', exposure_mode: str='exponential', proto_noise_sigma: float=0.0, profile_overhead: bool=False, cli_flags: Optional[Iterable[str]]=None, exp_type_override: Optional[str]=None, variant_override: Optional[str]=None) -> Tuple[str, str]:
    if exp_type_override is not None and variant_override is not None:
        return (str(exp_type_override), str(variant_override))
    flags = set(cli_flags or [])
    method_name = str(method)
    if bool(profile_overhead):
        return ('overhead', 'profile')
    if method_name == 'FedHCA' and ('--server_ablation' in flags or str(server_ablation).strip().lower() != 'full'):
        return ('server_ablation', str(server_ablation))
    if '--async_timing_mode' in flags or str(async_timing_mode).strip().lower() != 'fixed_default':
        return ('async_timing', str(async_timing_mode))
    if '--exposure_mode' in flags or str(exposure_mode).strip().lower() != 'exponential':
        return ('exposure', str(exposure_mode))
    if method_name == 'FedHCA' and ('--proto_noise_sigma' in flags or float(proto_noise_sigma) > 0.0):
        return ('proto_noise', f'sigma_{float(proto_noise_sigma):.2f}')
    return ('main', 'default')

def build_output_context(*, run_tag: str, dataset: str, method: str, seed: int, output_root: Optional[str], log_root: Optional[str], exp_type: str, variant: str) -> Dict[str, str]:
    safe_run_tag = sanitize_tag(run_tag or 'run')
    safe_dataset = sanitize_tag(dataset)
    safe_method = sanitize_tag(method)
    safe_exp_type = sanitize_tag(exp_type)
    safe_variant = sanitize_tag(variant)
    output_root_abs = os.path.abspath(output_root or DEFAULT_OUTPUT_ROOT)
    log_root_abs = os.path.abspath(log_root or DEFAULT_LOG_ROOT)
    per_round_dir = os.path.join(output_root_abs, 'per_round')
    meta_dir = os.path.join(output_root_abs, 'meta')
    patterns_dir = os.path.join(output_root_abs, 'patterns')
    os.makedirs(output_root_abs, exist_ok=True)
    os.makedirs(log_root_abs, exist_ok=True)
    os.makedirs(per_round_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(patterns_dir, exist_ok=True)
    stem = f'{safe_run_tag}_{safe_dataset}_{safe_method}_{safe_exp_type}_{safe_variant}_seed{int(seed)}'
    return {'run_tag': safe_run_tag, 'dataset_tag': safe_dataset, 'method_tag': safe_method, 'exp_type_tag': safe_exp_type, 'variant_tag': safe_variant, 'output_root': output_root_abs, 'log_root': log_root_abs, 'summary_csv': os.path.join(output_root_abs, f'{safe_run_tag}_{safe_dataset}_{safe_method}_{safe_exp_type}_summary.csv'), 'per_round_csv': os.path.join(per_round_dir, f'{stem}.csv'), 'pattern_csv': os.path.join(patterns_dir, f'{stem}.csv'), 'log_file': os.path.join(log_root_abs, f'{stem}.log'), 'meta_json': os.path.join(meta_dir, f'{stem}.json')}

def safe_float(value: object, default: float=0.0) -> float:
    if value in (None, '', 'None', 'nan', 'NaN'):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)

def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if len(vals) == 0:
        return (0.0, 0.0)
    if len(vals) == 1:
        return (float(vals[0]), 0.0)
    arr = np.asarray(vals, dtype=np.float64)
    return (float(arr.mean()), float(arr.std(ddof=1)))

def compute_acc_new_avg_from_taskwise_row(row: Dict[str, str]) -> float:
    incre_vals = []
    for key, value in row.items():
        if not str(key).startswith('incre_'):
            continue
        if value in (None, ''):
            continue
        incre_vals.append(safe_float(value))
    if len(incre_vals) == 0:
        return 0.0
    return float(np.mean(np.asarray(incre_vals, dtype=np.float64)))

def _task_start_round(task_id: int, task_global_round: int, first_exposure_offset: int) -> int:
    return 1 + (int(task_id) - 1) * int(task_global_round) + int(first_exposure_offset)

def build_async_timing_pattern(mode: str, seed: int, task_id: int, task_global_round: int=10, num_new_clients: int=4) -> List[int]:
    mode_l = str(mode).strip().lower()
    task_global_round = int(task_global_round)
    num_new_clients = int(num_new_clients)
    if task_global_round < 8:
        raise ValueError(f'task_global_round must be >= 8 for the timing robustness experiment, got {task_global_round}')
    if num_new_clients != 4:
        raise ValueError(f'Async timing robustness experiment expects num_new_clients=4, got {num_new_clients}')
    if mode_l == 'fixed_default':
        pattern = [0, 3, 6, 6]
    elif mode_l == 'fixed_staggered':
        pattern = [0, 2, 4, 6]
    elif mode_l == 'random_timing':
        rng = np.random.default_rng(int(seed) + int(task_id))
        delayed = rng.choice(np.arange(1, 7, dtype=int), size=3, replace=True).tolist()
        pattern = sorted([0] + [int(x) for x in delayed])
    else:
        raise ValueError(f'Unsupported async_timing_mode: {mode}')
    if len(pattern) != num_new_clients:
        raise ValueError(f'Invalid timing pattern length: expected {num_new_clients}, got {len(pattern)}')
    if pattern[0] != 0:
        raise ValueError(f'Timing pattern must start at internal round 0 (task round 1), got {pattern}')
    if sum((1 for x in pattern if int(x) == 0)) != 1:
        raise ValueError(f'Timing pattern must have exactly one first-round exposure client, got {pattern}')
    if any((int(x) < 0 or int(x) > 6 for x in pattern)):
        raise ValueError(f'Timing pattern values must stay within internal rounds [0,6], got {pattern}')
    return [int(x) for x in pattern]

def build_async_exposure_rounds(mode: str, seed: int, task_id: int, num_clients: int=10, num_new_clients: int=4, task_global_round: int=10) -> List[int]:
    pattern = build_async_timing_pattern(mode=mode, seed=seed, task_id=task_id, task_global_round=task_global_round, num_new_clients=num_new_clients)
    rounds = list(pattern)
    rounds.extend([int(task_global_round) - 1] * max(0, int(num_clients) - int(num_new_clients)))
    return rounds

def async_timing_pattern_onebased(pattern_zero_based: Sequence[int]) -> List[int]:
    return [int(x) + 1 for x in pattern_zero_based]

def describe_async_timing_mode(mode: str) -> str:
    mode_l = str(mode).strip().lower()
    if mode_l == 'fixed_default':
        return '[1,4,7,7]'
    if mode_l == 'fixed_staggered':
        return '[1,3,5,7]'
    if mode_l == 'random_timing':
        return 'taskwise random: [1] + sample({2,3,4,5,6,7}, size=3, replace=True), then sort'
    return str(mode)

def init_pattern_csv(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['run_tag', 'dataset', 'method', 'async_timing_mode', 'task_id', 'seed', 'timing_pattern_desc', 'timing_pattern_zero_based_json', 'timing_pattern_one_based_json'])

def append_pattern_record(*, csv_path: str, run_tag: str, dataset: str, method: str, async_timing_mode: str, task_id: int, seed: int, timing_pattern_zero_based: Sequence[int], timing_pattern_desc: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pattern_zero = [int(x) for x in timing_pattern_zero_based]
    pattern_one = async_timing_pattern_onebased(pattern_zero)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([str(run_tag), str(dataset), str(method), str(async_timing_mode), int(task_id), int(seed), str(timing_pattern_desc), str(pattern_zero), str(pattern_one)])

def compute_auc_new_progress(rounds: Sequence[int], acc_new_current: Sequence[float], incre_tasks: int, task_global_round: int, exposure_rounds: Sequence[int], new_client_num: int, window: int=4) -> Tuple[float, Dict[int, float], Dict[int, float]]:
    if len(rounds) != len(acc_new_current):
        raise ValueError('rounds and acc_new_current must have the same length')
    if len(rounds) == 0:
        return (0.0, {}, {})
    round_to_acc = {int(r): float(a) for r, a in zip(rounds, acc_new_current)}
    first_exposure_offset = 0
    if int(new_client_num) > 0 and len(exposure_rounds) > 0:
        first_exposure_offset = min((int(x) for x in list(exposure_rounds)[:int(new_client_num)]))
    task_auc_map: Dict[int, float] = {}
    progress_map: Dict[int, float] = {}
    completed_values: List[float] = []
    max_round = max((int(r) for r in rounds))
    for task_id in range(1, int(incre_tasks) + 1):
        start_round = _task_start_round(task_id, task_global_round, first_exposure_offset)
        end_round = start_round + int(window) - 1
        if end_round > max_round:
            break
        needed_rounds = [start_round + idx for idx in range(int(window))]
        if any((r not in round_to_acc for r in needed_rounds)):
            continue
        xs = np.asarray(needed_rounds, dtype=np.float64)
        ys = np.asarray([round_to_acc[r] for r in needed_rounds], dtype=np.float64)
        auc_value = float(np.trapz(ys, xs))
        task_auc_map[int(task_id)] = auc_value
        completed_values.append(auc_value)
        progress_map[int(end_round)] = float(np.mean(np.asarray(completed_values, dtype=np.float64)))
    final_auc = 0.0 if len(task_auc_map) == 0 else float(np.mean(np.asarray(list(task_auc_map.values()), dtype=np.float64)))
    return (final_auc, task_auc_map, progress_map)

def _summary_row_key(row: Dict[str, object]) -> Tuple[str, str, str, str, str, str]:
    return (str(row.get('run_tag', '')), str(row.get('dataset', '')), str(row.get('method', '')), str(row.get('exp_type', '')), str(row.get('variant', '')), str(row.get('seeds', '')))

def append_summary_row(summary_csv: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    payload = {key: row.get(key, '') for key in SUMMARY_FIELDNAMES}
    lock_path = summary_csv + '.lock'
    with open(lock_path, 'w', encoding='utf-8') as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        existing_rows: List[Dict[str, object]] = []
        if os.path.exists(summary_csv) and os.path.getsize(summary_csv) > 0:
            with open(summary_csv, 'r', newline='', encoding='utf-8') as f:
                existing_rows = list(csv.DictReader(f))
        row_key = _summary_row_key(payload)
        replaced = False
        for idx, old_row in enumerate(existing_rows):
            if _summary_row_key(old_row) == row_key:
                existing_rows[idx] = payload
                replaced = True
                break
        if not replaced:
            existing_rows.append(payload)
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
            writer.writeheader()
            for item in existing_rows:
                writer.writerow({key: item.get(key, '') for key in SUMMARY_FIELDNAMES})
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

def write_standardized_outputs(*, ctx: Dict[str, str], dataset: str, method: str, exp_type: str, variant: str, seed: int, legacy_acc_global_csv: str, legacy_acc_taskwise_csv: str, incre_tasks: int, task_global_round: int, exposure_rounds: Sequence[int], new_client_num: int, auc_window: int=4, overhead_metrics: Optional[Dict[str, object]]=None, summary_extras: Optional[Dict[str, object]]=None) -> Dict[str, object]:
    if not os.path.exists(legacy_acc_global_csv):
        raise FileNotFoundError(f'Missing legacy acc_global CSV: {legacy_acc_global_csv}')
    if not os.path.exists(legacy_acc_taskwise_csv):
        raise FileNotFoundError(f'Missing legacy acc_taskwise CSV: {legacy_acc_taskwise_csv}')
    with open(legacy_acc_global_csv, 'r', newline='', encoding='utf-8') as f:
        global_rows = list(csv.DictReader(f))
    with open(legacy_acc_taskwise_csv, 'r', newline='', encoding='utf-8') as f:
        taskwise_rows = list(csv.DictReader(f))
    if len(global_rows) != len(taskwise_rows):
        min_len = min(len(global_rows), len(taskwise_rows))
        global_rows = global_rows[:min_len]
        taskwise_rows = taskwise_rows[:min_len]
    rounds: List[int] = []
    acc_new_current: List[float] = []
    combined_rows: List[Dict[str, object]] = []
    for g_row, t_row in zip(global_rows, taskwise_rows):
        round_id = int(safe_float(g_row.get('round'), default=0))
        rounds.append(round_id)
        cur_new = safe_float(g_row.get('acc_new'), default=0.0)
        acc_new_current.append(cur_new)
        combined_rows.append({'round': round_id, 'acc_all': safe_float(g_row.get('acc_all'), default=0.0), 'acc_old': safe_float(g_row.get('acc_old'), default=0.0), 'acc_new_current': cur_new, 'acc_new_avg': compute_acc_new_avg_from_taskwise_row(t_row)})
    auc_new_mean, _, progress_map = compute_auc_new_progress(rounds=rounds, acc_new_current=acc_new_current, incre_tasks=incre_tasks, task_global_round=task_global_round, exposure_rounds=exposure_rounds, new_client_num=new_client_num, window=int(auc_window))
    running_auc = 0.0
    for row in combined_rows:
        round_id = int(row['round'])
        if round_id in progress_map:
            running_auc = float(progress_map[round_id])
        row['auc_new_so_far'] = running_auc
    per_round_fieldnames = ['round', 'dataset', 'method', 'exp_type', 'variant', 'seed', 'acc_all', 'acc_old', 'acc_new_current', 'acc_new_avg', 'auc_new_so_far']
    with open(ctx['per_round_csv'], 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=per_round_fieldnames)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow({'round': int(row['round']), 'dataset': str(dataset), 'method': str(method), 'exp_type': str(exp_type), 'variant': str(variant), 'seed': int(seed), 'acc_all': f'{float(row['acc_all']):.6f}', 'acc_old': f'{float(row['acc_old']):.6f}', 'acc_new_current': f'{float(row['acc_new_current']):.6f}', 'acc_new_avg': f'{float(row['acc_new_avg']):.6f}', 'auc_new_so_far': f'{float(row['auc_new_so_far']):.6f}'})
    final_row = combined_rows[-1] if combined_rows else {'acc_all': 0.0, 'acc_old': 0.0, 'acc_new_avg': 0.0}
    summary_row = {'run_tag': ctx['run_tag'], 'dataset': str(dataset), 'method': str(method), 'exp_type': str(exp_type), 'variant': str(variant), 'auc_window': int(auc_window), 'ACC_all': f'{float(final_row['acc_all']):.6f}', 'ACC_old': f'{float(final_row['acc_old']):.6f}', 'ACC_new(avg)': f'{float(final_row['acc_new_avg']):.6f}', 'AUC_new': f'{float(auc_new_mean):.6f}', 'seed_count': 1, 'seeds': str(int(seed)), 'acc_all_mean': f'{float(final_row['acc_all']):.6f}', 'acc_all_std': f'{0.0:.6f}', 'acc_old_mean': f'{float(final_row['acc_old']):.6f}', 'acc_old_std': f'{0.0:.6f}', 'acc_new_avg_mean': f'{float(final_row['acc_new_avg']):.6f}', 'acc_new_avg_std': f'{0.0:.6f}', 'auc_new_mean': f'{float(auc_new_mean):.6f}', 'auc_new_std': f'{0.0:.6f}', 'per_round_csv': ctx['per_round_csv'], 'legacy_acc_global_csv': legacy_acc_global_csv, 'legacy_acc_taskwise_csv': legacy_acc_taskwise_csv}
    if overhead_metrics is not None:
        for key in ['comm_model_only', 'comm_extra_proto_meta', 'comm_total', 'comm_ratio_vs_fedavg', 'server_time_mean', 'server_time_ratio_vs_fedavg']:
            if key in overhead_metrics:
                summary_row[key] = overhead_metrics[key]
    if summary_extras is not None:
        for key, value in summary_extras.items():
            summary_row[key] = value
    append_summary_row(ctx['summary_csv'], summary_row)
    return {'per_round_csv': ctx['per_round_csv'], 'summary_csv': ctx['summary_csv'], 'auc_new': float(auc_new_mean), 'acc_new_avg': float(final_row['acc_new_avg'])}

def estimate_state_dict_nbytes(state_dict: Dict[str, torch.Tensor]) -> int:
    total = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total += int(value.numel()) * int(value.element_size())
    return int(total)

def estimate_model_upload_nbytes(models: Sequence[Dict[str, torch.Tensor]]) -> int:
    return int(sum((estimate_state_dict_nbytes(model) for model in models)))

def estimate_proto_meta_upload_nbytes(client_prototypes: Sequence[Dict[int, np.ndarray]], client_weights: Sequence[Dict[int, int]], target_labels: Optional[Sequence[int]]=None, include_support_flags: bool=True) -> Tuple[int, int]:
    labels_filter = None if target_labels is None else {int(x) for x in target_labels}
    proto_bytes = 0
    meta_bytes = 0
    for cid, proto_dict in enumerate(client_prototypes):
        if include_support_flags:
            meta_bytes += 1
        for label, proto in proto_dict.items():
            label_i = int(label)
            if labels_filter is not None and label_i not in labels_filter:
                continue
            arr = np.asarray(proto, dtype=np.float32)
            proto_bytes += int(arr.size) * 4
            if cid < len(client_weights):
                if label_i in client_weights[cid]:
                    meta_bytes += 8
    return (int(proto_bytes), int(meta_bytes))

def build_noisy_uploaded_prototypes(client_prototypes: Sequence[Dict[int, np.ndarray]], sigma: float, seed: int, global_round: int, target_labels: Optional[Sequence[int]]=None) -> Sequence[Dict[int, np.ndarray]]:
    sigma = float(sigma)
    if sigma <= 0.0:
        return client_prototypes
    labels_filter = None if target_labels is None else {int(x) for x in target_labels}
    noisy_prototypes: List[Dict[int, np.ndarray]] = []
    for cid, proto_dict in enumerate(client_prototypes):
        copied = dict(proto_dict)
        for label, proto in proto_dict.items():
            label_i = int(label)
            if labels_filter is not None and label_i not in labels_filter:
                continue
            proto_arr = np.asarray(proto, dtype=np.float32).copy()
            noise_seed = int(seed) + 10007 * int(global_round + 1) + 101 * int(cid) + int(label_i)
            rng = np.random.default_rng(noise_seed)
            proto_arr = proto_arr + rng.normal(loc=0.0, scale=sigma, size=proto_arr.shape).astype(np.float32)
            norm = float(np.linalg.norm(proto_arr))
            if norm <= 1e-12:
                proto_arr = np.asarray(proto, dtype=np.float32).copy()
                norm = float(np.linalg.norm(proto_arr))
            proto_arr = proto_arr / max(norm, 1e-12)
            copied[label_i] = proto_arr
        noisy_prototypes.append(copied)
    return noisy_prototypes
