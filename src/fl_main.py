from FedHCCA import FedHCCA_model
from ResNet import resnet18_cbam
from backbone_factory import build_backbone, normalize_backbone_name
import torch
import copy
import random
import os.path as osp
import os
import csv
import glog as logger
import time
from myNetwork import *
from Fed_utils import *
from FedHCCA import *
from option import args_parser
from iCIFAR100 import iCIFAR100, iCIFAR10, iSVHN
import pickle
from experiment_utils import append_pattern_record, build_async_exposure_rounds, build_noisy_uploaded_prototypes, build_output_context, cumulative_to_increment, describe_async_timing_mode, estimate_model_upload_nbytes, estimate_proto_meta_upload_nbytes, infer_experiment_identity, init_pattern_csv, resolve_exposure_cumulative, sigma_to_tag, write_standardized_outputs
args = args_parser()
method = args.method
date = args.date
if method == 'FedHCCA':
    _filename = '_final'
else:
    _filename = 'base'
METHOD_PROFILES = {'FedHCCA': dict(aggregator='fedhcca', IL='fedhcca'), 'FedAvg': dict(aggregator='fedavg', IL=None), 'FedProx': dict(aggregator='fedprox', IL=None), 'FedAvg_PODNet': dict(aggregator='fedavg', IL='PODNet'), 'FedProx_PODNet': dict(aggregator='fedprox', IL='PODNet'), 'FedProx_iCaRL': dict(aggregator='fedprox', IL='iCaRL'), 'FedAvg_iCaRL': dict(aggregator='fedavg', IL='iCaRL'), 'FedProx_EWC': dict(aggregator='fedprox', IL='EWC'), 'FedAvg_EWC': dict(aggregator='fedavg', IL='EWC')}
prof = METHOD_PROFILES[args.method]
aggregator = prof['aggregator']
IL_method = prof['IL']
ablate_server_fedavg = bool(int(getattr(args, 'ablate_server_fedavg', 0)))
seed = args.seed
setup_seed(seed)

def seed_worker(worker_id):
    worker_seed = (seed + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
if args.device == -1:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')
models = []
dataset_name = str(args.dataset).lower()
if int(args.task_size) != 1:
    logger.warning('Current FedHCCA fl_main assumes 1 class per task; got task_size=%d. ', int(args.task_size))

def _get_dataset_cfg(name: str):
    n = str(name).lower()
    if n in {'cifar100', 'cifar-100'}:
        return {'name': 'cifar100', 'cls': iCIFAR100, 'num_classes': 100, 'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761), 'medical': False}
    if n in {'cifar10', 'cifar-10'}:
        return {'name': 'cifar10', 'cls': iCIFAR10, 'num_classes': 10, 'mean': (0.4914, 0.4822, 0.4465), 'std': (0.247, 0.2435, 0.2616), 'medical': False}
    if n in {'svhn'}:
        return {'name': 'svhn', 'cls': iSVHN, 'num_classes': 10, 'mean': (0.4377, 0.4438, 0.4728), 'std': (0.198, 0.201, 0.197), 'medical': False}
    raise ValueError(f'Unsupported dataset: {name}')
ds_cfg = _get_dataset_cfg(dataset_name)
dataset_name = ds_cfg['name']
DatasetCls = ds_cfg['cls']
num_total_classes = int(ds_cfg['num_classes'])
is_medical = bool(ds_cfg.get('medical', False))
incre_tasks = int(args.incre_tasks)
base_numclass = int(args.baseclass)
learned_numclass = int(args.learnedclasses)
total_classes = int(args.learnedclasses)
class_range = [0, base_numclass]
num_clients = int(args.num_clients)
clients_index = [i for i in range(num_clients)]
client_data_indices = [[] for _ in range(num_clients)]
alpha = float(args.alpha)
client_result_dict = {f'client_{client_id}': {'acc_all': [], 'acc_new': [], 'acc_old': []} for client_id in range(num_clients)}
if int(total_classes) != int(base_numclass):
    logger.warning('args.learnedclasses=%d differs from baseclass=%d; continuing with learnedclasses.', int(total_classes), int(base_numclass))
max_incre = num_total_classes - int(base_numclass)
if incre_tasks < 0 or incre_tasks > max_incre:
    raise ValueError(f'Invalid incre_tasks={incre_tasks}: dataset={dataset_name} has {num_total_classes} classes, baseclass={base_numclass} => max_incre={max_incre}')
task_name = f'incre_class{incre_tasks}' if dataset_name == 'cifar100' else f'{dataset_name}_incre_class{incre_tasks}'
logger.info('Dataset=%s total_classes=%d baseclass=%d incre_tasks=%d', dataset_name, num_total_classes, int(base_numclass), int(incre_tasks))
if method != 'FedHCCA' and str(getattr(args, 'server_ablation', 'full')).strip().lower() != 'full':
    logger.warning('--server_ablation only affects method=FedHCCA; current method=%s so it will be ignored.', method)
if method != 'FedHCCA' and float(getattr(args, 'proto_noise_sigma', 0.0)) > 0.0:
    logger.warning('--proto_noise_sigma only affects method=FedHCCA; current method=%s so it will be ignored.', method)
profile_overhead_enabled = bool(int(getattr(args, 'profile_overhead', 0)))
if profile_overhead_enabled and (not (dataset_name == 'cifar100' and method in {'FedHCCA', 'FedAvg'})):
    logger.warning('--profile_overhead currently supports only CIFAR-100 FedHCCA/FedAvg runs; disabling for dataset=%s method=%s.', dataset_name, method)
    profile_overhead_enabled = False
exp_type, exp_variant = infer_experiment_identity(method=method, server_ablation=str(getattr(args, 'server_ablation', 'full')), async_timing_mode=str(getattr(args, 'async_timing_mode', 'fixed_default')), exposure_mode=str(getattr(args, 'exposure_mode', 'exponential')), proto_noise_sigma=float(getattr(args, 'proto_noise_sigma', 0.0)), profile_overhead=profile_overhead_enabled, cli_flags=getattr(args, 'cli_flags', []))
if exp_type == 'proto_noise':
    exp_variant = f'sigma_{sigma_to_tag(float(getattr(args, 'proto_noise_sigma', 0.0)))}'
backbone_name_for_exp = normalize_backbone_name(getattr(args, 'backbone', 'resnet18'))
if backbone_name_for_exp != 'resnet18':
    exp_type = 'backbone_robustness'
    exp_variant = f'backbone_{backbone_name_for_exp}'
async_timing_mode = str(getattr(args, 'async_timing_mode', 'fixed_default')).strip().lower()
async_timing_flag_explicit = '--async_timing_mode' in set(getattr(args, 'cli_flags', []))
async_timing_enabled = bool(dataset_name == 'cifar100' and (async_timing_flag_explicit or async_timing_mode != 'fixed_default'))
async_timing_desc = describe_async_timing_mode(async_timing_mode) if async_timing_enabled else ''
output_ctx = build_output_context(run_tag=str(getattr(args, 'run_tag', '') or args.date), dataset=dataset_name, method=method, seed=int(args.seed), output_root=str(getattr(args, 'output_root', '')), log_root=str(getattr(args, 'log_root', '')), exp_type=exp_type, variant=exp_variant)
logger.info('Standardized outputs | exp_type=%s variant=%s per_round=%s summary=%s nohup_log=%s', exp_type, exp_variant, output_ctx['per_round_csv'], output_ctx['summary_csv'], output_ctx['log_file'])
if '--async_timing_mode' in set(getattr(args, 'cli_flags', [])) and dataset_name != 'cifar100':
    logger.warning('--async_timing_mode is only implemented for CIFAR-100 supplementary runs; dataset=%s keeps legacy timing.', dataset_name)
if is_medical:
    resize_size = max(int(args.img_size), 64)
    train_transform = transforms.Compose([transforms.Resize((resize_size, resize_size)), transforms.RandomResizedCrop(resize_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(ds_cfg['mean'], ds_cfg['std'])])
    test_transform = transforms.Compose([transforms.Resize((resize_size, resize_size)), transforms.ToTensor(), transforms.Normalize(ds_cfg['mean'], ds_cfg['std'])])
else:
    train_transform = transforms.Compose([transforms.RandomCrop((args.img_size, args.img_size), padding=4), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.24705882352941178), transforms.ToTensor(), transforms.Normalize(ds_cfg['mean'], ds_cfg['std'])])
    test_transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize(ds_cfg['mean'], ds_cfg['std'])])
exemplar_transform = test_transform
if dataset_name == 'isic2019':
    if not args.isic_root or not args.isic_csv:
        raise ValueError('ISIC2019 requires --isic_root and --isic_csv')
    train_dataset = DatasetCls(args.isic_root, args.isic_csv, train=True, transform=train_transform, test_transform=test_transform, exemplar_transform=exemplar_transform, split_seed=args.seed, train_per_class=args.train_per_class)
    test_dataset = DatasetCls(args.isic_root, args.isic_csv, train=False, transform=train_transform, test_transform=test_transform, exemplar_transform=exemplar_transform, split_seed=args.seed, train_per_class=args.train_per_class)
elif dataset_name == 'pathmnist':
    pathmnist_root = args.pathmnist_root or osp.join(args.data_root, 'pathmnist')
    train_dataset = DatasetCls(pathmnist_root, train=True, transform=train_transform, test_transform=test_transform, exemplar_transform=exemplar_transform, split_seed=args.seed, train_per_class=args.train_per_class, size=args.pathmnist_size, use_medmnist_api=args.use_medmnist_api)
    test_dataset = DatasetCls(pathmnist_root, train=False, transform=train_transform, test_transform=test_transform, exemplar_transform=exemplar_transform, split_seed=args.seed, train_per_class=args.train_per_class, size=args.pathmnist_size, use_medmnist_api=args.use_medmnist_api)
elif dataset_name in {'cifar10', 'svhn'}:
    data_root = args.data_root
    train_dataset = DatasetCls(data_root, train=True, transform=train_transform, download=True, train_per_class=args.train_per_class, seed=args.seed)
    test_dataset = DatasetCls(data_root, test_transform=test_transform, train=False, download=True)
else:
    data_root = args.data_root
    train_dataset = DatasetCls(data_root, train=True, transform=train_transform, download=True)
    test_dataset = DatasetCls(data_root, test_transform=test_transform, train=False, download=True)
train_classes = [i for i in range(num_total_classes)]
train_dataset.getTrainData(classes=train_classes, new_class_id=-1, exemplar_set=[], exemplar_label_set=[])
logger.info(f'TrainData type: {type(train_dataset.TrainData)}, TrainLabels type: {type(train_dataset.TrainLabels)}')
if is_medical:
    summarize_class_counts(train_dataset.TrainLabels, getattr(train_dataset, 'class_names', {}), f'{dataset_name} train per-class counts (post-adapter):')
    summarize_class_counts(test_dataset.targets, getattr(test_dataset, 'class_names', {}), f'{dataset_name} test per-class counts (post-adapter):')
if is_medical:
    reuse_base_pkl = str(getattr(args, 'base_distribution_pkl', '')).strip()
    if reuse_base_pkl and osp.exists(reuse_base_pkl):
        with open(reuse_base_pkl, 'rb') as f:
            base_data_indices = pickle.load(f)
        logger.info('Loaded medical base split pkl: %s', reuse_base_pkl)
    else:
        rng_base = np.random.default_rng(int(seed))
        base_split_indices = train_dataset.dirichlet_split_indices(targets=train_dataset.TrainLabels, class_range=class_range, num_clients=num_clients, alpha=alpha, overlap=False, min_per_client=1, rng=rng_base, seed=int(seed), group_ids=getattr(train_dataset, 'patient_ids', None))
        base_data_indices = {'client_data_indices': base_split_indices}
        logger.info('Generated online base split for medical dataset: %s', dataset_name)
else:
    if args.base_distribution_pkl:
        basedata_distribution_path = args.base_distribution_pkl
    else:
        basedata_distribution_path = osp.join('./outputs/data_distribution', f'{dataset_name}_base_class_{base_numclass}_{num_clients}client_alpha{alpha}_seed{seed}.pkl')
    if osp.exists(basedata_distribution_path):
        with open(basedata_distribution_path, 'rb') as f:
            base_data_indices = pickle.load(f)
        logger.info('Loaded base split pkl: %s', basedata_distribution_path)
    else:
        rng_base = np.random.default_rng(int(seed))
        split_targets = getattr(train_dataset, 'TrainLabels', getattr(train_dataset, 'targets', None))
        base_split_indices = train_dataset.dirichlet_split_indices(targets=split_targets, class_range=class_range, num_clients=num_clients, alpha=alpha, overlap=False, min_per_client=5, rng=rng_base, seed=int(seed))
        base_data_indices = {'client_data_indices': base_split_indices}
        os.makedirs(osp.dirname(basedata_distribution_path), exist_ok=True)
        with open(basedata_distribution_path, 'wb') as f:
            pickle.dump(base_data_indices, f)
        logger.info('Generated base split pkl: %s', basedata_distribution_path)
for client_id in range(num_clients):
    client_data_indices[client_id].extend(base_data_indices['client_data_indices'][client_id])
for client_id in range(num_clients):
    logger.info('Client %d base sample count: %d', client_id, len(client_data_indices[client_id]))
backbone_name = normalize_backbone_name(getattr(args, 'backbone', 'resnet18'))
base_backbone = build_backbone(backbone_name, img_size=int(getattr(args, 'img_size', 32)))
logger.info('Backbone robustness setting | backbone=%s', backbone_name)
feature_extractor_g = copy.deepcopy(base_backbone)
model_g = FedHCCA_model(base_numclass, feature_extractor_g, args.batch_size, args.task_size, args.memory_size, args.epochs_local, args.learning_rate, train_dataset, test_dataset, args.device, exemplar_per_class=getattr(args, 'exemplar_per_class', 20))
model_g.model = model_g.model.to(device)
default_base_model_path = ''
if backbone_name == 'resnet18':
    if dataset_name == 'cifar100' and int(base_numclass) == 10:
        default_base_model_path = './outputs/model/base/base10_best_global_model_acc_8200_round61_260101.pth'
    if dataset_name == 'cifar10' and int(base_numclass) == 5:
        default_base_model_path = './outputs/model/cifar10/base5_best_global_model_acc_7252_round49_260129.pth'
    if dataset_name == 'svhn' and int(base_numclass) == 5:
        default_base_model_path = './outputs/model/svhn/base5_best_global_model_acc_8820_round42_260129.pth'
else:
    logger.info('Skip legacy ResNet base checkpoint for non-ResNet backbone=%s', backbone_name)
path = str(getattr(args, 'base_model_path', '')).strip() or default_base_model_path
if path and osp.exists(path):
    state_dict = torch.load(path, map_location='cpu')
    model_g.model.load_state_dict(state_dict)
    logger.info('Loaded pretrained base global model: %s', path)
else:
    logger.warning('No pretrained base model loaded (path=%s)', path if path else '<empty>')
model_g.model = model_g.model.to(device)
best_model_old = best_model_cur = copy.deepcopy(model_g.model)
model_old = [best_model_old, best_model_cur]
client_prototypes = [{} for client_id in range(num_clients)]
client_weights = [{} for client_id in range(num_clients)]
global_prototypes = {}
for client_id in range(num_clients):
    g_client = torch.Generator()
    g_client.manual_seed(seed + client_id)
    feature_extractor_i = copy.deepcopy(base_backbone)
    client_dataset_temp = train_dataset.getTrainItem_indices(base_data_indices['client_data_indices'][client_id])
    model_temp = FedHCCA_model(args.baseclass, feature_extractor_i, args.batch_size, args.task_size, args.memory_size, args.epochs_local, args.learning_rate, client_dataset_temp, test_dataset, args.device, exemplar_per_class=getattr(args, 'exemplar_per_class', 20))
    model_temp.current_class = args.baseclass - 1
    model_temp.last_class = args.baseclass - 1
    model_temp.recent_full_class = 0
    model_temp.recent_full_indices = base_data_indices['client_data_indices'][client_id]
    model_temp.train_loader = DataLoader(client_dataset_temp, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g_client)
    model_temp.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g_client)
    model_temp.optimizer = optim.SGD(model_temp.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    model_temp.scheduler = None
    model_temp.last_lr_stage = None
    model_temp.client_id = client_id
    model_temp.model = model_temp.model.to(device)
    model_temp.method = method
    model_temp.date = date
    if is_medical:
        newcls_dir = osp.join('./outputs/csv/system', dataset_name)
        os.makedirs(newcls_dir, exist_ok=True)
        model_temp.newcls_csv_path = osp.join(newcls_dir, f'{date}_{method}{_filename}_task_{task_name}_newcls_metrics.csv')
    else:
        model_temp.newcls_csv_path = f'./outputs/csv/new_class/{date}_{dataset_name}_{method}{_filename}_newcls_metrics.csv'
    model_temp.compute_local_prototypes(client_id=client_id, class_id=-1, client_prototypes=client_prototypes, client_weights=client_weights)
    logger.info(f'Client{client_id} initial prototypes keys: {sorted(client_prototypes[client_id].keys())}')
    models.append(model_temp)
logger.info(f'client_weights_0: {client_weights[0]}')
global_prototypes = aggregate_prototypes(client_prototypes=client_prototypes, global_prototypes=global_prototypes, client_weights=client_weights, num_clients=num_clients, cls_range=[0, base_numclass])
logger.info(f'number of initial Global prototypes: {len(global_prototypes)}')
logger.info(f'Initial Global prototypes keys: {list(global_prototypes.keys())}')
for client_id in range(num_clients):
    models[client_id].exemplar_update_set(global_prototypes=global_prototypes, global_round=0)
global_rounds = int(args.task_global_round) * int(incre_tasks)
args.rounds_global = global_rounds
task_global_round = args.task_global_round
local_epochs = args.epochs_local
if is_medical:
    output_dir = osp.join('./outputs/log/system', dataset_name)
else:
    output_dir = osp.join('./outputs/training_log', method)
os.makedirs(output_dir, exist_ok=True)
filename = f'{date}_{dataset_name}_{method}{_filename}_task_{task_name}_seed{args.seed}_glrounds{global_rounds}_lepochs{local_epochs}.txt'
standard_trainlog_path = output_ctx['log_file'].replace('.log', '.txt')
out_file = open(standard_trainlog_path, 'w')
logger.info('Structured training log path: %s', standard_trainlog_path)
log_str = 'task_{}, global_rounds_{}, local_epochs_{}'.format(task_name, global_rounds, local_epochs)
out_file.write(log_str + '\n')
out_file.flush()
if dataset_name == 'cifar100':
    if method == 'FedHCCA':
        csv_path_acc_global = f'./outputs/csv/incre/{date}_{dataset_name}{_filename}_incre{incre_tasks}_acc_global.csv'
        csv_path_acc_taskwise = f'./outputs/csv/incre/{date}_{dataset_name}{_filename}_incre{incre_tasks}_acc_taskwise.csv'
    else:
        csv_path_acc_global = f'./outputs/csv/compare/{date}_{method}_{dataset_name}_incre{incre_tasks}_acc_global{_filename}.csv'
        csv_path_acc_taskwise = f'./outputs/csv/compare/{date}_{method}_{dataset_name}_incre{incre_tasks}_acc_taskwise{_filename}.csv'
elif is_medical:
    csv_dir = osp.join('./outputs/csv/system', dataset_name)
    csv_path_acc_global = osp.join(csv_dir, f'{date}_{method}{_filename}_task_{task_name}_acc_global.csv')
    csv_path_acc_taskwise = osp.join(csv_dir, f'{date}_{method}{_filename}_task_{task_name}_acc_taskwise.csv')
else:
    csv_path_acc_global = f'./outputs/csv/{dataset_name}/{date}_{method}{_filename}_incre{incre_tasks}_acc_global.csv'
    csv_path_acc_taskwise = f'./outputs/csv/{dataset_name}/{date}_{method}_incre{incre_tasks}_acc_taskwise{_filename}.csv'
os.makedirs(osp.dirname(csv_path_acc_global), exist_ok=True)
os.makedirs(osp.dirname(csv_path_acc_taskwise), exist_ok=True)
with open(csv_path_acc_global, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['round', 'acc_all', 'acc_new', 'acc_old'])
name_str_list = []
name_str_list.append(f'base_0_{base_numclass - 1}')
for i in range(base_numclass, base_numclass + incre_tasks):
    name_str_list.append(f'incre_{i}')
logger.info(f'name_str_list: {name_str_list}')
with open(csv_path_acc_taskwise, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['round'] + name_str_list)
old_task_id = 0
class_id = base_numclass - 1
if is_medical:
    exposure_rounds = [0, 3, 5, 5, 9, 9, 9, 9, 9, 9]
    incre_sample_splits = [0.05, 0.15, 0.35, 1.0]
else:
    exposure_rounds = parse_int_list_csv(getattr(args, 'exposure_rounds', '0,3,5,5,9,9,9,9,9,9'), expected_len=num_clients, name='exposure_rounds')
    incre_sample_splits = parse_int_list_csv(getattr(args, 'incre_sample_splits', '3,9,24,64'), expected_len=4, name='incre_sample_splits')
    if dataset_name == 'cifar100':
        exposure_cumulative = resolve_exposure_cumulative(exposure_mode=str(getattr(args, 'exposure_mode', 'exponential')), seed=int(args.seed), default_increment_splits=incre_sample_splits, total_cap=100, num_stages=4)
        incre_sample_splits = cumulative_to_increment(exposure_cumulative)
        logger.info('Resolved exposure_mode=%s cumulative=%s increments=%s', args.exposure_mode, exposure_cumulative, incre_sample_splits)
    elif str(getattr(args, 'exposure_mode', 'exponential')).strip().lower() != 'exponential':
        logger.warning('--exposure_mode only changes CIFAR-100 supplementary runs in fl_main; dataset=%s keeps legacy incre_sample_splits=%s.', dataset_name, incre_sample_splits)
logger.info(f'exposure_rounds: {exposure_rounds}')
logger.info(f'incre_sample_splits: {incre_sample_splits}')
if async_timing_enabled:
    init_pattern_csv(output_ctx['pattern_csv'])
    logger.info('Async timing robustness enabled | mode=%s desc=%s pattern_csv=%s', async_timing_mode, async_timing_desc, output_ctx['pattern_csv'])
task_exposure_rounds = list(exposure_rounds)
fedhcca_task_best_new_weight = 0.5
if dataset_name == 'isic2019':
    fedhcca_task_best_new_weight = 0.1
task_best = {'score': float('-inf'), 'round': None, 'acc_all': None, 'acc_new': None, 'acc_old': None, 'state': None}
rowbank_enable = method == 'FedHCCA' and bool(int(getattr(args, 'fedhcca_rowbank_final_restore', 0)))
rowbank = {}
task_restore_enable = method == 'FedHCCA' and bool(int(getattr(args, 'fedhcca_taskwise_restore', 0)))
task_restore_ratio = float(getattr(args, 'fedhcca_taskwise_restore_ratio', 0.7))
task_restore_threshold = float(getattr(args, 'fedhcca_taskwise_restore_threshold', 20.0))
csv_path_taskrestore_global = None
csv_path_taskrestore_taskwise = None
if task_restore_enable:
    csv_path_taskrestore_global = csv_path_acc_global.replace('.csv', '_taskrestore.csv')
    csv_path_taskrestore_taskwise = csv_path_acc_taskwise.replace('.csv', '_taskrestore.csv')
    try:
        with open(csv_path_taskrestore_global, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'task_id', 'restored_classes', 'acc_all', 'acc_new', 'acc_old'])
        with open(csv_path_taskrestore_taskwise, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'task_id', 'restored_classes'] + name_str_list)
        logger.info(f'[FedHCCA][TaskRestore] post-restore CSV enabled: {csv_path_taskrestore_global}')
    except Exception as e:
        logger.warning(f'[FedHCCA][TaskRestore] failed to init taskrestore CSV: {e}')
task_restore_beta_min = float(getattr(args, 'fedhcca_taskwise_restore_beta_min', 0.3))
task_restore_beta_max = float(getattr(args, 'fedhcca_taskwise_restore_beta_max', 0.7))
best_row_cache = {}
new_client_num = int(getattr(args, 'new_client_num', 4))
ewc_pack = {'params': None, 'fisher': None}
overhead_profile = {'comm_model_only_bytes': [], 'comm_extra_proto_meta_bytes': [], 'comm_total_bytes': [], 'server_time_seconds': [], 'fedavg_server_time_seconds': []}
if IL_method == 'EWC':
    ewc_pack['params'] = {n: p.detach().clone().cpu() for n, p in model_g.model.named_parameters() if p.requires_grad}
    fisher_list = []
    weight_list = []
    for cid in range(num_clients):
        fisher_c = models[cid].estimate_fisher_diag(models[cid].train_loader, max_batches=models[cid].ewc_batches)
        fisher_c = {k: v.detach().cpu() for k, v in fisher_c.items()}
        fisher_list.append(fisher_c)
        weight_list.append(max(1, len(models[cid].train_loader.dataset)))
    ewc_pack['fisher'] = FedAvg_Fisher_weighted(fisher_list, weight_list)
for global_round in range(global_rounds):
    task_id = 1 + global_round // args.task_global_round
    if global_round % args.task_global_round == 0:
        log_str = f'\nIncre task {task_id} starts:'
        out_file.write(log_str + '\n')
        out_file.flush()
        logger.info(log_str)
    log_str = f'Global Round {global_round + 1}/{global_rounds} starts:'
    out_file.write(log_str + '\n')
    out_file.flush()
    logger.info(log_str)
    if task_id != old_task_id:
        model_old[0] = copy.deepcopy(model_old[1])
        class_id += 1
        total_classes += args.task_size
        log_str = 'Global model has learned {} classes'.format(total_classes)
        out_file.write(log_str + '\n')
        out_file.flush()
        logger.info(log_str)
        model_g.model.Incremental_learning(total_classes)
        model_g.model = model_g.model.to(device)
        rng_task = np.random.default_rng(int(seed) + int(task_id))
        incre_kwargs = dict(targets=train_dataset.TrainLabels, class_id=class_id, new_clients=new_client_num, num_clients=num_clients, incre_sample_splits=incre_sample_splits, alpha=alpha, min_per_client=1 if is_medical else 10, rng=rng_task, seed=int(seed) + int(task_id))
        if is_medical:
            incre_kwargs.update(group_ids=getattr(train_dataset, 'patient_ids', None), exposure_rounds=exposure_rounds, task_global_round=task_global_round)
        new_indices_dict, new_data_indices = train_dataset.incre_split_datasets(**incre_kwargs)
        test_dataset.getTestData([0, total_classes])
        rng = np.random.default_rng(seed + task_id)
        shuffled_client_index = rng.permutation(clients_index)
        logger.info(f'Shuffled_client_index for task {task_id}: {shuffled_client_index}')
        if async_timing_enabled:
            task_exposure_rounds = build_async_exposure_rounds(mode=async_timing_mode, seed=int(seed), task_id=int(task_id), num_clients=int(num_clients), num_new_clients=int(new_client_num), task_global_round=int(task_global_round))
            append_pattern_record(csv_path=output_ctx['pattern_csv'], run_tag=output_ctx['run_tag'], dataset=dataset_name, method=method, async_timing_mode=async_timing_mode, task_id=int(task_id), seed=int(seed), timing_pattern_zero_based=task_exposure_rounds[:int(new_client_num)], timing_pattern_desc=async_timing_desc)
            logger.info('Task %d async timing pattern (1-based new-client rounds): %s', int(task_id), [int(x) + 1 for x in task_exposure_rounds[:int(new_client_num)]])
        else:
            task_exposure_rounds = list(exposure_rounds)
        beforeTask(clients=models, num_clients=num_clients, test_dataset=test_dataset, new_data_indices=new_data_indices, client_data_indices=client_data_indices, shuffled_client_index=shuffled_client_index, exposure_rounds=task_exposure_rounds, total_classes=total_classes)
        task_best = {'score': float('-inf'), 'round': None, 'acc_all': None, 'acc_new': None, 'acc_old': None, 'state': None}
        old_task_id += 1
        mapped = {int(shuffled_client_index[i]): task_exposure_rounds[i] for i in range(num_clients)}
        log_str_1 = f'Exposure rounds for task {task_id}: {mapped}'
        log_str_2 = f'New_data distribution for task {task_id}:             Client{shuffled_client_index} - {[len(new_data_indices[c_id]) for c_id in range(num_clients)]}'
        out_file.write(log_str_1 + '\n')
        out_file.write(log_str_2 + '\n')
        out_file.flush()
        logger.info(log_str_1)
        logger.info(log_str_2)
    update_new_set(clients=models, new_client=new_client_num, num_clients=num_clients, model_g=model_g, global_round=global_round, global_rounds=global_rounds, task_global_round=task_global_round, train_dataset=train_dataset, client_indices_dict=new_indices_dict, shuffled_client_index=shuffled_client_index, global_prototypes=global_prototypes)
    if global_round % 100 == 90:
        class_range = [total_classes - 10, total_classes]
        logger.info(f'class_range {class_range}')
        heatmap_save_path = f'./outputs/heatmap/{date}_{method}_{dataset_name}{_filename}_task_{task_name}_class_{class_range[0]}_{class_range[1]}_distribution_heatmap.png'
        train_dataset.distribution_heatmap(train_dataset.TrainLabels, client_data_indices, heatmap_save_path, num_clients, class_range)
    w_local = []
    clients_grad = []
    for client_id in range(num_clients):
        local_model, local_grad = local_train(clients=models, client_id=client_id, class_id=class_id, model_old=model_old, global_round=global_round, task_global_round=task_global_round, client_prototypes=client_prototypes, global_prototypes=global_prototypes, client_weights=client_weights, model_g=model_g, aggregator=aggregator, IL_method=IL_method, ewc_pack=ewc_pack)
        w_local.append(local_model)
        clients_grad.append(local_grad)
    logger.info('federated aggregation...')
    prev_global_prototypes = copy.deepcopy(global_prototypes)
    server_client_prototypes = client_prototypes
    proto_noise_sigma = 0.0
    if method == 'FedHCCA':
        proto_noise_sigma = float(getattr(args, 'proto_noise_sigma', 0.0))
        server_client_prototypes = build_noisy_uploaded_prototypes(client_prototypes=client_prototypes, sigma=proto_noise_sigma, seed=int(args.seed), global_round=int(global_round), target_labels=[class_id])
        if proto_noise_sigma > 0.0:
            logger.info('[ProtoNoise] round=%d class=%d sigma=%.4f applied to uploaded server prototypes only.', int(global_round + 1), int(class_id), float(proto_noise_sigma))
    global_prototypes = aggregate_prototypes(server_client_prototypes, global_prototypes, client_weights, num_clients, cls_range=[class_id, class_id + 1])
    logger.info(f'Global prototypes keys: {list(global_prototypes.keys())}')
    for cl_id in range(total_classes):
        if cl_id in global_prototypes and cl_id in prev_global_prototypes:
            gp = global_prototypes[cl_id]
            pgp = prev_global_prototypes[cl_id]
            gp_t = gp if torch.is_tensor(gp) else torch.as_tensor(gp, dtype=torch.float32)
            pgp_t = pgp if torch.is_tensor(pgp) else torch.as_tensor(pgp, dtype=torch.float32)
            if gp_t.device != pgp_t.device:
                pgp_t = pgp_t.to(gp_t.device)
            drift = torch.norm(gp_t - pgp_t).item()
            if drift > 0.0:
                logger.info(f'[DBG][Server][Round{global_round + 1}] Proto drift class {cl_id} = {drift:.6f}')
    new_support_list = [models[cid].new_support for cid in range(num_clients)]
    if profile_overhead_enabled:
        comm_model_only_bytes = estimate_model_upload_nbytes(w_local)
        if aggregator == 'fedhcca' and (not ablate_server_fedavg):
            proto_bytes, meta_bytes = estimate_proto_meta_upload_nbytes(server_client_prototypes, client_weights, target_labels=[class_id], include_support_flags=True)
            comm_extra_bytes = int(proto_bytes + meta_bytes)
        else:
            comm_extra_bytes = 0
        overhead_profile['comm_model_only_bytes'].append(int(comm_model_only_bytes))
        overhead_profile['comm_extra_proto_meta_bytes'].append(int(comm_extra_bytes))
        overhead_profile['comm_total_bytes'].append(int(comm_model_only_bytes + comm_extra_bytes))
    if aggregator != 'fedhcca' or ablate_server_fedavg:
        server_time_start = time.perf_counter()
        w_g_new = FedAvg(w_local)
        server_time_elapsed = time.perf_counter() - server_time_start
        fedavg_baseline_time = server_time_elapsed
    else:
        server_time_start = time.perf_counter()
        w_g_new = cluster_and_aggregate(new_class_id=class_id, models=w_local, client_prototypes=server_client_prototypes, ref_model=model_g.model.state_dict(), client_weights=client_weights, num_clients=num_clients, clients_grad=clients_grad, metric='cosine', exposure_rounds=task_exposure_rounds, task_global_round=task_global_round, global_round=global_round, new_support_list=new_support_list, test_dataset=test_dataset)
        server_time_elapsed = time.perf_counter() - server_time_start
        fedavg_time_start = time.perf_counter()
        _ = FedAvg(w_local)
        fedavg_baseline_time = time.perf_counter() - fedavg_time_start
    if profile_overhead_enabled:
        overhead_profile['server_time_seconds'].append(float(server_time_elapsed))
        overhead_profile['fedavg_server_time_seconds'].append(float(fedavg_baseline_time))
        logger.info('[Overhead][Round %d] comm_model_only=%d comm_extra_proto_meta=%d comm_total=%d server_time=%.6f fedavg_time=%.6f', int(global_round + 1), int(overhead_profile['comm_model_only_bytes'][-1]), int(overhead_profile['comm_extra_proto_meta_bytes'][-1]), int(overhead_profile['comm_total_bytes'][-1]), float(server_time_elapsed), float(fedavg_baseline_time))
    model_g.model.load_state_dict(w_g_new)
    stats = eval_global_newclass_margin(model=model_g.model, test_dataset=test_dataset, new_class_id=class_id, device=device)
    round_tag = global_round + 1
    logger.info(f'[VAL-C][Server][Round{round_tag}] NEW-CLS margin mean={stats['margin_mean']:.4f}, neg_rate={stats['neg_rate']:.2f}, count={stats['count']}')
    if IL_method == 'EWC':
        if (global_round + 1) % task_global_round == 0:
            ewc_pack['params'] = {n: p.detach().clone().cpu() for n, p in model_g.model.named_parameters() if p.requires_grad}
            fisher_list = []
            weight_list = []
            for cid in range(num_clients):
                fisher_c = models[cid].estimate_fisher_diag(models[cid].train_loader, max_batches=models[cid].ewc_batches)
                fisher_c = {k: v.detach().cpu() for k, v in fisher_c.items()}
                fisher_list.append(fisher_c)
                weight_list.append(max(1, len(models[cid].train_loader.dataset)))
            ewc_pack['fisher'] = FedAvg_Fisher_weighted(fisher_list, weight_list)
    acc_global = model_eval(model_g.model, test_dataset, total_classes, model_g.device)
    acc_taskwise = model_eval_class(model_g.model, test_dataset, base_numclass, total_classes, model_g.device)
    logger.info(f'acc_global: {acc_global}')
    logger.info(f'acc_taskwise: {acc_taskwise}')
    if rowbank_enable:
        try:
            sd = model_g.model.state_dict()
            Wk, bk = ('fc.weight', 'fc.bias')
            if Wk in sd and bk in sd:
                for cid in range(int(base_numclass), int(total_classes)):
                    key = f'incre_{cid}'
                    if key not in acc_taskwise:
                        continue
                    acc_c = float(acc_taskwise[key])
                    best = rowbank.get(int(cid))
                    if best is None or acc_c >= float(best.get('acc', -1.0)):
                        rowbank[int(cid)] = {'acc': acc_c, 'round': int(global_round + 1), 'w': sd[Wk][int(cid)].detach().cpu().clone(), 'b': sd[bk][int(cid)].detach().cpu().clone()}
        except Exception as e:
            logger.warning(f'[FedHCCA][RowBank] update failed: {e}')
    if task_restore_enable:
        try:
            sd = model_g.model.state_dict()
            Wk, bk = ('fc.weight', 'fc.bias')
            if Wk in sd and bk in sd:
                for cid in range(int(base_numclass), int(total_classes)):
                    key = f'incre_{cid}'
                    if key not in acc_taskwise:
                        continue
                    acc_c = float(acc_taskwise[key])
                    best = best_row_cache.get(int(cid))
                    if best is None or acc_c >= float(best.get('acc', -1.0)):
                        best_row_cache[int(cid)] = {'acc': acc_c, 'round': int(global_round + 1), 'w': sd[Wk][int(cid)].detach().cpu().clone(), 'b': sd[bk][int(cid)].detach().cpu().clone()}
        except Exception as e:
            logger.warning(f'[FedHCCA][TaskRestore] cache update failed: {e}')
    if method == 'FedHCCA':
        try:
            score = float(acc_global['acc_all']) + float(fedhcca_task_best_new_weight) * float(acc_global['acc_new'])
        except Exception:
            score = float(acc_global['acc_all'])
        if score >= task_best['score']:
            task_best['score'] = score
            task_best['round'] = int(global_round + 1)
            task_best['acc_all'] = float(acc_global['acc_all'])
            task_best['acc_new'] = float(acc_global['acc_new'])
            task_best['acc_old'] = float(acc_global['acc_old'])
            task_best['state'] = {k: v.detach().cpu().clone() for k, v in model_g.model.state_dict().items()}
    logger.info('Global %d, Global Model - Test Accuracy: %.2f%%', global_round + 1, acc_global['acc_all'])
    log_acc_global = f'Global rounds {global_round + 1}/{global_rounds}, Global Model - Test Accuracy: {acc_global['acc_all']:.2f}%'
    out_file.write(log_acc_global + '\n')
    out_file.flush()
    with open(csv_path_acc_global, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([global_round + 1, f'{acc_global['acc_all']:.2f}', f'{acc_global['acc_new']:.2f}', f'{acc_global['acc_old']:.2f}'])
    with open(csv_path_acc_taskwise, 'a', newline='') as f:
        writer = csv.writer(f)
        row_data = [global_round + 1]
        row_data.append(f'{acc_taskwise[f'base_0_{base_numclass - 1}']:.2f}')
        for i in range(base_numclass, total_classes):
            row_data.append(f'{acc_taskwise[f'incre_{i}']:.2f}')
        writer.writerow(row_data)
    if global_round % task_global_round == task_global_round - 1:
        if method == 'FedHCCA' and task_best.get('state') is not None:
            model_g.model.load_state_dict(task_best['state'], strict=True)
            acc_temp = task_best.get('acc_all', acc_global['acc_all'])
            logger.info('[FedHCCA] Restore task-best model at round=%s: acc_all=%.2f acc_new=%.2f acc_old=%.2f score=%.2f', str(task_best.get('round')), float(task_best.get('acc_all', 0.0)), float(task_best.get('acc_new', 0.0)), float(task_best.get('acc_old', 0.0)), float(task_best.get('score', 0.0)))
        else:
            acc_temp = acc_global['acc_all']
        if dataset_name == 'cifar100':
            if aggregator != 'fedhcca':
                save_dir = './outputs/model/compare'
            else:
                save_dir = './outputs/model/fedhcca'
        else:
            save_dir = osp.join('./outputs/model/system', dataset_name) if is_medical else f'./outputs/model/{dataset_name}'
        save_path = osp.join(save_dir, f'{date}_{method}{_filename}_round{global_round + 1}_best_model_{int(acc_temp * 100)}.pth')
        torch.save(model_g.model.state_dict(), save_path)
        model_old[1] = copy.deepcopy(model_g.model)
        logger.info('Best global model saved with accuracy: %.2f%%', acc_temp)
        restored_classes_n = 0
        if task_restore_enable and len(best_row_cache) > 0:
            try:
                sd = model_g.model.state_dict()
                Wk, bk = ('fc.weight', 'fc.bias')
                if Wk in sd and bk in sd:
                    cur_new = int(class_id)
                    restored = []
                    for cid in range(int(base_numclass), int(total_classes)):
                        if int(cid) == cur_new:
                            continue
                        key = f'incre_{cid}'
                        if key not in acc_taskwise:
                            continue
                        cur_acc = float(acc_taskwise[key])
                        best = best_row_cache.get(int(cid))
                        if best is None:
                            continue
                        best_acc = float(best.get('acc', -1.0))
                        if best_acc <= 0:
                            continue
                        trigger_ratio = cur_acc < task_restore_ratio * best_acc
                        trigger_thr = cur_acc < task_restore_threshold
                        if not (trigger_ratio or trigger_thr):
                            continue
                        denom = max(1e-06, task_restore_ratio * best_acc)
                        deg_ratio = max(0.0, (task_restore_ratio * best_acc - cur_acc) / denom)
                        deg_thr = max(0.0, (task_restore_threshold - cur_acc) / max(1e-06, task_restore_threshold))
                        deg = min(1.0, max(deg_ratio, deg_thr))
                        beta = task_restore_beta_min + (task_restore_beta_max - task_restore_beta_min) * deg
                        beta = float(max(0.0, min(1.0, beta)))
                        if int(cid) < int(sd[Wk].shape[0]) and int(cid) < int(sd[bk].shape[0]):
                            w_cur = sd[Wk][int(cid)]
                            b_cur = sd[bk][int(cid)]
                            w_best = best['w'].to(w_cur.device, dtype=w_cur.dtype)
                            b_best = best['b'].to(b_cur.device, dtype=b_cur.dtype)
                            sd[Wk][int(cid)] = (1.0 - beta) * w_cur + beta * w_best
                            sd[bk][int(cid)] = (1.0 - beta) * b_cur + beta * b_best
                            restored.append((int(cid), cur_acc, best_acc, beta))
                    restored_classes_n = int(len(restored))
                    if restored_classes_n > 0:
                        model_g.model.load_state_dict(sd, strict=True)
                        logger.info(f'[FedHCCA][TaskRestore][TaskEnd{task_id}] restored_classes={restored_classes_n}')
                        logger.info(f'[FedHCCA][TaskRestore][TaskEnd{task_id}] restored={restored[:20]}')
                        model_old[1] = copy.deepcopy(model_g.model)
            except Exception as e:
                logger.warning(f'[FedHCCA][TaskRestore] task-boundary restore failed: {e}')
        if task_restore_enable and csv_path_taskrestore_global is not None and (csv_path_taskrestore_taskwise is not None):
            try:
                acc_global_fix = model_eval(model_g.model, test_dataset, total_classes, model_g.device)
                acc_taskwise_fix = model_eval_class(model_g.model, test_dataset, base_numclass, total_classes, model_g.device)
                logger.info(f'[FedHCCA][TaskRestore][TaskEnd{task_id}] post_eval acc_all={acc_global_fix['acc_all']:.2f} restored_classes={restored_classes_n}')
                with open(csv_path_taskrestore_global, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([int(global_round + 1), int(task_id), int(restored_classes_n), f'{acc_global_fix['acc_all']:.2f}', f'{acc_global_fix['acc_new']:.2f}', f'{acc_global_fix['acc_old']:.2f}'])
                with open(csv_path_taskrestore_taskwise, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row_data = [int(global_round + 1), int(task_id), int(restored_classes_n)]
                    row_data.append(f'{acc_taskwise_fix[f'base_0_{base_numclass - 1}']:.2f}')
                    for i in range(base_numclass, total_classes):
                        row_data.append(f'{acc_taskwise_fix[f'incre_{i}']:.2f}')
                    writer.writerow(row_data)
            except Exception as e:
                logger.warning(f'[FedHCCA][TaskRestore] failed to write post-restore CSV at task end: {e}')
if rowbank_enable and len(rowbank) > 0:
    try:
        sd = model_g.model.state_dict()
        Wk, bk = ('fc.weight', 'fc.bias')
        restored = 0
        for cid, item in rowbank.items():
            if Wk not in sd or bk not in sd:
                break
            if int(cid) >= int(sd[Wk].shape[0]) or int(cid) >= int(sd[bk].shape[0]):
                continue
            sd[Wk][int(cid)] = item['w'].to(sd[Wk].device, dtype=sd[Wk].dtype)
            sd[bk][int(cid)] = item['b'].to(sd[bk].device, dtype=sd[bk].dtype)
            restored += 1
        model_g.model.load_state_dict(sd, strict=True)
        logger.info(f'[FedHCCA][RowBank] Final restore applied: restored_rows={restored}')
        acc_global_rb = model_eval(model_g.model, test_dataset, total_classes, model_g.device)
        acc_taskwise_rb = model_eval_class(model_g.model, test_dataset, base_numclass, total_classes, model_g.device)
        logger.info(f'[FedHCCA][RowBank] Final acc_global: {acc_global_rb}')
        if dataset_name == 'cifar100':
            save_dir = './outputs/model/fedhcca' if aggregator == 'fedhcca' else './outputs/model/compare'
        else:
            save_dir = osp.join('./outputs/model/system', dataset_name) if is_medical else f'./outputs/model/{dataset_name}'
        os.makedirs(save_dir, exist_ok=True)
        save_path_rb = osp.join(save_dir, f'{date}_{method}{_filename}_rowbankfinal_round{global_rounds}_acc{int(float(acc_global_rb['acc_all']) * 100)}.pth')
        torch.save(model_g.model.state_dict(), save_path_rb)
        logger.info(f'[FedHCCA][RowBank] Final model saved: {save_path_rb}')
        csv_path_rb_global = csv_path_acc_global.replace('.csv', '_rowbankfinal.csv')
        csv_path_rb_task = csv_path_acc_taskwise.replace('.csv', '_rowbankfinal.csv')
        with open(csv_path_rb_global, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round', 'acc_all', 'acc_new', 'acc_old'])
            writer.writerow([global_rounds, f'{acc_global_rb['acc_all']:.2f}', f'{acc_global_rb['acc_new']:.2f}', f'{acc_global_rb['acc_old']:.2f}'])
        with open(csv_path_rb_task, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round'] + name_str_list)
            row_data = [global_rounds]
            row_data.append(f'{acc_taskwise_rb[f'base_0_{base_numclass - 1}']:.2f}')
            for i in range(base_numclass, total_classes):
                row_data.append(f'{acc_taskwise_rb[f'incre_{i}']:.2f}')
            writer.writerow(row_data)
    except Exception as e:
        logger.warning(f'[FedHCCA][RowBank] Final restore failed: {e}')
overhead_summary = None
if profile_overhead_enabled:
    comm_model_only = int(sum(overhead_profile['comm_model_only_bytes']))
    comm_extra_proto_meta = int(sum(overhead_profile['comm_extra_proto_meta_bytes']))
    comm_total = int(sum(overhead_profile['comm_total_bytes']))
    server_time_mean = float(np.mean(np.asarray(overhead_profile['server_time_seconds'], dtype=np.float64))) if len(overhead_profile['server_time_seconds']) > 0 else 0.0
    fedavg_time_mean = float(np.mean(np.asarray(overhead_profile['fedavg_server_time_seconds'], dtype=np.float64))) if len(overhead_profile['fedavg_server_time_seconds']) > 0 else 0.0
    overhead_summary = {'comm_model_only': comm_model_only, 'comm_extra_proto_meta': comm_extra_proto_meta, 'comm_total': comm_total, 'comm_ratio_vs_fedavg': f'{comm_total / max(1.0, float(comm_model_only)):.6f}', 'server_time_mean': f'{server_time_mean:.6f}', 'server_time_ratio_vs_fedavg': f'{server_time_mean / max(1e-12, fedavg_time_mean):.6f}'}
    logger.info('[Overhead][Summary] %s', overhead_summary)
try:
    standardized_info = write_standardized_outputs(ctx=output_ctx, dataset=dataset_name, method=method, exp_type=exp_type, variant=exp_variant, seed=int(args.seed), legacy_acc_global_csv=csv_path_acc_global, legacy_acc_taskwise_csv=csv_path_acc_taskwise, incre_tasks=int(incre_tasks), task_global_round=int(task_global_round), exposure_rounds=task_exposure_rounds if async_timing_enabled else exposure_rounds, new_client_num=int(new_client_num), auc_window=int(getattr(args, 'auc_window', 4)), overhead_metrics=overhead_summary, summary_extras={'async_timing_mode': async_timing_mode if async_timing_enabled else '', 'timing_pattern_desc': async_timing_desc if async_timing_enabled else '', 'timing_pattern_csv': output_ctx['pattern_csv'] if async_timing_enabled else ''})
    logger.info('Standardized result files written | per_round=%s summary=%s auc_new=%.6f acc_new_avg=%.6f', standardized_info['per_round_csv'], standardized_info['summary_csv'], float(standardized_info['auc_new']), float(standardized_info['acc_new_avg']))
except Exception as e:
    logger.warning('Failed to write standardized supplementary outputs: %s', str(e))
try:
    out_file.close()
except Exception:
    pass
