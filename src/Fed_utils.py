import torch.nn as nn
import torch
import copy
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import *
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader, WeightedRandomSampler
import random
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from collections import Counter
from option import args_parser
import csv
import json
args = args_parser()
FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)

def parse_int_list_csv(s: str, expected_len: int=None, name: str='list'):
    items = [int(x.strip()) for x in str(s).split(',') if str(x).strip() != '']
    if expected_len is not None and len(items) != int(expected_len):
        raise ValueError(f'Invalid {name}: expected len={expected_len}, got len={len(items)} | value={s}')
    return items

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def model_to_device(model, parallel, device):
    if parallel:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        card = torch.device('cuda:{}'.format(device))
        model.to(card)
    return model

def _ensure_sgd_and_set_lr(client, lr):
    params = [p for p in client.model.parameters() if p.requires_grad]
    if client.optimizer is None:
        client.optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    else:
        for pg in client.optimizer.param_groups:
            pg['lr'] = lr

def _ensure_sgd_fedhcca_grouped(client, lr_head: float, lr_backbone: float):
    lr_head = float(lr_head)
    lr_backbone = float(lr_backbone)
    feature_params = []
    fc_params = []
    if hasattr(client.model, 'feature'):
        feature_params = [p for p in client.model.feature.parameters() if p.requires_grad]
    if hasattr(client.model, 'fc'):
        fc_params = [p for p in client.model.fc.parameters() if p.requires_grad]
    param_groups = []
    if len(feature_params) > 0:
        param_groups.append({'params': feature_params, 'lr': lr_backbone, 'name': 'feature'})
    if len(fc_params) > 0:
        param_groups.append({'params': fc_params, 'lr': lr_head, 'name': 'fc'})
    if len(param_groups) == 0:
        all_params = [p for p in client.model.parameters()]
        param_groups = [{'params': all_params, 'lr': lr_head, 'name': 'all'}]
    need_rebuild = client.optimizer is None or not bool(getattr(client, '_fedhcca_opt_grouped', False))
    if not need_rebuild:
        try:
            cur_names = [pg.get('name', None) for pg in client.optimizer.param_groups]
            new_names = [pg.get('name', None) for pg in param_groups]
            if len(cur_names) != len(new_names) or any((a != b for a, b in zip(cur_names, new_names))):
                need_rebuild = True
        except Exception:
            need_rebuild = True
    if need_rebuild:
        client.optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=0.0005)
        client._fedhcca_opt_grouped = True
    else:
        for pg in client.optimizer.param_groups:
            nm = pg.get('name', '')
            if nm == 'feature':
                pg['lr'] = lr_backbone
            elif nm == 'fc':
                pg['lr'] = lr_head
            else:
                pg['lr'] = lr_head

def count_trainable_params(model):
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

def _maybe_build_balanced_sampler(labels, new_class_id=None, new_boost: float=1.0):
    if labels is None:
        return None
    labels = np.asarray(labels)
    if labels.size == 0:
        return None
    counts = Counter(labels.tolist())
    if len(counts) <= 1:
        return None
    inv = {k: 1.0 / float(v) for k, v in counts.items()}
    weights = np.array([inv[int(y)] for y in labels], dtype=np.float64)
    if new_class_id is not None:
        weights = np.where(labels == int(new_class_id), weights * float(new_boost), weights)
    weights_t = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_t, num_samples=int(len(weights_t)), replacement=True)

def _is_bn_buffer(k: str) -> bool:
    return 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k

def _build_weighted_mix_model(ref_model, models_list, weights_list):
    assert len(models_list) == len(weights_list)
    s = float(sum(weights_list)) + 1e-12
    w_norm = [float(w) / s for w in weights_list]
    mix = copy.deepcopy(ref_model)
    for k in mix:
        if _is_bn_buffer(k):
            mix[k] = ref_model[k].clone()
            continue
        if mix[k].dtype in FLOAT_DTYPES:
            mix[k].zero_()
    for m, w in zip(models_list, w_norm):
        for k in mix:
            if _is_bn_buffer(k):
                continue
            if mix[k].dtype in FLOAT_DTYPES:
                mix[k] += w * m[k]
            else:
                mix[k] = ref_model[k]
    return mix

def trapezoid_auc(xs, ys):
    if len(xs) < 2:
        return 0.0
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    return float(np.trapz(ys, xs))

def update_newclass_efficiency(client, class_id, global_round, expo, t, acc_cls, window=4, taus=(20.0, 40.0, 60.0), slope_k=4):
    client.newcls_curve[class_id].append((global_round, acc_cls))
    if not (expo != 9 and expo <= t <= expo + (window - 1)):
        return
    pts = client.newcls_curve[class_id]
    pts_win = pts[-window:] if len(pts) >= window else pts[:]
    rs = [p[0] for p in pts_win]
    accs = [p[1] for p in pts_win]
    auc = trapezoid_auc(rs, accs)
    auc_norm = auc / max(1.0, rs[-1] - rs[0]) if len(rs) >= 2 else 0.0
    pts_k = pts_win[:min(slope_k, len(pts_win))]
    if len(pts_k) >= 2:
        xk = np.array([p[0] for p in pts_k], dtype=np.float64)
        yk = np.array([p[1] for p in pts_k], dtype=np.float64)
        slope = float(np.polyfit(xk, yk, deg=1)[0])
    else:
        slope = 0.0
    pts_from_expo = [(r, a) for r, a in client.newcls_curve[class_id] if r >= global_round - t + expo]
    for tau in taus:
        key = (class_id, tau)
        if key not in client.newcls_metrics:
            for r, a in pts_from_expo:
                if a >= tau:
                    client.newcls_metrics[key] = int(r - (global_round - t + expo))
                    break
    client.newcls_metrics[class_id, 'AUC_norm'] = auc_norm
    client.newcls_metrics[class_id, 'slope'] = slope
    if expo != 9 and t == expo + (window - 1):
        pts = client.newcls_curve[class_id]
        pts_win = pts[-window:] if len(pts) >= window else pts[:]
        tta_dict = {tau: client.newcls_metrics.get((class_id, tau), None) for tau in taus}
        method = getattr(client, 'method', 'Unknown')
        cid = getattr(client, 'client_id', -1)
        append_newcls_metrics_csv(csv_path=getattr(client, 'newcls_csv_path', './newcls_metrics.csv'), method=method, client_id=cid, class_id=class_id, expo=expo, global_round=global_round, window=window, acc_cls=acc_cls, auc_norm=auc_norm, slope=slope, tta_dict=tta_dict, curve_pts=pts_win)

def exemplar_update(clients, num_clients, global_round, task_global_round, global_prototypes):
    for client_id in range(num_clients):
        if clients[client_id].exposure_round_init <= global_round % task_global_round <= clients[client_id].exposure_round_init + 3:
            clients[client_id].update_exemplar_set(global_prototypes, global_round, task_global_round)

def beforeTask(clients, num_clients, test_dataset, new_data_indices, client_data_indices, shuffled_client_index, exposure_rounds, total_classes):
    for logical_id in range(num_clients):
        real_id = int(shuffled_client_index[logical_id])
        clients[real_id].exposure_round_init = exposure_rounds[logical_id]
        clients[real_id].task_has_new_data = len(new_data_indices[logical_id]) > 0
        client_data_indices[real_id].extend(new_data_indices[logical_id])
    for client_id in range(num_clients):
        clients[client_id].new_class_indices = []
        clients[client_id].test_dataset = test_dataset
        clients[client_id].total_classes = total_classes
        clients[client_id].test_loader = DataLoader(clients[client_id].test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        clients[client_id].last_lr_stage = None
        clients[client_id].new_support = False
        if not hasattr(clients[client_id], 'task_has_new_data'):
            clients[client_id].task_has_new_data = False

def update_new_set(clients, new_client, num_clients, model_g, global_round, global_rounds, task_global_round, train_dataset, client_indices_dict, shuffled_client_index, global_prototypes=None):
    for c_id in range(num_clients):
        expanded = False
        if clients[c_id].model.fc.out_features != model_g.model.fc.out_features:
            clients[c_id].model.Incremental_learning(model_g.model.fc.out_features)
            expanded = True
        clients[c_id].model.load_state_dict(model_g.model.state_dict(), strict=True)
        clients[c_id].model = clients[c_id].model.to(clients[c_id].device)
        if expanded:
            clients[c_id].optimizer = None
            clients[c_id].scheduler = None
            clients[c_id].last_lr_stage = None
    for logical_id in range(num_clients):
        real_id = int(shuffled_client_index[logical_id])
        expo = clients[real_id].exposure_round_init
        t = global_round % task_global_round
        client_indices_temp = None
        if logical_id < new_client and getattr(clients[real_id], 'task_has_new_data', False):
            if expo != -1 and expo <= t <= expo + 3:
                client_indices_temp = []
                for r in range(1 + t - expo):
                    client_indices_temp.extend(client_indices_dict[f'client_index{logical_id}'][f'round_{r}'])
        if client_indices_temp is not None:
            clients[real_id].new_class_indices = client_indices_temp

def local_train(clients, client_id, class_id, model_old, global_round, task_global_round, client_prototypes=None, global_prototypes=None, client_weights=None, model_g=None, aggregator=None, IL_method=None, ewc_pack=None):
    pre_state = {}
    for name, param in clients[client_id].model.named_parameters():
        if param.data.dtype.is_floating_point:
            pre_state[name] = param.detach().cpu().clone()
    expo = clients[client_id].exposure_round_init
    t = global_round % task_global_round
    task_has_new_data = bool(getattr(clients[client_id], 'task_has_new_data', False))
    Wk, bk = ('fc.weight', 'fc.bias')
    w0 = clients[client_id].model.state_dict()[Wk][class_id].detach().cpu().clone()
    b0 = clients[client_id].model.state_dict()[bk][class_id].detach().cpu().clone()
    if task_has_new_data and expo == t:
        if clients[client_id].current_class != clients[client_id].init_classes - 1:
            clients[client_id].last_class = clients[client_id].current_class
        clients[client_id].current_class = class_id
        clients[client_id].learned_numclass += 1
        if task_has_new_data:
            clients[client_id].new_support = True
    ablate_local_plain_train = bool(int(getattr(args, 'ablate_local_plain_train', 0)))
    enable_fedhcca_tweaks = aggregator == 'fedhcca' and getattr(args, 'method', None) == 'FedHCCA' and (not ablate_local_plain_train)
    balance_sampler = enable_fedhcca_tweaks and bool(int(getattr(args, 'balance_sampler', 1)))
    new_class_boost = float(getattr(args, 'new_class_boost', 2.0)) if enable_fedhcca_tweaks else 1.0
    sampler_early_rounds = int(getattr(args, 'fedhcca_sampler_early_rounds', -1)) if enable_fedhcca_tweaks else -1
    is_exposed_train = task_has_new_data and expo <= t <= expo + 3
    policy = str(getattr(args, 'fedhcca_backbone_policy', 'legacy')).lower()
    lowlr_scale = float(getattr(args, 'fedhcca_backbone_lowlr_scale', 0.05))
    freeze_mode = str(getattr(args, 'fedhcca_freeze_backbone', 'none')).lower()
    freeze_backbone = False
    use_backbone_lowlr = False
    if not enable_fedhcca_tweaks:
        freeze_backbone = False
    elif policy == 'all_freeze':
        freeze_backbone = True
        use_backbone_lowlr = False
    elif policy == 'legacy':
        if task_has_new_data:
            if freeze_mode == 'task':
                freeze_backbone = t >= expo
            elif freeze_mode == 'exposure':
                freeze_backbone = is_exposed_train
            else:
                freeze_backbone = False
        else:
            freeze_backbone = False
    elif not task_has_new_data:
        freeze_backbone = True
        use_backbone_lowlr = False
    elif is_exposed_train:
        freeze_backbone = False
        use_backbone_lowlr = policy == 'exp_lowlr_then_freeze'
    else:
        freeze_backbone = True
        use_backbone_lowlr = False
    if enable_fedhcca_tweaks and hasattr(clients[client_id].model, 'feature') and hasattr(clients[client_id].model, 'fc'):
        for p in clients[client_id].model.fc.parameters():
            p.requires_grad = True
        if freeze_backbone:
            for p in clients[client_id].model.feature.parameters():
                p.requires_grad = False
        else:
            for p in clients[client_id].model.feature.parameters():
                p.requires_grad = True
        prev_key = getattr(clients[client_id], '_fedhcca_backbone_key', None)
        key = (bool(freeze_backbone), bool(use_backbone_lowlr))
        if prev_key is None or prev_key != key:
            clients[client_id].optimizer = None
            setattr(clients[client_id], '_fedhcca_opt_grouped', False)
        clients[client_id]._fedhcca_backbone_key = key
    if not task_has_new_data or t < expo:
        clients[client_id].train_dataset.getTrainData(classes=None, new_class_id=clients[client_id].recent_full_class, exemplar_set=clients[client_id].exemplar_set, exemplar_label_set=clients[client_id].exemplar_label_set, new_class_indices=clients[client_id].recent_full_indices)
        sampler = None
        if balance_sampler and sampler_early_rounds < 0:
            labels = getattr(clients[client_id].train_dataset, 'TrainLabels', None)
            if labels is not None:
                labels_arr = np.asarray(labels)
                if labels_arr.size > 0:
                    sampler = _maybe_build_balanced_sampler(labels_arr)
        clients[client_id].train_loader = DataLoader(clients[client_id].train_dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=4)
    if task_has_new_data and expo <= t <= expo + 3:
        clients[client_id].compute_local_prototypes(client_id=client_id, class_id=class_id, client_prototypes=client_prototypes, client_weights=client_weights, new_class_indices=clients[client_id].new_class_indices)
        clients[client_id].recent_full_class = class_id
        clients[client_id].recent_full_indices = clients[client_id].new_class_indices
        clients[client_id].train_dataset.getTrainData(classes=None, new_class_id=class_id, exemplar_set=clients[client_id].exemplar_set, exemplar_label_set=clients[client_id].exemplar_label_set, new_class_indices=clients[client_id].new_class_indices)
        sampler = None
        if balance_sampler:
            use_sampler = False
            if sampler_early_rounds < 0:
                use_sampler = True
            elif sampler_early_rounds == 0:
                use_sampler = False
            else:
                use_sampler = t <= expo + sampler_early_rounds - 1
        if balance_sampler and use_sampler:
            labels = getattr(clients[client_id].train_dataset, 'TrainLabels', None)
            if labels is not None:
                labels_arr = np.asarray(labels)
                if labels_arr.size > 0:
                    sampler = _maybe_build_balanced_sampler(labels_arr, new_class_id=class_id, new_boost=new_class_boost)
        clients[client_id].train_loader = DataLoader(clients[client_id].train_dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=4)
    IL_eff = IL_method
    if ablate_local_plain_train and IL_eff == 'fedhcca':
        IL_eff = None
    if IL_eff == 'fedhcca':
        if getattr(args, 'dataset', '') == 'isic2019':
            lr0 = [0.0003, 0.0002, 0.0001, 0.0001]
            base_lr_old = 5e-05
        else:
            lr0 = [0.001, 0.0008, 0.0005, 0.0005]
            base_lr_old = 0.0003
        t = global_round % task_global_round
        expo = clients[client_id].exposure_round_init
        OLD_STAGE = 999
        if not task_has_new_data or t < expo:
            stage = OLD_STAGE
            if clients[client_id].last_lr_stage != stage or clients[client_id].optimizer is None:
                _ensure_sgd_and_set_lr(clients[client_id], base_lr_old)
                clients[client_id].last_lr_stage = stage
        else:
            stage = t - expo
            if stage < len(lr0):
                if clients[client_id].last_lr_stage != stage or clients[client_id].optimizer is None:
                    lr_head = float(lr0[stage])
                    if enable_fedhcca_tweaks and use_backbone_lowlr and (not freeze_backbone):
                        _ensure_sgd_fedhcca_grouped(clients[client_id], lr_head=lr_head, lr_backbone=lr_head * float(lowlr_scale))
                    else:
                        _ensure_sgd_and_set_lr(clients[client_id], lr_head)
                    clients[client_id].last_lr_stage = stage
            else:
                stage = OLD_STAGE
                if clients[client_id].last_lr_stage != stage:
                    _ensure_sgd_and_set_lr(clients[client_id], base_lr_old)
                    clients[client_id].last_lr_stage = stage
    elif t == 0:
        clients[client_id].optimizer = optim.SGD(clients[client_id].model.parameters(), lr=clients[client_id].learning_rate, weight_decay=1e-05)
        clients[client_id].scheduler = torch.optim.lr_scheduler.MultiStepLR(clients[client_id].optimizer, milestones=[10, 15, 18], gamma=0.2)
    if t == global_round % task_global_round:
        opt = clients[client_id].optimizer
        lr_now = opt.param_groups[0]['lr'] if opt is not None else -1
        trainable_cnt = count_trainable_params(clients[client_id].model)

    def _param_in_optimizer(p, opt):
        if opt is None:
            return False
        for pg in opt.param_groups:
            for q in pg['params']:
                if q is p:
                    return True
        return False
    fc_w = clients[client_id].model.fc.weight
    fc_b = clients[client_id].model.fc.bias
    feature_in_opt = None
    if hasattr(clients[client_id].model, 'feature'):
        feature_params = list(clients[client_id].model.feature.parameters())
        if len(feature_params) > 0:
            feature_in_opt = any((_param_in_optimizer(p, clients[client_id].optimizer) for p in feature_params))
        else:
            feature_in_opt = False
    all_labels = []
    for _, _, labels in clients[client_id].train_loader:
        all_labels.extend(labels.cpu().numpy())
    new_class_labels = [l for l in all_labels if l == class_id]
    unique, counts = np.unique(all_labels, return_counts=True)
    label_count_dict = dict(zip(unique, counts))
    if not task_has_new_data:
        count_new_samples = 0
    else:
        count_new_samples = len([l for l in all_labels if l == class_id])
    thr = clients[client_id].proto_stage_threshold
    lambda_max = clients[client_id].lambda_max_proto
    if count_new_samples == 0:
        lambda_proto = 0.0
    else:
        ratio = min(1.0, count_new_samples / float(thr))
        lambda_proto = lambda_max * (1.0 - ratio) ** 2
    local_client_prototypes = None
    if client_prototypes is not None:
        local_client_prototypes = client_prototypes[client_id]
    new_loss_weight = float(getattr(args, 'fedhcca_new_loss_weight', 1.0))
    protect_new_head = bool(int(getattr(args, 'fedhcca_protect_new_head', 1)))
    protect_recent_old_head = bool(int(getattr(args, 'fedhcca_protect_recent_old_head', 1)))
    recent_old_weight = float(getattr(args, 'fedhcca_recent_old_weight', 0.0))
    freeze_past_new_rows_grad = bool(int(getattr(args, 'fedhcca_freeze_past_new_rows_grad', 0)))
    bn_policy = int(getattr(args, 'fedhcca_freeze_bn', 1))
    if not enable_fedhcca_tweaks or bn_policy <= 0:
        freeze_bn_stats = False
    elif bn_policy == 1:
        freeze_bn_stats = bool(freeze_backbone)
    elif bn_policy == 2:
        freeze_bn_stats = bool(freeze_backbone) or not bool(is_exposed_train)
    else:
        freeze_bn_stats = True
    if aggregator != 'fedhcca' or ablate_local_plain_train:
        plain_aggregator = aggregator if aggregator != 'fedhcca' else 'fedavg'
        plain_IL = IL_method if not ablate_local_plain_train else None
        clients[client_id].train_compare(global_round=global_round, task_global_round=task_global_round, model_old=model_old, client_id=client_id, model_g=model_g, aggregator=plain_aggregator, IL_method=plain_IL, ewc_pack=ewc_pack)
    else:
        clients[client_id].train(g_round=global_round, task_global_round=task_global_round, model_old=model_old, client_id=client_id, class_id=class_id, client_prototypes=local_client_prototypes, global_prototypes=global_prototypes, lambda_proto=lambda_proto, count_new_samples=count_new_samples, is_exposed_train=is_exposed_train, freeze_backbone=freeze_backbone, freeze_bn_stats=freeze_bn_stats, new_loss_weight=new_loss_weight if is_exposed_train else 1.0, protect_new_head=protect_new_head, protect_recent_old_head=protect_recent_old_head, recent_old_weight=recent_old_weight, freeze_past_new_rows_grad=freeze_past_new_rows_grad)
    new_cls = [class_id, class_id + 1]
    acc_new = eval_acc(clients[client_id].model, new_cls, clients[client_id].test_dataset, clients[client_id].device)
    update_newclass_efficiency(clients[client_id], class_id=class_id, global_round=global_round, expo=expo, t=t, acc_cls=acc_new, window=4, taus=(20.0, 40.0, 60.0), slope_k=4)
    if task_has_new_data and t == expo + 3:
        m = clients[client_id].newcls_metrics
    if task_has_new_data and expo <= t <= expo + 3:
        fc_w = clients[client_id].model.fc.weight
        if fc_w.grad is None:
            pass
        else:
            gnorm = fc_w.grad[class_id].data.norm().item()
    local_model = clients[client_id].model.state_dict()
    acc_local = model_eval(clients[client_id].model, clients[client_id].test_dataset, clients[client_id].total_classes, clients[client_id].device)
    local_grad = {}
    for name, param in clients[client_id].model.named_parameters():
        if param.requires_grad and param.data.dtype.is_floating_point:
            pre = pre_state.get(name)
            if pre is None:
                pre = param.detach().cpu().clone()
            local_grad[name] = param.detach().cpu() - pre
    if t == 9:
        if IL_method == 'fedhcca' or IL_method == 'iCaRL':
            clients[client_id].exemplar_update_set(global_prototypes=global_prototypes, global_round=global_round, new_class_indices=clients[client_id].new_class_indices, new_class_id=clients[client_id].current_class, IL_method=IL_method)
        if clients[client_id].current_class not in clients[client_id].old_classes:
            clients[client_id].old_classes.append(clients[client_id].current_class)
        if IL_method == 'EWC':
            fisher = clients[client_id].estimate_fisher_diag(clients[client_id].train_loader, max_batches=clients[client_id].ewc_batches)
            ref = clients[client_id].snapshot_params()
            clients[client_id].update_ewc_state(fisher, ref)
    w1 = clients[client_id].model.state_dict()[Wk][class_id].detach().cpu()
    b1 = clients[client_id].model.state_dict()[bk][class_id].detach().cpu()
    return (local_model, local_grad)

def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            if w_avg[k].dtype in [torch.float32, torch.float64, torch.float16]:
                for i in range(1, len(models)):
                    w_avg[k] += models[i][k]
                w_avg[k] = w_avg[k] / len(models)
            else:
                w_avg[k] = models[0][k]
    return w_avg

def _to_vec(proto):
    if proto is None:
        return None
    if isinstance(proto, list):
        if len(proto) == 0:
            return None
        v = np.mean(np.stack(proto, axis=0), axis=0)
    else:
        v = proto
    n = np.linalg.norm(v) + 1e-12
    return v / n

def aggregate_prototypes(client_prototypes, global_prototypes, client_weights, num_clients, cls_range):
    updated_labels = []
    for label in range(cls_range[0], cls_range[1]):
        items = []
        total_weight = 0.0
        for cid in range(num_clients):
            if label not in client_prototypes[cid]:
                continue
            w = float(client_weights[cid].get(label, 0.0))
            if w <= 0:
                continue
            v = _to_vec(client_prototypes[cid][label])
            if v is None:
                continue
            items.append((v, w))
            total_weight += w
        if total_weight <= 0 or len(items) == 0:
            continue
        dim = items[0][0].shape[0]
        g = np.zeros((dim,), dtype=np.float32)
        for v, w in items:
            g += w / total_weight * v
        g = g / (np.linalg.norm(g) + 1e-12)
        global_prototypes[label] = g
        updated_labels.append(label)
    return global_prototypes

def cosine(a, b, eps=1e-08):
    a = a.astype(float)
    b = b.astype(float)
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def euclidean(a, b):
    return float(np.linalg.norm(a - b))

def compute_group_contrib(model_group, ref_model):
    contrib = 0.0
    for name, param in model_group.items():
        if name in ref_model:
            if param.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                continue
            diff = (param - ref_model[name]).detach().view(-1)
            contrib += torch.norm(diff).item()
    return contrib

def cos_sim(a, b, eps=1e-12):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def to_unit(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x) + eps
    return x / n

def grad_to_vec(gdict, keys_keep=None, ref_numel_map=None):
    if gdict is None:
        return None
    if keys_keep is None or ref_numel_map is None:
        vecs = []
        for k, v in gdict.items():
            if keys_keep is not None and k not in keys_keep:
                continue
            if not torch.is_tensor(v):
                continue
            vv = v.detach().float().view(-1).cpu().numpy()
            vecs.append(vv)
        if len(vecs) == 0:
            return None
        return np.concatenate(vecs, axis=0)
    vecs = []
    for k in keys_keep:
        expected = int(ref_numel_map.get(k, 0))
        if expected <= 0:
            continue
        v = gdict.get(k, None)
        if torch.is_tensor(v):
            vv = v.detach().float().view(-1).cpu().numpy()
            if vv.shape[0] == expected:
                vecs.append(vv)
            elif vv.shape[0] > expected:
                vecs.append(vv[:expected])
            else:
                pad = np.zeros((expected - vv.shape[0],), dtype=np.float32)
                vecs.append(np.concatenate([vv.astype(np.float32, copy=False), pad], axis=0))
        else:
            vecs.append(np.zeros((expected,), dtype=np.float32))
    if len(vecs) == 0:
        return None
    return np.concatenate(vecs, axis=0)

def _safe_get(sd, k):
    return sd[k] if sd is not None and k in sd else None

def _tensor_norm(x):
    return float(torch.norm(x.detach()).item())

def _print_recent_old_probe(new_class_id, models, client_prototypes, ref_model, client_weights, num_clients):
    recent_old = new_class_id - 1
    if recent_old < 0:
        return
    clients_ro = []
    sum_w_ro = 0.0
    for cid in range(num_clients):
        cp = client_prototypes[cid]
        if recent_old in cp:
            clients_ro.append(cid)
            sum_w_ro += float(client_weights[cid].get(recent_old, 0.0))
    ref_w = _safe_get(ref_model, 'fc.weight')
    ref_b = _safe_get(ref_model, 'fc.bias')
    if ref_w is None:
        return
    dwn = []
    dbn = []
    for cid in clients_ro:
        w = _safe_get(models[cid], 'fc.weight')
        b = _safe_get(models[cid], 'fc.bias')
        if w is None:
            continue
        if recent_old < w.shape[0]:
            dwn.append(_tensor_norm(w[recent_old] - ref_w[recent_old]))
        if b is not None and ref_b is not None and (recent_old < b.shape[0]):
            dbn.append(abs(float(b[recent_old] - ref_b[recent_old])))
    if len(dwn) > 0:
        pass
    if len(dbn) > 0:
        pass

def cluster_and_aggregate(new_class_id, models, client_prototypes, ref_model, client_weights, num_clients, clients_grad, metric='cosine', exposure_rounds=None, task_global_round=10, global_round=None, new_support_list=None, test_dataset=None):
    server_ablation = str(getattr(args, 'server_ablation', 'full')).strip().lower()
    if server_ablation == '':
        server_ablation = 'full'
    ref_w_all = _safe_get(ref_model, 'fc.weight')
    if ref_w_all is None:
        raise KeyError("ref_model missing key 'fc.weight'")
    if new_class_id >= ref_w_all.shape[0]:
        raise IndexError(f'new_class_id={new_class_id} out of range for fc.weight with shape {tuple(ref_w_all.shape)}')
    Wg_before = ref_w_all[new_class_id].detach().clone()
    round_tag = int(global_round) + 1 if global_round is not None else '?'
    new_clients_list = []
    old_clients_list = []
    for cid in range(num_clients):
        cp = client_prototypes[cid]
        has_proto = new_class_id in cp
        support = new_support_list[cid]
        if has_proto and support:
            new_clients_list.append(cid)
        else:
            old_clients_list.append(cid)
    avg_old_model = None
    has_old = len(old_clients_list) > 0
    if has_old:
        old_models = [models[cid] for cid in old_clients_list]
        avg_old_model = FedAvg(old_models)
    if has_old:
        Wk, bk = ('fc.weight', 'fc.bias')
        avg_old_model[Wk][new_class_id] = ref_model[Wk][new_class_id].clone()
        avg_old_model[bk][new_class_id] = ref_model[bk][new_class_id].clone()
    avg_new_models = []
    if len(new_clients_list) == 0:
        pass
    elif len(new_clients_list) == 1:
        only_cid = new_clients_list[0]
        avg_new_models.append(models[only_cid])
    else:
        cluster2clients = {}
        if server_ablation == 'w/o_clustering':
            cluster2clients[0] = list(new_clients_list)
        else:
            m = len(new_clients_list)
            sim_matrix = np.zeros((m, m), dtype=float)
            keys_keep = None
            ref_numel_map = None
            sample_gid = clients_grad[new_clients_list[0]]
            keys_keep = [k for k in sample_gid.keys() if 'fc' in k or 'classifier' in k]
            if len(keys_keep) == 0:
                keys_keep = None
            else:
                ref_numel_map = {}
                for k in keys_keep:
                    v = sample_gid.get(k, None)
                    if torch.is_tensor(v):
                        ref_numel_map[k] = int(v.numel())
            for i in range(m):
                cid_i = new_clients_list[i]
                proto_i = to_unit(client_prototypes[cid_i][new_class_id])
                gvec_i = grad_to_vec(clients_grad[cid_i], keys_keep, ref_numel_map)
                sim_matrix[i, i] = 1.0
                for j in range(i + 1, m):
                    cid_j = new_clients_list[j]
                    proto_j = to_unit(client_prototypes[cid_j][new_class_id])
                    gvec_j = grad_to_vec(clients_grad[cid_j], keys_keep, ref_numel_map)
                    proto_sim = cos_sim(proto_i, proto_j)
                    grad_sim = cos_sim(gvec_i, gvec_j) if gvec_i is not None and gvec_j is not None else 0.0
                    if server_ablation == 'w/o_prototype_similarity':
                        combined_sim = grad_sim
                    elif server_ablation == 'w/o_update_similarity':
                        combined_sim = proto_sim
                    else:
                        combined_sim = 0.5 * proto_sim + 0.5 * grad_sim
                    sim_matrix[i, j] = combined_sim
                    sim_matrix[j, i] = combined_sim
            sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
            dist_matrix = 1.0 - sim_matrix
            num_clusters = min(max(1, len(new_clients_list) // 2), len(new_clients_list) - 1)
            clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='average')
            labels = clustering.fit_predict(dist_matrix)
            for idx, cid in enumerate(new_clients_list):
                c = int(labels[idx])
                cluster2clients.setdefault(c, []).append(cid)
        for c, members in cluster2clients.items():
            if len(members) == 1:
                avg_new_models.append(models[members[0]])
                continue
            ws = []
            for cid in members:
                w = float(client_weights[cid].get(new_class_id, 1.0))
                ws.append(w)
            s = sum(ws) + 1e-12
            w_avg = {k: torch.zeros_like(models[members[0]][k]) for k in models[members[0]].keys()}
            for cid, w in zip(members, ws):
                for k in w_avg:
                    if w_avg[k].dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                        w_avg[k] += w / s * models[cid][k]
                    else:
                        w_avg[k] = models[members[0]][k]
            avg_new_models.append(w_avg)
    group_list = []
    if len(old_clients_list) > 0:
        c_old = compute_group_contrib(avg_old_model, ref_model)
        group_list.append(('old', avg_old_model, c_old))
    for idx, avg_model in enumerate(avg_new_models):
        c_new = compute_group_contrib(avg_model, ref_model)
        group_list.append((f'new_{idx}', avg_model, c_new))
    use_uniform_group_weights = server_ablation == 'w/o_contribution_weighting'
    if use_uniform_group_weights:
        w = np.ones((len(group_list),), dtype=np.float32) / max(1, len(group_list))
    else:
        contribs = np.array([x[2] for x in group_list], dtype=np.float32)
        contribs = np.sqrt(contribs + 1e-12)
        w = contribs / (contribs.sum() + 1e-12)
    if has_old and (not use_uniform_group_weights):
        w_old_min = 0.3
        old_min_triggered = False
        w_before = w.copy()
        if w[0] < w_old_min:
            old_min_triggered = True
            rest = w.sum() - w[0]
            w[0] = w_old_min
            scale = (1.0 - w_old_min) / (rest + 1e-12)
            for i in range(1, len(w)):
                w[i] *= scale
        group_names = [x[0] for x in group_list]
        w_map_after = {name: float(wi) for name, wi in zip(group_names, w)}
        w_map_before = {name: float(wi) for name, wi in zip(group_names, w_before)}
        w_old_after = float(w[0])
        w_new_sum_after = float(w[1:].sum()) if len(w) > 1 else 0.0
    elif not has_old:
        group_names = [x[0] for x in group_list]
        w_map_after = {name: float(wi) for name, wi in zip(group_names, w)}
    else:
        group_names = [x[0] for x in group_list]
        w_map_after = {name: float(wi) for name, wi in zip(group_names, w)}
    global_model = copy.deepcopy(ref_model)
    w_old_model = avg_old_model if has_old else ref_model
    if len(avg_new_models) > 0:
        offset = 1 if has_old else 0
        new_ws = []
        for j in range(len(avg_new_models)):
            new_ws.append(float(w[j + offset]))
        w_new_mix = _build_weighted_mix_model(ref_model, avg_new_models, new_ws)
    else:
        w_new_mix = None
    backbone_new_ratio, old_fc_keep, mu, conflict_cos = pick_ab_params_by_conflict(new_class_id=new_class_id, has_old=has_old, avg_old_model=avg_old_model, w_new_mix=w_new_mix, ref_model=ref_model, new_clients_list=new_clients_list, old_clients_list=old_clients_list)
    for k in global_model:
        if global_model[k].dtype not in FLOAT_DTYPES:
            continue
        if k.startswith('fc.'):
            continue
        if w_new_mix is None:
            global_model[k] = w_old_model[k].clone()
        else:
            global_model[k] = (1.0 - backbone_new_ratio) * w_old_model[k] + backbone_new_ratio * w_new_mix[k]
    for k in global_model:
        if global_model[k].dtype not in FLOAT_DTYPES:
            continue
        if k.startswith('fc.'):
            global_model[k] = w_old_model[k].clone()
    Wk, bk = ('fc.weight', 'fc.bias')
    if w_new_mix is not None and Wk in global_model and (Wk in w_new_mix):
        if has_old and old_fc_keep < 1.0:
            global_model[Wk] = old_fc_keep * global_model[Wk] + (1.0 - old_fc_keep) * ref_model[Wk]
            global_model[bk] = old_fc_keep * global_model[bk] + (1.0 - old_fc_keep) * ref_model[bk]
        if mu > 0:
            global_model[Wk][new_class_id] = (1.0 - mu) * global_model[Wk][new_class_id] + mu * w_new_mix[Wk][new_class_id]
            global_model[bk][new_class_id] = (1.0 - mu) * global_model[bk][new_class_id] + mu * w_new_mix[bk][new_class_id]
    if getattr(args, 'method', None) == 'FedHCCA':
        if w_new_mix is not None and Wk in global_model and (Wk in w_new_mix):
            if int(new_class_id) < int(global_model[Wk].shape[0]):
                global_model[Wk][new_class_id] = w_new_mix[Wk][new_class_id].clone()
                if bk in global_model and bk in w_new_mix and (int(new_class_id) < int(global_model[bk].shape[0])):
                    global_model[bk][new_class_id] = w_new_mix[bk][new_class_id].clone()
        if int(getattr(args, 'fedhcca_protect_past_new_rows', 1)) == 1:
            base_cls = int(getattr(args, 'baseclass', 0))
            max_old_new = int(new_class_id) - 1
            protected = 0
            locked = 0
            beta = float(getattr(args, 'fedhcca_protect_past_new_beta', 0.3))
            beta = 0.0 if beta < 0 else 1.0 if beta > 1.0 else beta
            min_support = int(getattr(args, 'fedhcca_protect_past_new_min_support', 2))
            if min_support < 1:
                min_support = 1
            min_samples = int(getattr(args, 'fedhcca_protect_past_new_min_samples', 0))
            if min_samples < 0:
                min_samples = 0
            if Wk in global_model and max_old_new >= base_cls:
                for cls_id in range(base_cls, max_old_new + 1):
                    cls_clients = []
                    cls_ws = []
                    for cid in range(int(num_clients)):
                        w_cnt = float(client_weights[cid].get(cls_id, 0.0))
                        if w_cnt <= 0.0:
                            continue
                        if min_samples > 0 and w_cnt < float(min_samples):
                            continue
                        if Wk not in models[cid] or cls_id >= int(models[cid][Wk].shape[0]):
                            continue
                        cls_clients.append(cid)
                        cls_ws.append(w_cnt)
                    if len(cls_clients) < min_support:
                        if Wk in ref_model and cls_id < int(ref_model[Wk].shape[0]) and (cls_id < int(global_model[Wk].shape[0])):
                            global_model[Wk][cls_id] = ref_model[Wk][cls_id].clone()
                            if bk in global_model and bk in ref_model and (cls_id < int(ref_model[bk].shape[0])) and (cls_id < int(global_model[bk].shape[0])):
                                global_model[bk][cls_id] = ref_model[bk][cls_id].clone()
                            locked += 1
                        continue
                    s = float(sum(cls_ws)) + 1e-12
                    row_w = None
                    for cid, w_cnt in zip(cls_clients, cls_ws):
                        part = float(w_cnt) / s * models[cid][Wk][cls_id]
                        row_w = part.clone() if row_w is None else row_w + part
                    if row_w is not None and cls_id < int(global_model[Wk].shape[0]):
                        if beta >= 1.0:
                            global_model[Wk][cls_id] = row_w.clone()
                        elif beta > 0.0:
                            global_model[Wk][cls_id] = (1.0 - beta) * global_model[Wk][cls_id] + beta * row_w
                    if bk in global_model:
                        row_b = None
                        for cid, w_cnt in zip(cls_clients, cls_ws):
                            if bk not in models[cid] or cls_id >= int(models[cid][bk].shape[0]):
                                continue
                            part = float(w_cnt) / s * models[cid][bk][cls_id]
                            row_b = part.clone() if row_b is None else row_b + part
                        if row_b is not None and cls_id < int(global_model[bk].shape[0]):
                            if beta >= 1.0:
                                global_model[bk][cls_id] = row_b.clone()
                            elif beta > 0.0:
                                global_model[bk][cls_id] = (1.0 - beta) * global_model[bk][cls_id] + beta * row_b
                    protected += 1
            if protected > 0:
                pass
            elif locked > 0:
                pass
    if w_new_mix is not None and 'fc.weight' in w_new_mix:
        cos = torch.nn.functional.cosine_similarity(global_model['fc.weight'][new_class_id].flatten(), w_new_mix['fc.weight'][new_class_id].flatten(), dim=0).item()
    if getattr(args, 'method', None) == 'FedHCCA':
        wa_mode = int(getattr(args, 'fedhcca_weight_align_new_row', 0))
        if wa_mode in (1, 2):
            Wk, bk = ('fc.weight', 'fc.bias')
            if Wk in global_model and int(new_class_id) < int(global_model[Wk].shape[0]) and (int(new_class_id) > 0):
                W = global_model[Wk]
                if W.dtype in FLOAT_DTYPES:
                    base_cls = int(getattr(args, 'baseclass', 0))
                    if int(new_class_id) > base_cls:
                        old_rows = W[base_cls:int(new_class_id)]
                    else:
                        old_rows = W[:int(new_class_id)]
                    if old_rows.numel() > 0:
                        target_norm = old_rows.norm(p=2, dim=1).mean().detach()

                        def _align_row(row_id: int):
                            if row_id < 0 or row_id >= int(W.shape[0]):
                                return None
                            nrm = W[row_id].norm(p=2)
                            sc = (target_norm / (nrm + 1e-12)).clamp(min=0.0)
                            global_model[Wk][row_id] = global_model[Wk][row_id] * sc
                            return (float(nrm), float(sc))
                        if wa_mode == 1:
                            new_norm_before, scale = _align_row(int(new_class_id))
                            bias_scale = float(getattr(args, 'fedhcca_new_row_bias_scale', 1.0))
                            if bk in global_model and int(new_class_id) < int(global_model[bk].shape[0]) and (bias_scale != 1.0):
                                global_model[bk][int(new_class_id)] = global_model[bk][int(new_class_id)] * bias_scale
                            try:
                                pass
                            except Exception:
                                pass
                        else:
                            aligned = 0
                            for rid in range(base_cls, int(new_class_id) + 1):
                                out = _align_row(int(rid))
                                if out is not None:
                                    aligned += 1
                            bias_scale = float(getattr(args, 'fedhcca_new_row_bias_scale', 1.0))
                            if bk in global_model and int(new_class_id) < int(global_model[bk].shape[0]) and (bias_scale != 1.0):
                                global_model[bk][int(new_class_id)] = global_model[bk][int(new_class_id)] * bias_scale
                            try:
                                pass
                            except Exception:
                                pass
    Wg_after = global_model['fc.weight'][new_class_id]
    delta_g = (Wg_after - Wg_before).norm().item()
    deltas_all = []
    deltas_new = []
    deltas_old = []
    new_set = set(new_clients_list)
    old_set = set(old_clients_list)
    for cid, w in enumerate(models):
        Wi = w['fc.weight'][new_class_id]
        delta_i = float((Wi - Wg_before).norm().item())
        deltas_all.append(delta_i)
        if cid in new_set:
            deltas_new.append(delta_i)
        elif cid in old_set:
            deltas_old.append(delta_i)
    mean_all = sum(deltas_all) / max(len(deltas_all), 1)
    mean_new = sum(deltas_new) / max(len(deltas_new), 1) if len(deltas_new) > 0 else 0.0
    mean_old = sum(deltas_old) / max(len(deltas_old), 1) if len(deltas_old) > 0 else 0.0
    shrink_all = delta_g / (mean_all + 1e-12)
    shrink_new = delta_g / (mean_new + 1e-12) if mean_new > 0 else float('inf')
    shrink_old = delta_g / (mean_old + 1e-12) if mean_old > 0 else float('inf')
    _print_recent_old_probe(new_class_id, models, client_prototypes, ref_model, client_weights, num_clients)
    if getattr(args, 'method', None) == 'FedHCCA':
        ema = float(getattr(args, 'server_ema', 0.0))
        ema_fc_new = float(getattr(args, 'server_ema_fc_new', 0.0))
        if ema > 0.0 or ema_fc_new > 0.0:
            prev_sd = ref_model
            sd = global_model
            if ema > 0.0:
                for k in list(sd.keys()):
                    if k not in prev_sd:
                        continue
                    if torch.is_tensor(sd[k]) and sd[k].dtype.is_floating_point:
                        sd[k] = ema * prev_sd[k] + (1.0 - ema) * sd[k]
            if ema_fc_new > 0.0 and 'fc.weight' in sd and ('fc.weight' in prev_sd):
                if int(new_class_id) < int(sd['fc.weight'].shape[0]):
                    sd['fc.weight'][int(new_class_id)] = ema_fc_new * prev_sd['fc.weight'][int(new_class_id)] + (1.0 - ema_fc_new) * sd['fc.weight'][int(new_class_id)]
            if ema_fc_new > 0.0 and 'fc.bias' in sd and ('fc.bias' in prev_sd):
                if int(new_class_id) < int(sd['fc.bias'].shape[0]):
                    sd['fc.bias'][int(new_class_id)] = ema_fc_new * prev_sd['fc.bias'][int(new_class_id)] + (1.0 - ema_fc_new) * sd['fc.bias'][int(new_class_id)]
            global_model = sd
    return global_model

def eval_acc(model, cls_range, test_dataset, device):
    model.eval()
    test_dataset.getTestData(cls_range)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=args.batch_size)
    correct, total = (0, 0)
    test_labels = []
    for _, _, labels in test_loader:
        test_labels.extend(labels.cpu().numpy())
    with torch.no_grad():
        for step, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = (imgs.cuda(device), labels.cuda(device))
            outputs = model(imgs)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total if total > 0 else 0
    return acc

def model_eval(model, test_dataset, total_classes, device):
    all_cls = [0, total_classes]
    new_cls = [total_classes - 1, total_classes]
    old_cls = [0, total_classes - 1]
    acc_all = acc_new = acc_old = 0
    acc_all = eval_acc(model, all_cls, test_dataset, device)
    acc_new = eval_acc(model, new_cls, test_dataset, device)
    acc_old = eval_acc(model, old_cls, test_dataset, device)
    accuracy = {'acc_all': acc_all, 'acc_new': acc_new, 'acc_old': acc_old}
    test_dataset.getTestData(all_cls)
    model.train()
    return accuracy

def model_eval_class(model, test_dataset, base_numclass, total_classes, device):
    accuracy = {}
    old_cls = [0, base_numclass]
    acc_old = eval_acc(model, old_cls, test_dataset, device)
    base_key = f'base_0_{base_numclass - 1}'
    accuracy[base_key] = acc_old
    for i in range(base_numclass, total_classes):
        cls_range = [i, i + 1]
        acc = eval_acc(model, cls_range, test_dataset, device)
        accuracy[f'incre_{i}'] = acc
    test_dataset.getTestData([0, total_classes])
    model.train()
    return accuracy

def FedAvg_Fisher_weighted(fisher_list, weight_list=None):
    if weight_list is None:
        weight_list = [1.0 for _ in range(len(fisher_list))]
    out = {n: torch.zeros_like(fisher_list[0][n]) for n in fisher_list[0]}
    total = sum(weight_list) + 1e-12
    for fk, wk in zip(fisher_list, weight_list):
        pk = wk / total
        for n in out:
            out[n] += pk * fk[n]
    return out

def pick_ab_params_by_conflict(new_class_id, has_old, avg_old_model, w_new_mix, ref_model, new_clients_list, old_clients_list):
    n_new = len(new_clients_list)
    n_old = len(old_clients_list)
    Wk = 'fc.weight'
    conflict_cos = None
    if w_new_mix is not None and Wk in ref_model and (Wk in w_new_mix):
        dn = (w_new_mix[Wk][new_class_id] - ref_model[Wk][new_class_id]).norm().item()
    if has_old and avg_old_model is not None and (w_new_mix is not None) and (Wk in ref_model) and (Wk in avg_old_model) and (Wk in w_new_mix) and (new_class_id < ref_model[Wk].shape[0]) and (new_class_id < avg_old_model[Wk].shape[0]) and (new_class_id < w_new_mix[Wk].shape[0]):
        d_old = avg_old_model[Wk][new_class_id] - ref_model[Wk][new_class_id]
        d_new = w_new_mix[Wk][new_class_id] - ref_model[Wk][new_class_id]
        denom = float(d_old.norm().item() * d_new.norm().item())
        if denom > 1e-12:
            conflict_cos = float(torch.dot(d_old.flatten(), d_new.flatten()).item() / denom)
        else:
            conflict_cos = 0.0
    if not has_old:
        backbone_new_ratio = 1.0
    elif n_new <= 1:
        backbone_new_ratio = 0.25
    elif n_new <= 3:
        backbone_new_ratio = 0.15
    else:
        backbone_new_ratio = 0.1
    if not has_old:
        old_fc_keep = 0.0
    else:
        old_fc_keep = 0.9
        if n_new <= 1:
            old_fc_keep = 0.95
        if conflict_cos is not None and conflict_cos < -0.2:
            old_fc_keep = max(old_fc_keep, 0.95)
    if w_new_mix is None or Wk not in w_new_mix:
        mu = 0.0
    elif not has_old:
        mu = 1.0
    elif n_new <= 1:
        mu = 1.0
    elif n_new <= 3:
        mu = 0.45
    else:
        mu = 0.2
    if conflict_cos is not None:
        if conflict_cos < -0.5:
            mu = 1.0
        elif conflict_cos < -0.2:
            mu = max(mu, 0.8)
    return (backbone_new_ratio, old_fc_keep, mu, conflict_cos)

@torch.no_grad()
def eval_global_newclass_margin(model, test_dataset, new_class_id, device, max_batches=20):
    model.eval()
    margins = []
    neg_cnt = 0
    total = 0
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=args.batch_size)
    for b, batch in enumerate(test_loader):
        if b >= max_batches:
            break
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            _, images, labels = batch
        else:
            images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        mask = labels == new_class_id
        if mask.sum() == 0:
            continue
        logits = model(images)[mask]
        y = labels[mask]
        z_y = logits[:, new_class_id]
        logits_others = logits.clone()
        logits_others[:, new_class_id] = -1000000000.0
        z_max_other, _ = logits_others.max(dim=1)
        margin = z_y - z_max_other
        margins.append(margin.detach().cpu())
        neg_cnt += (margin < 0).sum().item()
        total += margin.numel()
    if total == 0:
        return {'margin_mean': 0.0, 'neg_rate': 0.0, 'count': 0}
    margins = torch.cat(margins)
    return {'margin_mean': float(margins.mean()), 'neg_rate': float(neg_cnt / total), 'count': int(total)}

def append_newcls_metrics_csv(csv_path, method, client_id, class_id, expo, global_round, window, acc_cls, auc_norm, slope, tta_dict, curve_pts):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ['method', 'client_id', 'class_id', 'expo', 'window', 'global_round_end', 'acc_end', 'auc_norm', 'slope', 'tta20', 'tta40', 'tta60', 'curve_json']
    row = {'method': method, 'client_id': client_id, 'class_id': class_id, 'expo': expo, 'window': window, 'global_round_end': global_round, 'acc_end': float(acc_cls), 'auc_norm': float(auc_norm), 'slope': float(slope), 'tta20': tta_dict.get(20.0, None), 'tta40': tta_dict.get(40.0, None), 'tta60': tta_dict.get(60.0, None), 'curve_json': json.dumps(curve_pts, ensure_ascii=False)}
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)
