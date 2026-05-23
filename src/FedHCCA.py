import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from myNetwork import *
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import random
from Fed_utils import *
from tqdm import tqdm
from collections import defaultdict

def get_one_hot(target, num_class, device, class_offset=0):
    one_hot = torch.zeros(target.shape[0], num_class).cuda(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.0)
    return one_hot

def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-05)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def zero_grad_newclass_head(model, new_class_id):
    if model.fc.weight.grad is not None:
        model.fc.weight.grad.data[new_class_id].zero_()
    if model.fc.bias.grad is not None:
        model.fc.bias.grad.data[new_class_id].zero_()

def freeze_except_fc(model):
    for name, p in model.named_parameters():
        if name.startswith('fc.') or '.fc.' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def freeze_bn_layers(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

class FedHCCA_model:

    def __init__(self, init_classes, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate, train_set, test_set, device, exemplar_per_class: int=20):
        super(FedHCCA_model, self).__init__()
        self.local_epochs = epochs
        self.learning_rate = learning_rate
        self.model = network(init_classes, feature_extractor)
        self.device = device
        self.exemplar_set = []
        self.exemplar_label_set = []
        self.class_mean_set = []
        self.init_classes = init_classes
        self.learned_numclass = init_classes
        self.total_classes = init_classes
        self.current_class = None
        self.last_class = None
        self.task_id_old = -1
        self.old_classes = [i for i in range(init_classes)]
        self.transform = getattr(train_set, 'exemplar_transform', transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]))
        self.old_model = None
        self.train_dataset = train_set
        self.test_dataset = test_set
        self.train_loader = None
        self.test_loader = None
        self.scheduler = None
        self.last_lr_stage = -1
        self.criterion = nn.CrossEntropyLoss()
        self.signal = False
        self.last_entropy = 0
        self.exposure_round_init = -1
        self.exposure_round_current = -1
        self.batchsize = batch_size
        self.memory_size = memory_size
        self.exemplar_per_class = int(exemplar_per_class)
        self.task_size = task_size
        self.proto_stage_threshold = 20
        self.lambda_max_proto = 0.5
        self.new_data_indices = None
        self.new_support = False
        self.recent_full_class = None
        self.recent_full_indices = None
        self.recent_full_expire_total_classes = None
        self.mu = 0.001
        self.ewc_batches = 20
        self.ewc_lambda = 50
        self.ewc_state = {'params': None, 'fisher': None}
        self.lambda_pod = 1.0
        self.newcls_curve = defaultdict(list)
        self.newcls_metrics = {}
        self.date = None
        self.method = None
        self.client_id = None
        self.newcls_csv_path = None

    def update_ewc_state(self, fisher, ref_params):
        self.ewc_state = {'fisher': fisher, 'params': ref_params}

    def _get_train_and_test_dataloader(self, train_classes, mix):
        if mix:
            self.train_dataset.getTrainData(train_classes, self.exemplar_set, self.exemplar_label_set)
        else:
            self.train_dataset.getTrainData(train_classes, [], [])
        train_loader = DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=self.batchsize, num_workers=8, pin_memory=True)
        return train_loader

    def train(self, g_round, task_global_round, model_old, client_id, class_id, client_prototypes=None, global_prototypes=None, lambda_proto=0.0, count_new_samples=0, is_exposed_train: bool=False, freeze_backbone: bool=False, freeze_bn_stats: bool=False, new_loss_weight: float=1.0, protect_new_head: bool=True, protect_recent_old_head: bool=True, recent_old_weight: float=0.0, freeze_past_new_rows_grad: bool=False):
        self.model = model_to_device(self.model, False, self.device)
        opt = self.optimizer
        if g_round % task_global_round == 0:
            self.old_model = model_old[0]
        if self.old_model != None:
            self.old_model = model_to_device(self.old_model, False, self.device)
            self.old_model.eval()
        T = 2.0
        class_num = class_id + 1
        for local_epoch in range(self.local_epochs):
            diag_batches = 0
            diag_batches_with_new = 0
            diag_batches_with_old = 0
            diag_sum_n_new = 0
            diag_sum_n_old = 0
            diag_sum_ce_old = 0.0
            diag_sum_ce_new = 0.0
            diag_sum_ce_loss = 0.0
            diag_sum_alpha_new = 0.0
            diag_sum_gamma = 0.0
            diag_sum_new_weight = 0.0
            diag_sum_kd_loss = 0.0
            diag_sum_kd_alpha = 0.0
            diag_sum_eff_kd = 0.0
            diag_sum_eff_kd_over_ce_old = 0.0
            diag_sum_eff_kd_over_ce_loss = 0.0
            current_lr = self.optimizer.param_groups[0]['lr']
            self.model.train()
            if freeze_backbone:
                freeze_except_fc(self.model)
            else:
                for p in self.model.parameters():
                    p.requires_grad = True
            if freeze_bn_stats and (not freeze_backbone):
                freeze_bn_layers(self.model)
            total_loss, total_ce, total_kd, total_proto = (0.0, 0.0, 0.0, 0.0)
            invalid_target_warned = False
            for step, (indexs, images, target) in enumerate(self.train_loader):
                try:
                    n_classes = int(self.model.fc.weight.shape[0])
                except Exception:
                    n_classes = None
                target = target.long()
                if n_classes is not None:
                    valid = (target >= 0) & (target < n_classes)
                    if not bool(torch.all(valid)) and (not invalid_target_warned):
                        invalid_target_warned = True
                        invalid_cnt = int((~valid).sum().item())
                        tmin = int(target.min().item()) if target.numel() > 0 else None
                        tmax = int(target.max().item()) if target.numel() > 0 else None
                    if int(valid.sum().item()) == 0:
                        continue
                    images = images[valid]
                    target = target[valid]
                images = images.to(self.device)
                target = target.to(self.device)
                opt.zero_grad()
                feats, outputs = self.model(images, return_features=True)
                if is_exposed_train:
                    mask_new = target == class_id
                else:
                    mask_new = torch.zeros_like(target, dtype=torch.bool)
                mask_old = ~mask_new
                n_new = int(mask_new.sum().item())
                n_old = int(mask_old.sum().item())
                diag_batches += 1
                diag_sum_n_new += n_new
                diag_sum_n_old += n_old
                if n_new > 0:
                    diag_batches_with_new += 1
                if n_old > 0:
                    diag_batches_with_old += 1
                recent_old = class_id - 1
                mask_ro = target == recent_old if recent_old >= 0 else torch.zeros_like(target, dtype=torch.bool)
                n_ro = int(mask_ro.sum().item())
                dev = outputs.device
                ce_loss = torch.tensor(0.0, device=dev)
                ce_old = torch.tensor(0.0, device=dev)
                ce_new = torch.tensor(0.0, device=dev)
                if mask_old.any():
                    logits_old = outputs[mask_old, :class_num].clone()
                    if protect_new_head and (not mask_new.any()) and (class_id >= 0) and (class_id < int(logits_old.shape[1])):
                        logits_old[:, int(class_id)] = -1000000000.0
                    if protect_recent_old_head and recent_old >= 0 and (not mask_ro.any()) and (recent_old < int(logits_old.shape[1])):
                        logits_old[:, int(recent_old)] = -1000000000.0
                    ce_old = self.criterion(logits_old, target[mask_old])
                if mask_new.any():
                    ce_new = self.criterion(outputs[mask_new, :class_num], target[mask_new])
                if n_new > 0:
                    alpha_new = min(0.3, max(0.05, n_new / float(n_old + n_new)))
                    gamma = min(2.5, 1.0 + float(self.proto_stage_threshold) / max(1.0, float(count_new_samples)))
                    gamma = gamma * float(max(1.0, new_loss_weight))
                    ce_loss = (1.0 - alpha_new) * ce_old + alpha_new * gamma * ce_new
                else:
                    alpha_new = 0.0
                    gamma = 0.0
                    ce_loss = ce_old
                try:
                    ce_old_f = float(ce_old.detach().item())
                except Exception:
                    ce_old_f = float(ce_old) if ce_old is not None else 0.0
                try:
                    ce_new_f = float(ce_new.detach().item())
                except Exception:
                    ce_new_f = float(ce_new) if ce_new is not None else 0.0
                try:
                    ce_loss_f = float(ce_loss.detach().item())
                except Exception:
                    ce_loss_f = float(ce_loss) if ce_loss is not None else 0.0
                diag_sum_ce_old += ce_old_f
                diag_sum_ce_new += ce_new_f
                diag_sum_ce_loss += ce_loss_f
                if n_new > 0:
                    diag_sum_alpha_new += float(alpha_new)
                    diag_sum_gamma += float(gamma)
                    diag_sum_new_weight += float(alpha_new) * float(gamma)
                if recent_old_weight and recent_old_weight > 0.0 and (recent_old >= 0) and mask_ro.any():
                    ce_ro = self.criterion(outputs[mask_ro, :class_num], target[mask_ro])
                    ce_loss = ce_loss + float(recent_old_weight) * ce_ro
                kd_loss = torch.tensor(0.0, device=dev)
                kd_alpha_base = 0.5
                ratio_global = min(1.0, count_new_samples / float(self.proto_stage_threshold))
                kd_alpha = kd_alpha_base * (0.5 + 0.5 * ratio_global)
                if self.old_model is not None and mask_old.any():
                    with torch.no_grad():
                        old_outputs = self.old_model(images[mask_old])
                    old_num = old_outputs.shape[1]
                    kd_loss = self.distillation_loss(outputs[mask_old, :old_num], target[mask_old], old_outputs, T=T)
                try:
                    kd_loss_f = float(kd_loss.detach().item())
                except Exception:
                    kd_loss_f = float(kd_loss) if kd_loss is not None else 0.0
                kd_alpha_f = float(kd_alpha)
                eff_kd_f = kd_alpha_f * kd_loss_f
                diag_sum_kd_loss += kd_loss_f
                diag_sum_kd_alpha += kd_alpha_f
                diag_sum_eff_kd += eff_kd_f
                eps = 1e-06
                diag_sum_eff_kd_over_ce_old += eff_kd_f / (ce_old_f + eps)
                diag_sum_eff_kd_over_ce_loss += eff_kd_f / (ce_loss_f + eps)
                kd_ro = torch.tensor(0.0, device=dev)
                kd_alpha_ro = 0.0
                if self.old_model is not None and recent_old >= 0 and mask_ro.any():
                    with torch.no_grad():
                        old_out_ro = self.old_model(images[mask_ro])
                    old_num_ro = old_out_ro.shape[1]
                    kd_ro = self.distillation_loss(outputs[mask_ro, :old_num_ro], target[mask_ro], old_out_ro, T=T)
                    kd_alpha_ro = max(kd_alpha, 1.0)
                proto_loss = torch.tensor(0.0, device=dev)
                if count_new_samples > 0 and lambda_proto > 0 and (client_prototypes is not None):
                    local_proto = client_prototypes.get(class_id, None)
                    if local_proto is not None:
                        global_proto = None
                        if global_prototypes is not None:
                            global_proto = global_prototypes.get(class_id, None)
                        proto_loss = self.proto_alignment_loss(outputs=feats, targets=target, class_id=class_id, local_prototype=local_proto, global_prototype=global_proto, count_new_samples=count_new_samples, sample_threshold=self.proto_stage_threshold)
                if class_id == int(self.init_classes):
                    loss = ce_loss + kd_alpha * kd_loss + lambda_proto * proto_loss
                else:
                    loss = ce_loss + kd_alpha * kd_loss + kd_alpha_ro * kd_ro + lambda_proto * proto_loss
                loss.backward()
                if freeze_past_new_rows_grad and is_exposed_train and (class_id > int(self.init_classes)):
                    gw = self.model.fc.weight.grad
                    gb = self.model.fc.bias.grad if getattr(self.model.fc, 'bias', None) is not None else None
                    if gw is not None:
                        start = int(self.init_classes)
                        end = int(class_id)
                        if end > start:
                            gw[start:end].zero_()
                            if gb is not None:
                                gb[start:end].zero_()
                if not self.new_support:
                    zero_grad_newclass_head(self.model, class_id)
                with torch.no_grad():
                    gw = self.model.fc.weight.grad
                    gb = self.model.fc.bias.grad if getattr(self.model.fc, 'bias', None) is not None else None
                    if gw is not None:
                        db_val = gb[class_id].abs().item() if gb is not None else 0.0
                    else:
                        pass
                opt.step()
                total_loss += float(loss.item())
                total_ce += float(ce_loss.item())
                total_kd += float(kd_loss.item())
                total_proto += float(proto_loss.item())
                if (target == class_id).any():
                    with torch.no_grad():
                        ce_val = ce_loss.detach().item()
                        kd_val = kd_loss.detach().item() if kd_loss is not None else 0.0
            denom = max(1, len(self.train_loader))
            diag_denom = max(1, diag_batches)
            diag_new_denom = max(1, diag_batches_with_new)
            diag_avg_n_new = diag_sum_n_new / float(diag_denom)
            diag_avg_n_old = diag_sum_n_old / float(diag_denom)
            diag_avg_ce_old = diag_sum_ce_old / float(diag_denom)
            diag_avg_ce_new = diag_sum_ce_new / float(diag_denom)
            diag_avg_ce_loss = diag_sum_ce_loss / float(diag_denom)
            diag_avg_alpha_new = diag_sum_alpha_new / float(diag_new_denom)
            diag_avg_gamma = diag_sum_gamma / float(diag_new_denom)
            diag_avg_new_weight = diag_sum_new_weight / float(diag_new_denom)
            diag_avg_kd_loss = diag_sum_kd_loss / float(diag_denom)
            diag_avg_kd_alpha = diag_sum_kd_alpha / float(diag_denom)
            diag_avg_eff_kd = diag_sum_eff_kd / float(diag_denom)
            diag_avg_eff_kd_over_ce_old = diag_sum_eff_kd_over_ce_old / float(diag_denom)
            diag_avg_eff_kd_over_ce_loss = diag_sum_eff_kd_over_ce_loss / float(diag_denom)
            if self.scheduler is not None:
                self.scheduler.step()
            self.model.eval()
            correct, total = (0, 0)
            pred_count_all = torch.zeros(class_num, dtype=torch.long)
            label_count_all = torch.zeros(class_num, dtype=torch.long)
            new_total = 0
            new_neg = 0
            new_margin_sum = 0.0
            with torch.no_grad():
                for step, (indexs, images, labels) in enumerate(self.test_loader):
                    images, labels = (images.to(self.device), labels.to(self.device))
                    feats, logits = self.model(images, return_features=True)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    pred_count_all += torch.bincount(preds.detach().cpu(), minlength=class_num)
                    label_count_all += torch.bincount(labels.detach().cpu(), minlength=class_num)
                    new_mask = labels == class_id
                    if new_mask.any():
                        z_new = logits[new_mask, class_id]
                        if class_id > 0:
                            z_old_max = logits[new_mask][:, :class_id].max(dim=1)[0]
                            margin = z_new - z_old_max
                        else:
                            margin = z_new
                        new_total += int(margin.numel())
                        new_neg += int((margin < 0).sum().item())
                        new_margin_sum += float(margin.sum().item())
            acc = 100 * correct / total
            if new_total > 0:
                margin_mean = new_margin_sum / float(new_total)
                neg_rate = new_neg / float(new_total)
            recent_old = class_id - 1
            if recent_old >= 0:
                K = 10
                seen_ro = 0
                seen_total = 0
                for bi, (idxs, imgs, lbs) in enumerate(self.train_loader):
                    lbs = lbs.detach().cpu()
                    seen_ro += int((lbs == recent_old).sum().item())
                    seen_total += int(lbs.numel())
                    if bi + 1 >= K:
                        break
                ratio_ro = seen_ro / max(1, seen_total)
                correct_ro, total_ro = (0, 0)
                pred_as_new_ro = 0
                correct_ro_oldonly = 0
                with torch.no_grad():
                    for step, (indexs, images, labels) in enumerate(self.test_loader):
                        images, labels = (images.to(self.device), labels.to(self.device))
                        out = self.model(images, return_maps=False)
                        outputs = out[0] if isinstance(out, (tuple, list)) else out
                        preds = torch.argmax(outputs, dim=1)
                        mask = labels == recent_old
                        if mask.any():
                            correct_ro += (preds[mask] == labels[mask]).sum().item()
                            pred_as_new_ro += (preds[mask] == class_id).sum().item()
                            oldonly_preds = torch.argmax(outputs[mask, :class_id], dim=1)
                            correct_ro_oldonly += (oldonly_preds == labels[mask]).sum().item()
                            total_ro += mask.sum().item()
                acc_ro = 100.0 * correct_ro / max(1, total_ro)
                pred_as_new_rate_ro = 100.0 * float(pred_as_new_ro) / max(1, total_ro)
                acc_ro_oldonly = 100.0 * float(correct_ro_oldonly) / max(1, total_ro)
            if int(label_count_all.sum().item()) > 0:
                label_hist = (label_count_all.float() / label_count_all.sum().float()).cpu().numpy()
            if int(pred_count_all.sum().item()) > 0:
                pred_hist = (pred_count_all.float() / pred_count_all.sum().float()).cpu().numpy()

    def train_compare(self, global_round, task_global_round, model_old, client_id, model_g, aggregator, IL_method, ewc_pack):
        self.model = model_to_device(self.model, False, self.device)
        class_num = int(self.model.fc.out_features)
        opt = self.optimizer
        global_params = [p.detach() for p in model_g.model.parameters()]
        ref = None if ewc_pack is None else ewc_pack.get('params', None)
        fis = None if ewc_pack is None else ewc_pack.get('fisher', None)
        if global_round % task_global_round == 0:
            self.old_model = model_old[0]
        if self.old_model != None:
            self.old_model = model_to_device(self.old_model, False, self.device)
            self.old_model.eval()
        lambda_pod = self.lambda_pod
        sum_ce = 0.0
        sum_pod = 0.0
        sum_ewc = 0.0
        sum_prox = 0.0
        for local_epoch in range(self.local_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            total_loss = 0
            self.model.train()
            invalid_target_warned = False
            for step, (indexs, images, labels) in enumerate(self.train_loader):
                if local_epoch == 0 and step == 0:
                    fc_out_dim = int(self.model.fc.weight.data.shape[0])
                    monitor_idx = min(10, fc_out_dim - 1) if fc_out_dim > 0 else None
                    w_monitor_before = None if monitor_idx is None else self.model.fc.weight.data[monitor_idx].clone()
                try:
                    n_classes = int(self.model.fc.weight.data.shape[0])
                except Exception:
                    n_classes = None
                labels = labels.long()
                if n_classes is not None:
                    valid = (labels >= 0) & (labels < n_classes)
                    if not bool(torch.all(valid)) and (not invalid_target_warned):
                        invalid_target_warned = True
                        invalid_cnt = int((~valid).sum().item())
                        tmin = int(labels.min().item()) if labels.numel() > 0 else None
                        tmax = int(labels.max().item()) if labels.numel() > 0 else None
                    if int(valid.sum().item()) == 0:
                        continue
                    images = images[valid]
                    labels = labels[valid]
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, maps_new = self.model(images, return_maps=True)
                loss_ce = self.criterion(outputs, labels)
                loss = loss_ce
                loss_pod = torch.tensor(0.0, device=self.device)
                if IL_method == 'PODNet':
                    if self.old_model is not None and lambda_pod > 0 and (maps_new is not None):
                        with torch.no_grad():
                            _, maps_old = self.old_model(images, return_maps=True)
                        lp = self.pod_loss(maps_new, maps_old)
                        loss_pod = lp if torch.is_tensor(lp) else torch.tensor(lp, device=self.device, dtype=loss.dtype)
                        loss += lambda_pod * loss_pod
                loss_prox = torch.tensor(0.0).to(self.device)
                if aggregator == 'fedprox':
                    prox_denom = 0
                    for p, gp in zip(self.model.parameters(), global_params):
                        diff2 = (p - gp).pow(2)
                        loss_prox += diff2.sum()
                        prox_denom += int(diff2.numel())
                    if prox_denom > 0:
                        loss_prox = loss_prox / float(prox_denom)
                    loss = loss + 0.5 * self.mu * loss_prox
                loss_ewc = torch.tensor(0.0).to(self.device)
                if IL_method == 'EWC':
                    loss_ewc = self.ewc_penalty(self.model, ref, fis)
                    loss = loss + 0.5 * self.ewc_lambda * loss_ewc
                loss.backward()
                self.optimizer.step()
                if local_epoch == 0 and step == 0:
                    if monitor_idx is not None:
                        delta = (self.model.fc.weight.data[monitor_idx] - w_monitor_before).abs().mean().item()
                total_loss += loss.item()
                sum_ce += loss_ce.item()
                sum_pod += loss_pod.item()
                sum_prox += loss_prox.item()
                sum_ewc += loss_ewc.item()
            if local_epoch % 10 == 0 or local_epoch == self.local_epochs - 1:
                pass
            self.scheduler.step()
            self.model.eval()
            correct, total = (0, 0)
            pred_count_all = torch.zeros(class_num, dtype=torch.long)
            label_count_all = torch.zeros(class_num, dtype=torch.long)
            with torch.no_grad():
                for step, (indexs, images, labels) in enumerate(self.test_loader):
                    images, labels = (images.to(self.device), labels.to(self.device))
                    out = self.model(images, return_maps=False)
                    outputs = out[0] if isinstance(out, (tuple, list)) else out
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    pred_count_all += torch.bincount(preds.detach().cpu(), minlength=class_num)
                    label_count_all += torch.bincount(labels.detach().cpu(), minlength=class_num)
            acc = 100 * correct / total
            ce_val = float(sum_ce / len(self.test_loader)) if not hasattr(sum_ce, 'item') else float(sum_ce.item())
            prox_val = float(sum_prox / len(self.test_loader)) if not hasattr(sum_prox, 'item') else float(sum_prox.item())
            ewc_val = float(sum_ewc / len(self.test_loader)) if not hasattr(sum_ewc, 'item') else float(sum_ewc.item())
            pod_val = float(sum_pod / len(self.test_loader)) if not hasattr(sum_pod, 'item') else float(sum_pod.item())
            if int(label_count_all.sum().item()) > 0:
                label_hist = (label_count_all.float() / label_count_all.sum().float()).cpu().numpy()
            if int(pred_count_all.sum().item()) > 0:
                pred_hist = (pred_count_all.float() / pred_count_all.sum().float()).cpu().numpy()

    def cross_entropy_loss(self, output, target):
        return F.cross_entropy(output, target)

    def distillation_loss(self, output, target, old_output, T=2):
        output_soft = F.log_softmax(output / T, dim=1)
        old_output_soft = F.softmax(old_output / T, dim=1)
        loss_KD = F.kl_div(output_soft, old_output_soft.detach(), reduction='batchmean') * (T * T)
        return loss_KD

    def proto_alignment_loss(self, outputs, targets, class_id, local_prototype, global_prototype, count_new_samples, sample_threshold=20):
        proto_loss = torch.tensor(0.0, device=self.device)
        if count_new_samples > sample_threshold:
            return proto_loss

        def to_tensor(x):
            if x is None:
                return None
            if not torch.is_tensor(x):
                return torch.tensor(x, device=self.device, dtype=outputs.dtype)
            return x.to(self.device, dtype=outputs.dtype)
        local_prototype = to_tensor(local_prototype)
        global_prototype = to_tensor(global_prototype)
        if local_prototype is None:
            return proto_loss
        if global_prototype is None:
            proto = local_prototype
        else:
            alpha = min(1.0, count_new_samples / float(self.proto_stage_threshold))
            proto = alpha * global_prototype + (1 - alpha) * local_prototype
        mask = targets == class_id
        if not mask.any():
            return proto_loss
        feats_c = outputs[mask]
        feats_c = torch.nn.functional.normalize(feats_c, dim=1)
        proto = torch.nn.functional.normalize(proto, dim=0)
        proto_loss = ((feats_c - proto.unsqueeze(0)) ** 2).sum(dim=1).mean()
        return proto_loss

    def ewc_penalty(self, model, ref_params, fisher):
        if ref_params is None or fisher is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        denom = 0
        for name, p in model.named_parameters():
            if name in fisher and name in ref_params and p.requires_grad:
                f = fisher[name].to(device=p.device, dtype=p.dtype)
                r = ref_params[name].to(device=p.device, dtype=p.dtype)
                f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
                r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
                if p.shape == r.shape == f.shape:
                    term = f * (p - r).pow(2)
                    loss = loss + term.sum()
                    denom += int(term.numel())
                elif p.ndim == r.ndim == f.ndim and p.shape[0] >= r.shape[0] and (f.shape[0] >= r.shape[0]) and (p.shape[1:] == r.shape[1:] == f.shape[1:]):
                    p0 = p[:r.shape[0]]
                    f0 = f[:r.shape[0]]
                    term = f0 * (p0 - r).pow(2)
                    loss = loss + term.sum()
                    denom += int(term.numel())
                else:
                    continue
        if denom <= 0:
            return torch.tensor(0.0, device=loss.device)
        return loss / float(denom)

    def _init_fisher_like(self, model):
        fisher = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p, device=p.device)
        return fisher

    def estimate_fisher_diag(self, data_loader, max_batches=20):
        self.model.eval()
        fisher = self._init_fisher_like(self.model)
        count = 0
        invalid_target_total = 0
        invalid_target_batches = 0
        for batch_idx, (idx, x, y) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            x, y = (x.to(self.device), y.to(self.device))
            self.model.zero_grad()
            out = self.model(x)
            if out.ndim < 2:
                continue
            n_classes = int(out.shape[1])
            y = y.long()
            valid = (y >= 0) & (y < n_classes)
            if not torch.all(valid):
                invalid_target_total += int((~valid).sum().item())
                invalid_target_batches += 1
            if int(valid.sum().item()) == 0:
                continue
            loss = self.criterion(out[valid], y[valid])
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach().pow(2)
            count += 1
        for n in fisher:
            fisher[n] /= max(1, count)
        if invalid_target_total > 0:
            pass
        return fisher

    def snapshot_params(self):
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def pod_feature(self, fm):
        ch = fm.mean(dim=(2, 3))
        sp = fm.mean(dim=1).flatten(1)
        return torch.cat([ch, sp], dim=1)

    def pod_loss(self, maps_new, maps_old):
        if maps_new is None or maps_old is None:
            return 0.0
        loss = 0.0
        for fn, fo in zip(maps_new, maps_old):
            pn = F.normalize(self.pod_feature(fn), dim=1)
            po = F.normalize(self.pod_feature(fo), dim=1)
            loss = loss + F.mse_loss(pn, po)
        return loss / len(maps_new)

    def contrastive_loss(self, output, old_output):
        contrastive_loss = F.cosine_embedding_loss(output, old_output, torch.ones(output.size(0)).cuda(self.device))
        return contrastive_loss

    def Image_transform(self, images, transform):

        def _to_pil(img):
            if isinstance(img, str):
                return Image.open(img).convert('RGB')
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            img = np.asarray(img)
            if img.ndim == 3 and img.shape[0] in (1, 3) and (img.shape[1] > 1) and (img.shape[2] > 1):
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]
            if img.dtype != np.uint8:
                if np.issubdtype(img.dtype, np.floating):
                    img = np.clip(img, 0.0, 1.0)
                    img = (img * 255.0).round().astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            pil = Image.fromarray(img)
            return pil.convert('RGB') if pil.mode != 'RGB' else pil
        data = transform(_to_pil(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, transform(_to_pil(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        batch_size = 32
        features = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch_images = images[start:start + batch_size]
                x = self.Image_transform(batch_images, transform).to(self.device, non_blocking=True)
                batch_feats = self.model.feature_extractor(x)
                batch_feats = F.normalize(batch_feats.detach(), dim=1).cpu().numpy()
                features.append(batch_feats)
                del x, batch_feats
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        feature_extractor_output = np.concatenate(features, axis=0) if len(features) > 0 else np.empty((0, 0), dtype=np.float32)
        class_mean = np.mean(feature_extractor_output, axis=0)
        return (class_mean, feature_extractor_output)

    def compute_local_prototypes(self, client_id, class_id, client_prototypes, client_weights, new_class_indices=None):
        self.model.eval()
        with torch.no_grad():
            if self.total_classes == self.init_classes:
                class_counts = {}
                for _, images, labels in self.train_loader:
                    images, labels = (images.to(self.device), labels.to(self.device))
                    feats = self.model.feature_extractor(images)
                    feats = F.normalize(feats, dim=1)
                    for i in range(labels.size(0)):
                        lb = int(labels[i].item())
                        class_counts[lb] = class_counts.get(lb, 0) + 1
                        if lb not in client_prototypes[client_id] or not isinstance(client_prototypes[client_id][lb], list):
                            client_prototypes[client_id][lb] = []
                        client_prototypes[client_id][lb].append(feats[i].detach().cpu().numpy())
                for lb in list(client_prototypes[client_id].keys()):
                    if not isinstance(client_prototypes[client_id][lb], list):
                        continue
                    feats_np = np.asarray(client_prototypes[client_id][lb])
                    if feats_np.size == 0:
                        continue
                    proto = feats_np.mean(axis=0)
                    proto = proto / (np.linalg.norm(proto) + 1e-12)
                    client_prototypes[client_id][lb] = proto
                for lb, cnt in class_counts.items():
                    client_weights[client_id][lb] = int(cnt)
                self.model.train()
                return
            if new_class_indices is None or len(new_class_indices) == 0:
                self.model.train()
                return
            new_imgs = self.train_dataset.data[new_class_indices]
            x = self.Image_transform(new_imgs, self.transform).to(self.device)
            new_feats = self.model.feature_extractor(x)
            new_feats = F.normalize(new_feats, dim=1).detach().cpu().numpy()
            proto = new_feats.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            client_prototypes[client_id][class_id] = proto
            client_weights[client_id][class_id] = int(len(new_class_indices))
        self.model.train()

    def _find_exemplar_class_idx(self, class_id: int):
        for i, lb_block in enumerate(self.exemplar_label_set):
            if len(lb_block) > 0 and int(lb_block[0]) == int(class_id):
                return i
        return -1

    def _construct_exemplar_set(self, images, class_id, m=20):
        exemplar = []
        num_existing_samples = len(images)
        if num_existing_samples == 0:
            return
        elif num_existing_samples <= m:
            exemplar = list(images)
        else:
            class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
            now_class_mean = np.zeros((1, 512))
            for i in range(m):
                x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
                x = np.linalg.norm(x, axis=1)
                index = np.argmin(x)
                now_class_mean += feature_extractor_output[index]
                exemplar.append(images[index])
        self.exemplar_set.append(exemplar)
        self.exemplar_label_set.append(np.array([class_id] * len(exemplar)))

    def construct_exemplar_set(self, images, class_id, global_prototypes, m=20):
        num_existing_samples = len(images)
        exemplar = []
        if num_existing_samples == 0:
            return
        elif num_existing_samples <= m:
            exemplar = list(images)
        else:
            class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
            global_prototype = np.array(global_prototypes[class_id], dtype=float)
            if global_prototype.ndim > 1:
                global_prototype = np.mean(global_prototype, axis=0)
            global_prototype = global_prototype.reshape(1, -1)
            feat_dim = feature_extractor_output.shape[1]
            now_class_sum = np.zeros((1, feat_dim), dtype=feature_extractor_output.dtype)
            for i in range(m):
                x = global_prototype - (now_class_sum + feature_extractor_output) / (i + 1)
                x = np.linalg.norm(x, axis=1)
                index = np.argmin(x)
                now_class_sum += feature_extractor_output[index]
                exemplar.append(images[index])
                feature_extractor_output = np.delete(feature_extractor_output, index, axis=0)
                images = np.delete(images, index, axis=0)
        idx = self._find_exemplar_class_idx(class_id)
        label_block = np.array([class_id] * len(exemplar))
        if idx == -1:
            self.exemplar_set.append(exemplar)
            self.exemplar_label_set.append(label_block)
        else:
            self.exemplar_set[idx] = exemplar
            self.exemplar_label_set[idx] = label_block

    def exemplar_update_set(self, global_prototypes, global_round, new_class_indices=None, new_class_id=None, IL_method=None):
        if self.total_classes == self.init_classes:
            for class_id in range(self.init_classes):
                class_datas, class_labels = self.train_dataset.get_class_data(class_id)
                self.construct_exemplar_set(class_datas, class_id, global_prototypes, m=self.exemplar_per_class)
        if new_class_id is None:
            new_class_id = self.current_class
        if new_class_indices is None or len(new_class_indices) == 0:
            pass
        else:
            new_class_datas = self.train_dataset.data[new_class_indices]
            if IL_method == 'iCaRL':
                self._construct_exemplar_set(new_class_datas, new_class_id, m=self.exemplar_per_class)
            else:
                self.construct_exemplar_set(new_class_datas, new_class_id, global_prototypes, m=self.exemplar_per_class)
        total_exemplars = sum((len(exemplar) for exemplar in self.exemplar_set))
