from torchvision.datasets import CIFAR100, CIFAR10, SVHN
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

class iCIFAR100(CIFAR100):

    def __init__(self, root, train=True, transform=None, target_transform=None, test_transform=None, target_test_transform=None, download=False):
        super(iCIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.target_test_transform = target_test_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return (con_data, con_label)

    def getTestData(self, classes):
        datas, labels = ([], [])
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full(data.shape[0], label))
        self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def getTrainData(self, classes, new_class_id, exemplar_set, exemplar_label_set, new_class_indices=None):
        datas, labels = ([], [])
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            if new_class_id >= 1 and new_class_indices is not None:
                filtered_exemplar_set = []
                filtered_exemplar_label_set = []
                for data, label in zip(exemplar_set, exemplar_label_set):
                    lab = np.asarray(label)
                    if lab.ndim == 0:
                        if int(lab) != new_class_id:
                            filtered_exemplar_set.append(data)
                            filtered_exemplar_label_set.append(int(lab))
                        continue
                    mask = lab != new_class_id
                    if not np.any(mask):
                        continue
                    dat = np.asarray(data)
                    if dat.ndim >= 1 and dat.shape[0] == lab.shape[0]:
                        filtered_exemplar_set.append(dat[mask])
                        filtered_exemplar_label_set.append(lab[mask])
                    else:
                        filtered_exemplar_set.append(dat)
                        filtered_exemplar_label_set.append(lab)
                datas.extend(filtered_exemplar_set)
                labels.extend(filtered_exemplar_label_set)
            else:
                datas.extend(exemplar_set)
                labels.extend(exemplar_label_set)
        if new_class_id >= 1:
            if new_class_indices is None:
                self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
                return
            new_data = self.data[new_class_indices]
            datas.append(new_data)
            labels.append(np.full(len(new_class_indices), new_class_id))
        if new_class_id == -1:
            for label in classes:
                data = self.data[np.array(self.targets) == label]
                datas.append(data)
                labels.append(np.full(data.shape[0], label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def dirichlet_split(self, data, targets, class_range, num_clients=10, alpha=0.5, overlap=False, min_per_client=5, rng=None, seed=None):
        if rng is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random
        client_data = [[] for _ in range(num_clients)]
        client_targets = [[] for _ in range(num_clients)]
        for label in range(class_range[0], class_range[1]):
            idx = np.where(np.array(targets) == label)[0]
            n = len(idx)
            proportions = rng.dirichlet([alpha] * num_clients)
            if not overlap:
                proportions = (proportions * n).astype(int)
                for i in range(num_clients):
                    if proportions[i] < min_per_client:
                        proportions[i] = min_per_client
                while proportions.sum() < n:
                    proportions[np.argmax(proportions)] += 1
                while proportions.sum() > n:
                    max_idx = np.argmax(proportions)
                    if proportions[max_idx] > min_per_client:
                        proportions[max_idx] -= 1
                    else:
                        for j in range(num_clients):
                            if proportions[j] > min_per_client:
                                proportions[j] -= 1
                                break
                    proportions[np.argmax(proportions)] -= 1
                rng.shuffle(idx)
                start = 0
                for i, count in enumerate(proportions):
                    client_data[i].extend(data[idx[start:start + count]])
                    client_targets[i].extend([label] * count)
                    start += count
            else:
                scale = 10
                for i, p in enumerate(proportions):
                    sample_size = max(int(p * scale * n), min_per_client)
                    sampled_idx = rng.choice(idx, sample_size, replace=True)
                    client_data[i].extend(data[sampled_idx])
                    client_targets[i].extend([label] * sample_size)
        client_data = [np.array(d) for d in client_data]
        client_targets = [np.array(t) for t in client_targets]
        return (client_data, client_targets)

    def dirichlet_split_indices(self, targets, class_range, num_clients=10, alpha=0.5, overlap=False, min_per_client=5, max_per_client=None, rng=None, seed=None):
        if rng is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random
        targets = np.array(targets)
        client_data_indices = [[] for _ in range(num_clients)]
        for label in range(class_range[0], class_range[1]):
            idx = np.where(np.array(targets) == label)[0]
            rng.shuffle(idx)
            n = len(idx)
            proportions = rng.dirichlet([alpha] * num_clients)
            proportions = np.clip(proportions, 1e-06, 1)
            proportions = proportions / proportions.sum()
            if not overlap:
                splits = (proportions * n).astype(int)
                for i in range(num_clients):
                    if splits[i] < min_per_client:
                        splits[i] = min_per_client
                while splits.sum() > n:
                    max_idx = np.argmax(splits)
                    if splits[max_idx] > min_per_client:
                        splits[max_idx] -= 1
                    else:
                        break
                surplus_pool = []
                if max_per_client is not None:
                    for i in range(num_clients):
                        if splits[i] > max_per_client:
                            surplus = splits[i] - max_per_client
                            splits[i] = max_per_client
                            surplus_pool.extend(idx[:surplus])
                            idx = idx[surplus:]
                start = 0
                for i, count in enumerate(splits):
                    client_data_indices[i].extend(idx[start:start + count])
                    start += count
                if surplus_pool:
                    sample_counts = [len(client_data_indices[i]) for i in range(num_clients)]
                    least_clients = np.argsort(sample_counts)[:5]
                    surplus_proportions = rng.dirichlet([alpha] * 5)
                    surplus_proportions = (surplus_proportions / surplus_proportions.sum() * len(surplus_pool)).astype(int)
                    start = 0
                    for i, cid in enumerate(least_clients):
                        end = start + surplus_proportions[i]
                        client_data_indices[cid].extend(surplus_pool[start:end])
                        start = end
                    for i in range(start, len(surplus_pool)):
                        client_data_indices[least_clients[i % 5]].append(surplus_pool[i])
            else:
                scale = 10
                for i, p in enumerate(proportions):
                    sample_size = max(int(p * scale * n), min_per_client)
                    sampled_idx = rng.choice(idx, size=sample_size, replace=True)
                    client_data_indices[i].extend(sampled_idx.tolist())
        for i in range(num_clients):
            pass
        return client_data_indices

    def getSampleData(self, classes, exemplar_set, exemplar_label_set, group):
        datas, labels = ([], [])
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full(length, label) for label in exemplar_label_set]
        if group == 0:
            for label in classes:
                data = self.data[np.array(self.targets) == label]
                datas.append(data)
                labels.append(np.full(data.shape[0], label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def getTrainItem(self, index):
        img, target = (Image.fromarray(self.TrainData[index]), self.TrainLabels[index])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return (index, img, target)

    def getTestItem(self, index):
        img, target = (Image.fromarray(self.TestData[index]), self.TestLabels[index])
        if self.test_transform:
            img = self.test_transform(img)
        if self.target_test_transform:
            target = self.target_test_transform(target)
        return (index, img, target)

    def getTrainItem_indices(self, indices):
        subset_data = self.TrainData[indices]
        subset_labels = self.TrainLabels[indices]
        subset_dataset = iCIFAR100(root=self.root, train=True, transform=self.transform, target_transform=self.target_transform, download=False)
        subset_dataset.TrainData = subset_data
        subset_dataset.TrainLabels = subset_labels
        return subset_dataset

    def getTrainImg_indices(self, indices):
        subset_data = self.TrainData[indices]
        subset_labels = self.TrainLabels[indices]
        return (subset_data, subset_labels)

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if len(self.TrainData) > 0:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

    def distribution_heatmap(self, targets, client_data_indices, save_path, num_clients=10, class_range=[0, 100]):
        distribution_matrix = np.zeros((num_clients, class_range[1] - class_range[0]), dtype=int)
        for client_id in range(num_clients):
            labels = targets[client_data_indices[client_id]]
            for class_id in range(class_range[0], class_range[1]):
                cls_idx = class_id - class_range[0]
                distribution_matrix[client_id, cls_idx] = np.sum(labels == class_id)
        plt.figure(figsize=(24, 16))
        sns.heatmap(distribution_matrix, annot=False, fmt='d', cmap='YlGnBu', cbar=True, linewidths=0.5)
        plt.title('Client Data Distribution Heatmap')
        plt.xlabel('Classes')
        plt.ylabel('Clients')
        plt.savefig(save_path)
        plt.show()

    def incre_split_datasets(self, targets, class_id, new_clients, num_clients, incre_sample_splits, alpha=0.5, min_per_client=3, rng=None, seed=None):
        if rng is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random
        targets = np.array(targets)
        client_indices_dict = {f'client_index{index}': {'round_0': [], 'round_1': [], 'round_2': [], 'round_3': []} for index in range(num_clients)}
        client_data_indices = [[] for _ in range(num_clients)]
        idx = np.where(np.array(targets) == class_id)[0]
        rng.shuffle(idx)
        for i in range(new_clients):
            start = 0
            idx_temp = idx[i * 100:(i + 1) * 100]
            client_data_indices[i].extend(idx_temp)
            j = 0
            for j, size in enumerate(incre_sample_splits):
                client_indices_dict[f'client_index{i}'][f'round_{j}'] = idx_temp[start:start + size]
                start += size
        old_clients = num_clients - new_clients
        idx_remain = idx[400:]
        proportions = rng.dirichlet([alpha] * old_clients)
        proportions = np.clip(proportions, 1e-06, 1)
        proportions = proportions / proportions.sum()
        proportions = (proportions * 100).astype(int)
        for i in range(old_clients):
            if proportions[i] < min_per_client:
                proportions[i] = min_per_client
            while proportions.sum() > 100:
                max_idx = np.argmax(proportions)
                if proportions[max_idx] > min_per_client:
                    proportions[max_idx] -= 1
                else:
                    break
        start = 0
        for i, count in enumerate(proportions):
            client_indices_dict[f'client_index{new_clients + i}']['round_0'] = idx_remain[start:start + count]
            client_data_indices[new_clients + i].extend(idx_remain[start:start + count])
            start += count
        return (client_indices_dict, client_data_indices)

    def get_class_data(self, class_id):
        datas = self.TrainData[np.array(self.TrainLabels) == class_id]
        labels = self.TrainLabels[np.array(self.TrainLabels) == class_id]
        return (datas, labels)

class iCIFAR10(CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None, test_transform=None, target_test_transform=None, download=False, train_per_class: int=-1, seed: int=2021):
        super(iCIFAR10, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if bool(train) and int(train_per_class) and (int(train_per_class) > 0):
            rng = np.random.default_rng(int(seed))
            targets = np.asarray(self.targets)
            kept_indices = []
            for cls in range(10):
                cls_idx = np.where(targets == cls)[0]
                if cls_idx.size == 0:
                    continue
                if cls_idx.size <= int(train_per_class):
                    chosen = cls_idx
                else:
                    chosen = rng.choice(cls_idx, size=int(train_per_class), replace=False)
                kept_indices.extend([int(x) for x in chosen])
            kept_indices = np.asarray(kept_indices, dtype=int)
            rng.shuffle(kept_indices)
            self.data = self.data[kept_indices]
            self.targets = targets[kept_indices].tolist()
        self.target_test_transform = target_test_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return (con_data, con_label)

    def getTestData(self, classes):
        datas, labels = ([], [])
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full(data.shape[0], label))
        self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def getTrainData(self, classes, new_class_id, exemplar_set, exemplar_label_set, new_class_indices=None):
        datas, labels = ([], [])
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            if new_class_id >= 1 and new_class_indices is not None:
                filtered_exemplar_set = []
                filtered_exemplar_label_set = []
                for data, label in zip(exemplar_set, exemplar_label_set):
                    lab = np.asarray(label)
                    if lab.ndim == 0:
                        if int(lab) != new_class_id:
                            filtered_exemplar_set.append(data)
                            filtered_exemplar_label_set.append(int(lab))
                        continue
                    mask = lab != new_class_id
                    if not np.any(mask):
                        continue
                    dat = np.asarray(data)
                    if dat.ndim >= 1 and dat.shape[0] == lab.shape[0]:
                        filtered_exemplar_set.append(dat[mask])
                        filtered_exemplar_label_set.append(lab[mask])
                    else:
                        filtered_exemplar_set.append(dat)
                        filtered_exemplar_label_set.append(lab)
                datas.extend(filtered_exemplar_set)
                labels.extend(filtered_exemplar_label_set)
            else:
                datas.extend(exemplar_set)
                labels.extend(exemplar_label_set)
        if new_class_id >= 1:
            if new_class_indices is None:
                self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
                return
            new_data = self.data[new_class_indices]
            datas.append(new_data)
            labels.append(np.full(len(new_class_indices), new_class_id))
        if new_class_id == -1:
            for label in classes:
                data = self.data[np.array(self.targets) == label]
                datas.append(data)
                labels.append(np.full(data.shape[0], label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def dirichlet_split(self, data, targets, class_range, num_clients=10, alpha=0.5, overlap=False, min_per_client=5, rng=None, seed=None):
        if rng is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random
        client_data = [[] for _ in range(num_clients)]
        client_targets = [[] for _ in range(num_clients)]
        for label in range(class_range[0], class_range[1]):
            idx = np.where(np.array(targets) == label)[0]
            n = len(idx)
            proportions = rng.dirichlet([alpha] * num_clients)
            if not overlap:
                proportions = (proportions * n).astype(int)
                for i in range(num_clients):
                    if proportions[i] < min_per_client:
                        proportions[i] = min_per_client
                while proportions.sum() < n:
                    proportions[np.argmax(proportions)] += 1
                while proportions.sum() > n:
                    max_idx = np.argmax(proportions)
                    if proportions[max_idx] > min_per_client:
                        proportions[max_idx] -= 1
                    else:
                        for j in range(num_clients):
                            if proportions[j] > min_per_client:
                                proportions[j] -= 1
                                break
                    proportions[np.argmax(proportions)] -= 1
                rng.shuffle(idx)
                start = 0
                for i, count in enumerate(proportions):
                    client_data[i].extend(data[idx[start:start + count]])
                    client_targets[i].extend([label] * count)
                    start += count
            else:
                scale = 10
                for i, p in enumerate(proportions):
                    sample_size = max(int(p * scale * n), min_per_client)
                    sampled_idx = rng.choice(idx, sample_size, replace=True)
                    client_data[i].extend(data[sampled_idx])
                    client_targets[i].extend([label] * sample_size)
        client_data = [np.array(d) for d in client_data]
        client_targets = [np.array(t) for t in client_targets]
        return (client_data, client_targets)

    def dirichlet_split_indices(self, targets, class_range, num_clients=10, alpha=0.5, overlap=False, min_per_client=5, max_per_client=None, rng=None, seed=None):
        if rng is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random
        targets = np.array(targets)
        client_data_indices = [[] for _ in range(num_clients)]
        for label in range(class_range[0], class_range[1]):
            idx = np.where(np.array(targets) == label)[0]
            rng.shuffle(idx)
            n = len(idx)
            proportions = rng.dirichlet([alpha] * num_clients)
            proportions = np.clip(proportions, 1e-06, 1)
            proportions = proportions / proportions.sum()
            if not overlap:
                splits = (proportions * n).astype(int)
                for i in range(num_clients):
                    if splits[i] < min_per_client:
                        splits[i] = min_per_client
                while splits.sum() > n:
                    max_idx = np.argmax(splits)
                    if splits[max_idx] > min_per_client:
                        splits[max_idx] -= 1
                    else:
                        break
                surplus_pool = []
                if max_per_client is not None:
                    for i in range(num_clients):
                        if splits[i] > max_per_client:
                            surplus = splits[i] - max_per_client
                            splits[i] = max_per_client
                            surplus_pool.extend(idx[:surplus])
                            idx = idx[surplus:]
                start = 0
                for i, count in enumerate(splits):
                    client_data_indices[i].extend(idx[start:start + count])
                    start += count
                if surplus_pool:
                    sample_counts = [len(client_data_indices[i]) for i in range(num_clients)]
                    least_clients = np.argsort(sample_counts)[:5]
                    surplus_proportions = rng.dirichlet([alpha] * 5)
                    surplus_proportions = (surplus_proportions / surplus_proportions.sum() * len(surplus_pool)).astype(int)
                    start = 0
                    for i, cid in enumerate(least_clients):
                        end = start + surplus_proportions[i]
                        client_data_indices[cid].extend(surplus_pool[start:end])
                        start = end
                    for i in range(start, len(surplus_pool)):
                        client_data_indices[least_clients[i % 5]].append(surplus_pool[i])
            else:
                scale = 10
                for i, p in enumerate(proportions):
                    sample_size = max(int(p * scale * n), min_per_client)
                    sampled_idx = rng.choice(idx, size=sample_size, replace=True)
                    client_data_indices[i].extend(sampled_idx.tolist())
        for i in range(num_clients):
            pass
        return client_data_indices

    def getSampleData(self, classes, exemplar_set, exemplar_label_set, group):
        datas, labels = ([], [])
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full(length, label) for label in exemplar_label_set]
        if group == 0:
            for label in classes:
                data = self.data[np.array(self.targets) == label]
                datas.append(data)
                labels.append(np.full(data.shape[0], label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def getTrainItem(self, index):
        img, target = (Image.fromarray(self.TrainData[index]), self.TrainLabels[index])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return (index, img, target)

    def getTestItem(self, index):
        img, target = (Image.fromarray(self.TestData[index]), self.TestLabels[index])
        if self.test_transform:
            img = self.test_transform(img)
        if self.target_test_transform:
            target = self.target_test_transform(target)
        return (index, img, target)

    def getTrainItem_indices(self, indices):
        subset_data = self.TrainData[indices]
        subset_labels = self.TrainLabels[indices]
        subset_dataset = iCIFAR10(root=self.root, train=True, transform=self.transform, target_transform=self.target_transform, download=False)
        subset_dataset.TrainData = subset_data
        subset_dataset.TrainLabels = subset_labels
        return subset_dataset

    def getTrainImg_indices(self, indices):
        subset_data = self.TrainData[indices]
        subset_labels = self.TrainLabels[indices]
        return (subset_data, subset_labels)

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if len(self.TrainData) > 0:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

    def distribution_heatmap(self, targets, client_data_indices, save_path, num_clients=10, class_range=[0, 10]):
        distribution_matrix = np.zeros((num_clients, class_range[1] - class_range[0]), dtype=int)
        for client_id in range(num_clients):
            labels = targets[client_data_indices[client_id]]
            for class_id in range(class_range[0], class_range[1]):
                cls_idx = class_id - class_range[0]
                distribution_matrix[client_id, cls_idx] = np.sum(labels == class_id)
        plt.figure(figsize=(24, 16))
        sns.heatmap(distribution_matrix, annot=False, fmt='d', cmap='YlGnBu', cbar=True, linewidths=0.5)
        plt.title('Client Data Distribution Heatmap')
        plt.xlabel('Classes')
        plt.ylabel('Clients')
        plt.savefig(save_path)
        plt.show()

    def incre_split_datasets(self, targets, class_id, new_clients, num_clients, incre_sample_splits, alpha=0.5, min_per_client=3, rng=None, seed=None):
        if rng is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random
        targets = np.array(targets)
        client_indices_dict = {f'client_index{index}': {'round_0': [], 'round_1': [], 'round_2': [], 'round_3': []} for index in range(num_clients)}
        client_data_indices = [[] for _ in range(num_clients)]
        idx = np.where(np.array(targets) == class_id)[0]
        rng.shuffle(idx)
        for i in range(new_clients):
            start = 0
            idx_temp = idx[i * 100:(i + 1) * 100]
            client_data_indices[i].extend(idx_temp)
            for j, size in enumerate(incre_sample_splits):
                client_indices_dict[f'client_index{i}'][f'round_{j}'] = idx_temp[start:start + size]
                start += size
        old_clients = num_clients - new_clients
        idx_remain = idx[400:]
        proportions = rng.dirichlet([alpha] * old_clients)
        proportions = np.clip(proportions, 1e-06, 1)
        proportions = proportions / proportions.sum()
        proportions = (proportions * 100).astype(int)
        for i in range(old_clients):
            if proportions[i] < min_per_client:
                proportions[i] = min_per_client
            while proportions.sum() > 100:
                max_idx = np.argmax(proportions)
                if proportions[max_idx] > min_per_client:
                    proportions[max_idx] -= 1
                else:
                    break
        start = 0
        for i, count in enumerate(proportions):
            client_indices_dict[f'client_index{new_clients + i}']['round_0'] = idx_remain[start:start + count]
            client_data_indices[new_clients + i].extend(idx_remain[start:start + count])
            start += count
        return (client_indices_dict, client_data_indices)

    def get_class_data(self, class_id):
        datas = self.TrainData[np.array(self.TrainLabels) == class_id]
        labels = self.TrainLabels[np.array(self.TrainLabels) == class_id]
        return (datas, labels)

class iSVHN(SVHN):

    def __init__(self, root, train=True, transform=None, target_transform=None, test_transform=None, target_test_transform=None, download=False, train_per_class: int=-1, seed: int=2021):
        split = 'train' if bool(train) else 'test'
        super(iSVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        if hasattr(self, 'labels') and (not hasattr(self, 'targets')):
            self.targets = self.labels
        if bool(train) and int(train_per_class) and (int(train_per_class) > 0):
            rng = np.random.default_rng(int(seed))
            targets = np.asarray(self.targets)
            kept_indices = []
            for cls in range(10):
                cls_idx = np.where(targets == cls)[0]
                if cls_idx.size == 0:
                    continue
                if cls_idx.size <= int(train_per_class):
                    chosen = cls_idx
                else:
                    chosen = rng.choice(cls_idx, size=int(train_per_class), replace=False)
                kept_indices.extend([int(x) for x in chosen])
            kept_indices = np.asarray(kept_indices, dtype=int)
            rng.shuffle(kept_indices)
            self.data = self.data[kept_indices]
            self.targets = targets[kept_indices].tolist()
            if hasattr(self, 'labels'):
                self.labels = np.asarray(self.targets)
        if isinstance(self.data, np.ndarray) and self.data.ndim == 4 and (self.data.shape[1] in (1, 3)):
            self.data = np.transpose(self.data, (0, 2, 3, 1))
        if isinstance(self.data, np.ndarray) and self.data.dtype != np.uint8:
            self.data = self.data.astype(np.uint8)
        self.target_test_transform = target_test_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return (con_data, con_label)

    def getTestData(self, classes):
        datas, labels = ([], [])
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full(data.shape[0], label))
        self.TestData, self.TestLabels = self.concatenate(datas, labels)

    def getTrainData(self, classes, new_class_id, exemplar_set, exemplar_label_set, new_class_indices=None):
        datas, labels = ([], [])
        if len(exemplar_set) != 0 and len(exemplar_label_set) != 0:
            if new_class_id >= 1 and new_class_indices is not None:
                filtered_exemplar_set = []
                filtered_exemplar_label_set = []
                for data, label in zip(exemplar_set, exemplar_label_set):
                    lab = np.asarray(label)
                    if lab.ndim == 0:
                        if int(lab) != new_class_id:
                            filtered_exemplar_set.append(data)
                            filtered_exemplar_label_set.append(int(lab))
                        continue
                    mask = lab != new_class_id
                    if not np.any(mask):
                        continue
                    dat = np.asarray(data)
                    if dat.ndim >= 1 and dat.shape[0] == lab.shape[0]:
                        filtered_exemplar_set.append(dat[mask])
                        filtered_exemplar_label_set.append(lab[mask])
                    else:
                        filtered_exemplar_set.append(dat)
                        filtered_exemplar_label_set.append(lab)
                datas.extend(filtered_exemplar_set)
                labels.extend(filtered_exemplar_label_set)
            else:
                datas.extend(exemplar_set)
                labels.extend(exemplar_label_set)
        if new_class_id >= 1:
            if new_class_indices is None:
                self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
                return
            new_data = self.data[new_class_indices]
            datas.append(new_data)
            labels.append(np.full(len(new_class_indices), new_class_id))
        if new_class_id == -1:
            for label in classes:
                data = self.data[np.array(self.targets) == label]
                datas.append(data)
                labels.append(np.full(data.shape[0], label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)

    def dirichlet_split(self, data, targets, class_range, num_clients=10, alpha=0.5, overlap=False, min_per_client=5, rng=None, seed=None):
        if rng is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random
        client_data = [[] for _ in range(num_clients)]
        client_targets = [[] for _ in range(num_clients)]
        for label in range(class_range[0], class_range[1]):
            idx = np.where(np.array(targets) == label)[0]
            n = len(idx)
            proportions = rng.dirichlet([alpha] * num_clients)
            if not overlap:
                proportions = (proportions * n).astype(int)
                for i in range(num_clients):
                    if proportions[i] < min_per_client:
                        proportions[i] = min_per_client
                while proportions.sum() < n:
                    proportions[np.argmax(proportions)] += 1
                while proportions.sum() > n:
                    max_idx = np.argmax(proportions)
                    if proportions[max_idx] > min_per_client:
                        proportions[max_idx] -= 1
                    else:
                        for j in range(num_clients):
                            if proportions[j] > min_per_client:
                                proportions[j] -= 1
                                break
                    proportions[np.argmax(proportions)] -= 1
                rng.shuffle(idx)
                start = 0
                for i, count in enumerate(proportions):
                    client_data[i].extend(data[idx[start:start + count]])
                    client_targets[i].extend([label] * count)
                    start += count
            else:
                scale = 10
                for i, p in enumerate(proportions):
                    sample_size = max(int(p * scale * n), min_per_client)
                    sampled_idx = rng.choice(idx, sample_size, replace=True)
                    client_data[i].extend(data[sampled_idx])
                    client_targets[i].extend([label] * sample_size)
        client_data = [np.array(d) for d in client_data]
        client_targets = [np.array(t) for t in client_targets]
        return (client_data, client_targets)

    def dirichlet_split_indices(self, targets, class_range, num_clients=10, alpha=0.5, overlap=False, min_per_client=5, max_per_client=None, rng=None, seed=None):
        if rng is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random
        targets = np.array(targets)
        client_data_indices = [[] for _ in range(num_clients)]
        for label in range(class_range[0], class_range[1]):
            idx = np.where(np.array(targets) == label)[0]
            rng.shuffle(idx)
            n = len(idx)
            proportions = rng.dirichlet([alpha] * num_clients)
            proportions = np.clip(proportions, 1e-06, 1)
            proportions = proportions / proportions.sum()
            if not overlap:
                splits = (proportions * n).astype(int)
                for i in range(num_clients):
                    if splits[i] < min_per_client:
                        splits[i] = min_per_client
                while splits.sum() > n:
                    max_idx = np.argmax(splits)
                    if splits[max_idx] > min_per_client:
                        splits[max_idx] -= 1
                    else:
                        break
                surplus_pool = []
                if max_per_client is not None:
                    for i in range(num_clients):
                        if splits[i] > max_per_client:
                            surplus = splits[i] - max_per_client
                            splits[i] = max_per_client
                            surplus_pool.extend(idx[:surplus])
                            idx = idx[surplus:]
                start = 0
                for i, count in enumerate(splits):
                    client_data_indices[i].extend(idx[start:start + count])
                    start += count
                if surplus_pool:
                    sample_counts = [len(client_data_indices[i]) for i in range(num_clients)]
                    least_clients = np.argsort(sample_counts)[:5]
                    surplus_proportions = rng.dirichlet([alpha] * 5)
                    surplus_proportions = (surplus_proportions / surplus_proportions.sum() * len(surplus_pool)).astype(int)
                    start = 0
                    for i, cid in enumerate(least_clients):
                        end = start + surplus_proportions[i]
                        client_data_indices[cid].extend(surplus_pool[start:end])
                        start = end
                    for i in range(start, len(surplus_pool)):
                        client_data_indices[least_clients[i % 5]].append(surplus_pool[i])
            else:
                scale = 10
                for i, p in enumerate(proportions):
                    sample_size = max(int(p * scale * n), min_per_client)
                    sampled_idx = rng.choice(idx, size=sample_size, replace=True)
                    client_data_indices[i].extend(sampled_idx.tolist())
        for i in range(num_clients):
            pass
        return client_data_indices

    def getTrainItem(self, index):
        img_arr = self.TrainData[index]
        if img_arr.ndim == 3 and img_arr.shape[0] == 3:
            img_arr = np.transpose(img_arr, (1, 2, 0))
        img, target = (Image.fromarray(img_arr), self.TrainLabels[index])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return (index, img, target)

    def getTestItem(self, index):
        img_arr = self.TestData[index]
        if img_arr.ndim == 3 and img_arr.shape[0] == 3:
            img_arr = np.transpose(img_arr, (1, 2, 0))
        img, target = (Image.fromarray(img_arr), self.TestLabels[index])
        if self.test_transform:
            img = self.test_transform(img)
        if self.target_test_transform:
            target = self.target_test_transform(target)
        return (index, img, target)

    def getTrainItem_indices(self, indices):
        subset_data = self.TrainData[indices]
        subset_labels = self.TrainLabels[indices]
        subset_dataset = iSVHN(root=self.root, train=True, transform=self.transform, target_transform=self.target_transform, download=False)
        subset_dataset.TrainData = subset_data
        subset_dataset.TrainLabels = subset_labels
        return subset_dataset

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if len(self.TrainData) > 0:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

    def incre_split_datasets(self, targets, class_id, new_clients, num_clients, incre_sample_splits, alpha=0.5, min_per_client=3, rng=None, seed=None):
        if rng is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else np.random
        targets = np.array(targets)
        client_indices_dict = {f'client_index{index}': {'round_0': [], 'round_1': [], 'round_2': [], 'round_3': []} for index in range(num_clients)}
        client_data_indices = [[] for _ in range(num_clients)]
        idx = np.where(np.array(targets) == class_id)[0]
        rng.shuffle(idx)
        for i in range(new_clients):
            start = 0
            idx_temp = idx[i * 100:(i + 1) * 100]
            client_data_indices[i].extend(idx_temp)
            for j, size in enumerate(incre_sample_splits):
                client_indices_dict[f'client_index{i}'][f'round_{j}'] = idx_temp[start:start + size]
                start += size
        old_clients = num_clients - new_clients
        idx_remain = idx[400:]
        proportions = rng.dirichlet([alpha] * old_clients)
        proportions = np.clip(proportions, 1e-06, 1)
        proportions = proportions / proportions.sum()
        proportions = (proportions * 100).astype(int)
        for i in range(old_clients):
            if proportions[i] < min_per_client:
                proportions[i] = min_per_client
            while proportions.sum() > 100:
                max_idx = np.argmax(proportions)
                if proportions[max_idx] > min_per_client:
                    proportions[max_idx] -= 1
                else:
                    break
        start = 0
        for i, count in enumerate(proportions):
            client_indices_dict[f'client_index{new_clients + i}']['round_0'] = idx_remain[start:start + count]
            client_data_indices[new_clients + i].extend(idx_remain[start:start + count])
            start += count
        return (client_indices_dict, client_data_indices)

    def get_class_data(self, class_id):
        datas = self.TrainData[np.array(self.TrainLabels) == class_id]
        labels = self.TrainLabels[np.array(self.TrainLabels) == class_id]
        return (datas, labels)
