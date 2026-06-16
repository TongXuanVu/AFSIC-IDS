import copy
import logging
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.base import BaseLearner
from utils.inc_net import AFSICIDSNet
from utils.memory import LocalExemplarMemory, GlobalPrototypeMemory
from losses import compute_kd_loss, compute_fsp_loss, compute_proto_loss, compute_sparse_regularization, compute_fedprox_regularization

class AFSIC_IDS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = AFSICIDSNet(args, False)
        self._old_network = None
        self.local_memory = LocalExemplarMemory(
            memory_ratio=args.get("memory_ratio", 0.01),
            memory_per_class=args.get("memory_per_class", None)
        )
        self.global_proto_memory = GlobalPrototypeMemory()
        self.best_acc_per_task = {}

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager, skip_train=False):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        # Update network fc size
        self._network.update_fc(self._total_classes)
        self._network.to(self._device)
        
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._total_classes))
        
        if self._cur_task > 0:
            # Transition to incremental task: freeze stability encoder, initialize new adapter/gate
            self._network.transition_to_incremental_stage()
            self._network.freeze_stability_encoder()
            self._network.unfreeze_adapter()
            
            # _old_network for KD must be the previous global model before expansion.
            # The federated trainer injects it before local training; do not copy
            # the newly expanded adapter model here.
        
        # Setup Test Loader
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args["batch_size"], 
            shuffle=False, 
            num_workers=self.args.get("num_workers", 0)
        )

        if skip_train:
            logging.info(f"Skipping training for task {self._cur_task}")
            if not hasattr(self, 'train_loader'):
                self.train_loader = None
            return

        # Setup Train Loader
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args["batch_size"], 
            shuffle=True, 
            num_workers=self.args.get("num_workers", 0)
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        # Optimize only trainable incremental parameters during incremental stages
        if self._cur_task > 0:
            params = self._network.get_trainable_incremental_params()
        else:
            params = self._network.parameters()
            
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, params),
            lr=self.args.get("lr", 0.001),
            weight_decay=self.args.get("weight_decay", 0.0002),
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=self.args.get("milestones", [80, 120, 150]), 
            gamma=self.args.get("gamma", 0.1)
        )
        
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        start_round = self.args.get("start_round", 0)
        epochs = self.args.get("epochs", 30)
        
        # 1. Compute class weights for Class-balanced CE
        class_counts = torch.zeros(self._total_classes).to(self._device)
        for _, _, targets in train_loader:
            targets = targets.to(self._device)
            class_counts += torch.bincount(targets, minlength=self._total_classes)
        
        total_samples = class_counts.sum()
        class_weights = torch.zeros(self._total_classes).to(self._device)
        for c in range(self._total_classes):
            if class_counts[c] > 0:
                class_weights[c] = total_samples / (class_counts[c] * self._total_classes)
            else:
                class_weights[c] = 1.0

        # Save model params at start of local training round for FedProx
        self.global_model_params_round_start = {
            name: p.clone().detach()
            for name, p in self._network.named_parameters()
            if p.requires_grad
        }

        # Build prototype matrix for FSP and Proto Alignment
        proto_matrix = torch.zeros(self._total_classes, self._network.feature_dim).to(self._device)
        for c in range(self._total_classes):
            proto = self.global_proto_memory.get_prototype(c)
            if proto is not None:
                proto_matrix[c] = proto.to(self._device)
            else:
                # fallback to normalized classifier weight
                proto_matrix[c] = F.normalize(self._network.fc.weight.data[c], p=2, dim=0)

        for epoch in range(start_round, epochs):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                features = outputs["features"]
                
                # a. Class-balanced CE Loss
                loss_ce = F.cross_entropy(logits, targets, weight=class_weights)
                
                # b. KD Loss on old classes
                loss_kd = torch.tensor(0.0).to(self._device)
                if self._cur_task > 0 and self._old_network is not None:
                    self._old_network.eval()
                    with torch.no_grad():
                        old_outputs = self._old_network(inputs)
                        old_logits = old_outputs["logits"][:, :self._known_classes]
                    new_logits = logits[:, :self._known_classes]
                    loss_kd = compute_kd_loss(new_logits, old_logits, T=2.0)
                
                # c. FSP Loss (Few-Shot Sparse Pairwise Loss)
                loss_fsp = compute_fsp_loss(features, targets, proto_matrix, T_fsp=0.5)
                
                # d. Prototype Alignment Loss
                loss_proto = compute_proto_loss(features, targets, proto_matrix)
                
                # e. Sparse Regularization Loss (L1 norm on adapter and gate parameters)
                loss_rs = torch.tensor(0.0).to(self._device)
                if self._cur_task > 0:
                    loss_rs = compute_sparse_regularization(self._network, self._device)
                
                # f. FedProx Regularization Loss
                loss_prox = torch.tensor(0.0).to(self._device)
                if self._cur_task > 0:
                    loss_prox = compute_fedprox_regularization(self._network, self.global_model_params_round_start, self._device)
                
                # Hyperparameters. Base stage should use CE only; prototype losses
                # are meaningful after old-class prototypes exist.
                if self._cur_task == 0:
                    lambda_kd = lambda_fsp = lambda_proto = lambda_rs = lambda_prox = 0.0
                else:
                    lambda_kd = self.args.get("lambda_kd", 1.0)
                    lambda_fsp = self.args.get("lambda_fsp", 0.5)
                    lambda_proto = self.args.get("lambda_proto", 0.5)
                    lambda_rs = self.args.get("lambda_rs", 0.0001)
                    lambda_prox = self.args.get("lambda_prox", 0.01)
                
                loss = loss_ce + lambda_kd * loss_kd + lambda_fsp * loss_fsp + lambda_proto * loss_proto + lambda_rs * loss_rs + lambda_prox * loss_prox
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)
                
            scheduler.step()
            
            train_acc = np.around(correct.item() * 100 / total, decimals=2)
            if test_loader is not None:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc["total"]
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc
                )
            logging.info(info)

    def _get_memory(self):
        return self.local_memory.get_memory()

    @property
    def exemplar_size(self):
        mem = self._get_memory()
        if mem is None:
            return 0
        return len(mem[1])

    def build_rehearsal_memory(self, data_manager, per_class):
        self._network.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            if len(data) == 0:
                continue
            
            loader = DataLoader(dset, batch_size=self.args["batch_size"], shuffle=False, num_workers=0)
            features_list = []
            
            with torch.no_grad():
                for _, inputs, _ in loader:
                    inputs = inputs.to(self._device)
                    feats = self._network.extract_vector(inputs)
                    features_list.append(feats.cpu())
                    
            if features_list:
                features_all = torch.cat(features_list, dim=0)
                self.local_memory.construct_exemplars(class_idx, data, targets, features_all)

    def compute_local_prototypes(self, data_manager, class_ids=None, max_samples_per_class=None, seed=0):
        local_protos = {}
        self._network.eval()
        
        # Compute prototypes for selected active classes.
        if class_ids is None:
            class_ids = range(self._total_classes)

        rng = np.random.default_rng(seed)
        for class_idx in class_ids:
            data, targets, dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            if len(data) == 0:
                continue

            if max_samples_per_class is not None and len(data) > max_samples_per_class:
                selected = rng.choice(len(data), size=max_samples_per_class, replace=False)
                data = data[selected] if isinstance(data, np.ndarray) else [data[int(i)] for i in selected]
                targets = targets[selected] if isinstance(targets, np.ndarray) else [targets[int(i)] for i in selected]
                from utils.data_manager import DummyDataset
                from torchvision import transforms
                trsf = transforms.Compose([*data_manager._test_trsf, *data_manager._common_trsf])
                dset = DummyDataset(data, targets, trsf, data_manager.use_path)
            
            loader = DataLoader(dset, batch_size=self.args["batch_size"], shuffle=False, num_workers=0)
            features_list = []
            
            with torch.no_grad():
                for _, inputs, _ in loader:
                    inputs = inputs.to(self._device)
                    feats = self._network.extract_vector(inputs)
                    feats_norm = F.normalize(feats, p=2, dim=1)
                    features_list.append(feats_norm.cpu())
            
            if features_list:
                features_all = torch.cat(features_list, dim=0)
                prototype = torch.mean(features_all, dim=0)
                prototype = prototype / (torch.norm(prototype, p=2) + 1e-8)
                
                proto_np = prototype.numpy()
                feats_np = features_all.numpy()
                cos_sims = np.dot(feats_np, proto_np)
                cos_dists = 1.0 - cos_sims
                dispersion = float(np.mean(cos_dists))
                
                local_protos[class_idx] = {
                    "prototype": prototype,
                    "count": len(data),
                    "dispersion": dispersion
                }
        return local_protos


