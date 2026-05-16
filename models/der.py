import logging
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPS = 1e-8

class DER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = DERNet(args, True)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager, skip_train=False, start_round=0):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        
        # Robust expansion check
        if len(self._network.convnets) <= self._cur_task:
            self._network.update_fc(self._total_classes)
            
        self._network.to(self._device)
            
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        # Setup Test Loader (Always needed for evaluation)
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
            # Ensure train_loader is at least None to avoid AttributeErrors elsewhere
            if not hasattr(self, 'train_loader'):
                self.train_loader = None
            try:
                self.build_rehearsal_memory(data_manager, self.samples_per_class)
            except Exception as e:
                logging.warning(f"Could not build rehearsal memory for skipped task {self._cur_task}: {e}")
            return

        # Setup Train Loader (Only if not skip_train)
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
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self._network.parameters()),
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
        prog_bar = tqdm(range(start_round, self.args.get("epochs", 30)))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output = self._network(inputs)
                logits, aux_logits = output["logits"], output["aux_logits"]

                loss = F.cross_entropy(logits, targets)
                # Map global targets to local targets for auxiliary loss
                # New classes: 0 to new_task_size - 1
                # Old classes: new_task_size
                aux_targets = torch.where(
                    targets < self._known_classes,
                    torch.tensor(self._total_classes - self._known_classes).to(self._device),
                    targets - self._known_classes,
                )
                aux_loss = F.cross_entropy(aux_logits, aux_targets)
                loss = loss + aux_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if test_loader is not None:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args.get("epochs", 30), losses / len(train_loader), train_acc, test_acc["total"]
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args.get("epochs", 30), losses / len(train_loader), train_acc
                )
            prog_bar.set_description(info)
            
            # --- SAVE INTRA-TASK CHECKPOINT ---
            run_dir = self.args.get("run_dir", ".")
            ckpt_dir = os.path.join(run_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_name = f'ckpt_task{self._cur_task:02d}_round{epoch + 1:03d}.pth'
            torch.save({
                'task':             self._cur_task,
                'round':            epoch + 1,
                'model_state_dict': self._network.state_dict(),
                'known_classes':    self._known_classes,
            }, os.path.join(ckpt_dir, ckpt_name))
            
            

        logging.info(info)
        # Reset start_round for the next tasks so they start from 0
        self.args["start_round"] = 0