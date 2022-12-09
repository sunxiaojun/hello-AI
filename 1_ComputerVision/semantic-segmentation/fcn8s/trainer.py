import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as T

from one_hot import OneHotEncoder
from visualizer import Visualizer


class Trainer(object):
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, class_names, class_colors):
        self.model = model
        self.transform = T.Compose([
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.visualizer = Visualizer(class_names, class_colors)
        self.acc_thresholds = 0.7
        self.best_mean_iou = 0
        self.max_checkpoints = 5
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_dir = os.path.dirname(__file__)
        self.checkpoints_dir = os.path.join(self.save_dir, 'checkpoints')
        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)

    def train(self, epochs=50, learning_rate=1e-3, momentum=0.7, step_size=5, gamma=0.5, verbose=True):
        start_time = datetime.now()
        logging.info(f'start training at {start_time}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        criterion = nn.CrossEntropyLoss()

        self.model.to(device)
        self.model.train()
        for epoch in range(1, epochs + 1):
            lr_current = optimizer.param_groups[0]['lr']
            print(f'learning rate:{lr_current}')
            for batch_index, data in enumerate(self.train_loader):
                std_input = data[0].float()/255
                if self.transform:
                    std_input = self.transform(std_input)
                std_input = Variable(std_input.to(device))
                target = data[1].float().to(device)

                # forward
                score = self.model(std_input)
                loss = criterion(score, target)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose:
                    print(f'[train] epoch {epoch} / {epochs}: loss: {loss.item():.5f}')
            val_loss, val_acc, mean_iou = self.validate()
            logging.info(f'[validate] epoch {epoch} / {epochs}: loss: {val_loss:.5f}, accuracy:{val_acc:.5f}, mean IOU:{mean_iou:.5f}')
            self.save_checkpoint(epoch, val_acc)
            if val_acc > self.acc_thresholds and mean_iou > self.best_mean_iou:
                self.save_model()
                self.best_mean_iou = mean_iou
            scheduler.step()
        end_time = datetime.now()
        logging.info(f'end training at {end_time}. time elapse:{(end_time - start_time).seconds // 60 } min')

    def validate(self, verbose=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        val_loss = 0.0
        val_acc = 0.0
        mean_iou = 0.0
        num_batches = len(self.val_loader)

        self.model.to(device)
        self.model.eval()
        for batch_index, data in enumerate(self.val_loader):
            std_input = data[0].float() / 255
            if self.transform:
                std_input = self.transform(std_input)
            std_input = Variable(std_input.to(device))
            target = data[1].float().to(device)
            with torch.no_grad():
                score = self.model(std_input)

            # metrics
            loss = criterion(score, target)
            if np.isnan(loss.item()):
                raise ValueError('loss is nan while validating')
            val_loss += loss.item()
            pred = OneHotEncoder.encode_score(score)
            cm = Trainer.confusion_matrix(target, pred)

            acc = torch.diag(cm).sum().item() / torch.sum(cm).item()
            val_acc += acc
            iu = torch.diag(cm) / (cm.sum(dim=1) + cm.sum(dim=0) - torch.diag(cm))
            mean_iou += torch.nanmean(iu).item()
            if verbose:
                print(f'[validate] loss: {loss.item():.5f}, accuracy:{acc:.5f}, mean IOU:{torch.nanmean(iu).item():.5f}')

        val_loss /= num_batches
        val_acc /= num_batches
        mean_iou /= num_batches

        self.model.train()
        return val_loss, val_acc, mean_iou

    def test(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_target = None
        test_probability = None
        for batch_index, data in enumerate(self.test_loader):
            std_input = data[0].float() / 255
            if self.transform:
                std_input = self.transform(std_input)
            std_input = Variable(std_input.to(device))
            target = data[1].float().to(device)
            with torch.no_grad():
                score = self.model(std_input)
            probability = torch.nn.Softmax(dim=1)(score)
            if test_target is None:
                test_target = target
                test_probability = probability
            else:
                test_target = torch.cat((test_target, target), dim=0)
                test_probability = torch.cat((test_probability, probability), dim=0)

        self.visualizer.draw_roc_auc(test_target, test_probability, "ROC curve")
        self.visualizer.draw_pr(test_target, test_probability, "PR curve")
        return test_probability

    def save_checkpoint(self, epoch, acc):
        if self.max_checkpoints <= 0:
            return
        checkpoint = {
            'state': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(checkpoint, os.path.join(self.checkpoints_dir, f'checkpoint_{epoch % self.max_checkpoints}.pth'))
        logging.info(f'Checkpoint {epoch} saved!')

    def resume_checkpoint(self, epoch):
        saved_checkpoint = os.path.join(self.checkpoints_dir, f'checkpoint_epoch{epoch}.pth')
        if not os.path.exists(saved_checkpoint):
            return
        checkpoint = torch.load(saved_checkpoint)
        self.model.load_state_dict(checkpoint["state"])
        logging.info(f'Checkpoint {epoch} resumed!')

    def save_model(self):
        torch.save(self.model, os.path.join(self.save_dir, 'fcn8s.pth'))
        print('model fcn8s saved!')

    def load_model(self):
        saved_model = os.path.join(self.save_dir, 'fcn8s.pth')
        if not os.path.exists(saved_model):
            return
        self.model = torch.load(saved_model)
        print('model fcn8s loaded!')

    @staticmethod
    def confusion_matrix(target: Tensor, input: Tensor) -> Tensor:
        """
        生成多分类混淆矩阵

        Args:

            target: one-hot encoding target :math:`(N, C, H, W)` or `(C, H, W)`

            input: one-hot encoding input :math:`(N, C, H, W)` or `(C, H, W)`

        Returns:

            Tensor - Confusion Matrix
        """

        if target.dim() != input.dim():
            raise IOError('target and input must has same dimension')
        if 4 == target.dim():
            y_true = torch.flatten(target.permute(1, 0, 2, 3), 1, 3).int()
            y_pred = torch.flatten(input.permute(1, 0, 2, 3), 1, 3).int()
        elif 3 == target.dim():
            y_true = torch.flatten(target, 1, 2).int()
            y_pred = torch.flatten(input, 1, 2).int()
        else:
            raise IOError('target and input must be a 3D or 4D matrix')

        n_classes = y_true.shape[0]
        cm = torch.zeros((n_classes, n_classes)).cuda()
        for i in range(n_classes):
            for j in range(n_classes):
                num = torch.sum((y_true[i] & y_pred[j]).int())
                cm[i, j] += num
        return cm

