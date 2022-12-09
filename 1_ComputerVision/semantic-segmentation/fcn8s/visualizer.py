import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch import Tensor

from one_hot import OneHotEncoder


class Visualizer:
    def __init__(self, class_names, class_colors):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.class_colors = class_colors

    def draw_roc_auc(self, y_true: Tensor, y_pred: Tensor, title, x_label="False Positive Rate", y_label="True Positive Rate"):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            y_true_np = y_true[:, i, :, :].reshape(-1).cpu().numpy()
            y_pred_np = y_pred[:, i, :, :].reshape(-1).cpu().numpy()
            fpr[i], tpr[i], _ = roc_curve(y_true_np, y_pred_np)
            roc_auc[i] = auc(fpr[i], tpr[i])

        for i, color in zip(range(self.n_classes), self.class_colors):
            plt.plot(
                fpr[i],
                tpr[i],
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})".format(self.class_names[i], roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    def draw_pr(self, y_true: Tensor, y_pred: Tensor, title, x_label="Recall", y_label="Precision"):
        precision = dict()
        recall = dict()
        aps = dict()
        for i in range(self.n_classes):
            y_true_np = y_true[:, i, :, :].reshape(-1).cpu().numpy()
            y_pred_np = y_pred[:, i, :, :].reshape(-1).cpu().numpy()
            precision[i], recall[i], thresholds = precision_recall_curve(y_true_np, y_pred_np)
            aps[i] = average_precision_score(y_true_np, y_pred_np)

        for i, color in zip(range(self.n_classes), self.class_colors):
            plt.plot(
                recall[i],
                precision[i],
                lw=2,
                label="PR of class {0} (area = {1:0.2f})".format(self.class_names[i], aps[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    def draw_result(self, img: Tensor, mask: Tensor, y_pred: Tensor):
        mask_img = OneHotEncoder.decode(mask, self.class_colors)
        pred_img = OneHotEncoder.decode(y_pred, self.class_colors)
        plt.figure(figsize=(12, 5))
        plt.subplot(131)
        plt.imshow(img.permute(1, 2, 0))
        plt.subplot(132)
        plt.imshow(mask_img.permute(1, 2, 0))
        plt.subplot(133)
        plt.imshow(pred_img.permute(1, 2, 0))
        plt.show()

    def draw_overlay_grid(self, img: Tensor, overlay_classes, overlay_colors, y_pred: Tensor, label):
        font = {'color': 'green',
                'size': 20,
                'family': 'Times New Roman'}
        grid = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        w, h = img.shape[1:]
        k_size = grid.shape[0]
        left, top = 0, 0
        while top < h:
            left = 0
            bottom = min(top + k_size, h)
            while left < w:
                right = min(left + k_size, w)
                sum_pred = torch.sum(y_pred[:, top:bottom, left:right].flatten(1, 2), dim=1)
                klass = sum_pred.argmax()
                if klass in overlay_classes:
                    overlay_index = overlay_classes.index(klass)
                    img[:, top:bottom, left:right] = torch.mul(
                        img[:, top:bottom, left:right], grid[0:bottom-top,0:right-left]) + torch.mul(overlay_colors[overlay_index][:,None, None], grid ^ 1)
                left = right
            top = bottom

        plt.figure(figsize=(12, 5))
        plt.imshow(img.permute(1, 2, 0))
        if label:
            plt.text(10, 20, label, fontdict=font)
        plt.show()