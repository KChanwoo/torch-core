import os
from abc import ABCMeta, abstractmethod

import matplotlib
import torch

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, f1_score
from torch.nn.functional import one_hot


class Scorer(metaclass=ABCMeta):
    def __init__(self, base_path):
        self._base_path = base_path

        self._accuracy_list = []
        self._loss_list = []
        self._f1_list = []

        self._epoch_loss = 0.0
        self._labels_list = []
        self._preds_list = []
        self._output_list = []

        self._save_num = 0

    def reset_epoch(self):
        self._epoch_loss = 0.0
        self._labels_list = []
        self._preds_list = []
        self._output_list = []

    @abstractmethod
    def add_batch_result(self, outputs, labels, batch_loss):
        self._preds_list.extend(outputs.tolist())
        self._output_list.extend(outputs.tolist())
        self._labels_list.extend(labels.tolist())
        self._epoch_loss += batch_loss

    @abstractmethod
    def get_epoch_result(self, draw: bool, is_merged: bool):
        epoch_loss = self._epoch_loss / len(self._preds_list)

        print('Loss: {:.4f}'.format(epoch_loss))

        if is_merged:
            self._loss_list.append(epoch_loss)

        self.reset_epoch()
        return epoch_loss

    def draw_total_result(self):
        fig, ax = plt.subplots(facecolor="w")

        if len(self._accuracy_list) > 0:
            ax.plot(range(len(self._accuracy_list)), self._accuracy_list, label="accuracy")

        ax.plot(range(len(self._loss_list)), self._loss_list, label="loss")

        if len(self._f1_list) > 0:
            ax.plot(range(len(self._f1_list)), self._f1_list, label="f1")
        plt.xlim(0, len(self._accuracy_list))

        ax.legend()
        save_fig = os.path.join(self._base_path, 'fig.png')
        plt.savefig(save_fig)
        plt.close()

        return self._loss_list, self._accuracy_list, self._f1_list


class BinaryScorer(Scorer):
    def add_batch_result(self, outputs, labels, batch_loss):
        reshaped = outputs.squeeze()
        preds = [0 if item < 0.5 else 1 for item in reshaped]
        # update loss summation
        self._epoch_loss += batch_loss.item() * outputs.size(0)
        # update correct prediction summation
        preds = torch.tensor(preds).cpu()
        self._labels_list.extend(labels.tolist())
        self._preds_list.extend(preds.tolist())
        self._output_list.extend(reshaped.tolist())

    def get_epoch_result(self, draw: bool, is_merged):
        label_squeeze = [int(label_one[0]) for label_one in self._labels_list]

        fpr, tpr, _ = roc_curve(label_squeeze, self._output_list)
        if draw:
            plt.plot(fpr, tpr)
            plt.savefig(os.path.join(self._base_path, "val_roc" + str(self._save_num) + ".png"))
            plt.close()
            self._save_num += 1
        epoch_corrects = torch.sum(torch.tensor(self._preds_list).cpu() == torch.tensor(label_squeeze).cpu())
        epoch_f1 = f1_score(self._labels_list, self._preds_list)
        epoch_loss = self._epoch_loss / len(self._preds_list)
        epoch_acc = epoch_corrects.double() / len(self._preds_list)

        if is_merged:
            self._accuracy_list.append(epoch_acc.item())
            self._loss_list.append(epoch_loss)
            self._f1_list.append(epoch_f1)

        print('Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
        self.reset_epoch()

        return epoch_loss


class MulticlassScorer(Scorer):

    def __init__(self, base_path, n_class: int):
        super().__init__(base_path)
        self.n_class = n_class

    def add_batch_result(self, outputs, labels, batch_loss):
        reshaped = torch.argmax(outputs, 1)
        # update loss summation
        self._epoch_loss += batch_loss.item() * outputs.size(0)
        # update correct prediction summation
        self._labels_list.extend(labels.tolist())
        self._preds_list.extend(reshaped.tolist())
        self._output_list.extend(outputs.tolist())

    def get_epoch_result(self, draw: bool, is_merged: bool):

        if isinstance(self._labels_list[0], int):
            # labels are index of classes
            labels_total = one_hot(torch.tensor(self._labels_list), num_classes=self.n_class).tolist()
        else:
            labels_total = self._labels_list

        for i in range(self.n_class):

            labels = [label_one[i] for label_one in labels_total]
            outputs = [output_one[i] for output_one in self._output_list]

            fpr, tpr, _ = roc_curve(labels, outputs)

            if draw:
                plt.plot(fpr, tpr, label=str(i))

        if draw:
            plt.legend()
            plt.savefig(os.path.join(self._base_path, "val_roc" + str(self._save_num) + ".png"))
            plt.close()
            self._save_num += 1

        epoch_corrects = torch.sum(torch.tensor(self._preds_list).cpu() == torch.tensor(self._labels_list).cpu())
        preds_total = one_hot(torch.tensor(self._preds_list), num_classes=self.n_class).tolist()
        epoch_f1 = f1_score(labels_total, preds_total, average='weighted')
        epoch_loss = self._epoch_loss / len(self._preds_list)
        epoch_acc = epoch_corrects.double() / len(self._preds_list)

        if is_merged:
            self._accuracy_list.append(epoch_acc.item())
            self._loss_list.append(epoch_loss)
            self._f1_list.append(epoch_f1)

        print('Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))

        self.reset_epoch()
        return epoch_loss
