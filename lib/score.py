import math
import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, f1_score, mean_squared_error
from torch.nn.functional import one_hot


# matplotlib.use('Agg')


class Scorer:
    def __init__(self, base_path, show=False):
        self._base_path = base_path

        self._accuracy_list = []
        self._loss_list = []
        self._score_list = []

        self._epoch_loss = 0.0
        self._labels_list = []
        self._preds_list = []
        self._output_list = []

        self._save_num = 0
        self._show = show

    def show(self, plt, path):
        if self._show:
            plt.show()
        else:
            plt.savefig(path)
        plt.close()

    def reset_epoch(self):
        self._epoch_loss = 0.0
        self._labels_list = []
        self._preds_list = []
        self._output_list = []

    def add_batch_result(self, outputs, labels, batch_loss):
        self._preds_list.extend(outputs.tolist())
        self._output_list.extend(outputs.tolist())
        self._labels_list.extend(labels.tolist())
        self._epoch_loss += batch_loss

        return self._epoch_loss / len(self._output_list)

    def get_epoch_result(self, draw: bool, is_merged: bool, write=False, title="val"):
        """
        :return loss, accuracy (accuracy, MSE, etc), score (F1, RMSE, etc)
        """
        epoch_loss = self._epoch_loss / len(self._preds_list)

        print('Loss: {:.4f}'.format(epoch_loss))

        if is_merged:
            self._loss_list.append(epoch_loss)

        if write:
            write_path = os.path.join(self._base_path, '{0}.txt'.format(title))
            with open(write_path, 'w', encoding='utf8') as f:
                lines = [
                    "The number of {0} data: {1}\n".format(title, len(self._preds_list)),
                    "Loss: {0}".format(epoch_loss),
                ]

                f.writelines(lines)

        self.reset_epoch()
        return epoch_loss, epoch_loss, epoch_loss

    def draw_total_result(self):
        fig, ax = plt.subplots(facecolor="w")

        if len(self._accuracy_list) > 0:
            ax.plot(range(len(self._accuracy_list)), self._accuracy_list, label="accuracy")

        ax.plot(range(len(self._loss_list)), self._loss_list, label="loss")

        if len(self._score_list) > 0:
            ax.plot(range(len(self._score_list)), self._score_list, label="score")
        plt.xlim(0, len(self._loss_list))

        ax.legend()
        save_fig = os.path.join(self._base_path, 'fig.png')
        self.show(plt, save_fig)

        return self._loss_list, self._accuracy_list, self._score_list

    def get_outputs(self):
        return self._output_list

    def get_preds(self):
        return self._preds_list


class BinaryScorer(Scorer):
    def add_batch_result(self, outputs, labels, batch_loss):
        reshaped = outputs.squeeze(1)
        preds = (reshaped > 0.5).float().cpu()
        # update loss summation
        self._epoch_loss += batch_loss.item() * outputs.size(0)
        # update correct prediction summation
        self._labels_list.extend(labels.tolist())
        self._preds_list.extend(preds.tolist())
        self._output_list.extend(reshaped.tolist())

        return self._epoch_loss / len(self._output_list)

    def get_epoch_result(self, draw: bool, is_merged: bool, write=False, title="val"):
        label_squeeze = [int(label_one[0]) for label_one in self._labels_list]

        fpr, tpr, _ = roc_curve(label_squeeze, self._output_list)
        if draw:
            plt.plot(fpr, tpr)
            self.show(plt, os.path.join(self._base_path, "{0}_roc{1}.png".format(title, self._save_num)))
            self._save_num += 1
        epoch_corrects = torch.sum(torch.tensor(self._preds_list).cpu() == torch.tensor(label_squeeze).cpu())
        epoch_f1 = f1_score(self._labels_list, self._preds_list)
        epoch_loss = self._epoch_loss / len(self._preds_list)
        epoch_acc = epoch_corrects.double() / len(self._preds_list)

        if is_merged:
            self._accuracy_list.append(epoch_acc.item())
            self._loss_list.append(epoch_loss)
            self._score_list.append(epoch_f1)

        if write:
            write_path = os.path.join(self._base_path, '{0}.txt'.format(title))
            with open(write_path, 'w', encoding='utf8') as f:
                lines = [
                    "The number of {0} data: {1}\n".format(title, len(self._preds_list)),
                    "Accuracy: {0}\n".format(epoch_acc),
                    "Loss: {0}\n".format(epoch_loss),
                    "F1 Score: {0}\n".format(epoch_f1)
                ]

                f.writelines(lines)

        print('Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))
        self.reset_epoch()

        return epoch_loss, epoch_acc, epoch_f1


class MulticlassScorer(Scorer):

    def __init__(self, base_path, show=False):
        super().__init__(base_path, show)

    def add_batch_result(self, outputs, labels, batch_loss):
        reshaped = torch.argmax(outputs, 1)
        # update loss summation
        self._epoch_loss += batch_loss.item() * outputs.size(0)
        # update correct prediction summation
        self._labels_list.extend(labels.tolist())
        self._preds_list.extend(reshaped.tolist())
        self._output_list.extend(outputs.tolist())

        return self._epoch_loss / len(self._output_list)

    def get_epoch_result(self, draw: bool, is_merged: bool, write=False, title="val"):
        n_class = len(self._output_list[0])
        if isinstance(self._labels_list[0], int):
            # labels are index of classes
            labels_total = one_hot(torch.tensor(self._labels_list), num_classes=n_class).tolist()
        else:
            labels_total = self._labels_list

        for i in range(n_class):

            labels = [label_one[i] for label_one in labels_total]
            outputs = [output_one[i] for output_one in self._output_list]

            fpr, tpr, _ = roc_curve(labels, outputs)

            if draw:
                plt.plot(fpr, tpr, label=str(i))

        if draw:
            plt.legend()
            self.show(plt, os.path.join(self._base_path, "val_roc" + str(self._save_num) + ".png"))
            self._save_num += 1

        epoch_corrects = torch.sum(torch.tensor(self._preds_list).cpu() == torch.tensor(self._labels_list).cpu())
        preds_total = one_hot(torch.tensor(self._preds_list), num_classes=n_class).tolist()
        epoch_f1 = f1_score(labels_total, preds_total, average='weighted')
        epoch_loss = self._epoch_loss / len(self._preds_list)
        epoch_acc = epoch_corrects.double() / len(self._preds_list)

        if is_merged:
            self._accuracy_list.append(epoch_acc.item())
            self._loss_list.append(epoch_loss)
            self._score_list.append(epoch_f1)

        if write:
            write_path = os.path.join(self._base_path, '{0}.txt'.format(title))
            with open(write_path, 'w', encoding='utf8') as f:
                lines = [
                    "The number of {0} data: {1}\n".format(title, len(self._preds_list)),
                    "Accuracy: {0}\n".format(epoch_acc),
                    "Loss: {0}\n".format(epoch_loss),
                    "F1 Score: {0}\n".format(epoch_f1)
                ]

                f.writelines(lines)

        print('Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))

        self.reset_epoch()
        return epoch_loss, epoch_acc, epoch_f1


class MSEScorer(Scorer):
    def add_batch_result(self, outputs, labels, batch_loss):
        reshaped = outputs.squeeze()
        # update loss summation
        self._epoch_loss += batch_loss.item() * outputs.size(0)
        # update correct prediction summation
        self._labels_list.extend(labels.tolist())
        self._preds_list.extend(reshaped.tolist())
        self._output_list.extend(outputs.tolist())

        return self._epoch_loss / len(self._output_list)

    def get_epoch_result(self, draw: bool, is_merged: bool, write=False, title="val"):
        label_squeeze = [label_one[0] for label_one in self._labels_list]

        if draw:
            plt.scatter(label_squeeze, self._preds_list)
            self.show(plt, os.path.join(self._base_path, "{}_mse{}.png".format(title, self._save_num)))
            self._save_num += 1

        epoch_mse = mean_squared_error(label_squeeze, self._preds_list)
        epoch_loss = self._epoch_loss / len(self._preds_list)

        if is_merged:
            self._loss_list.append(epoch_loss)
            self._score_list.append(epoch_mse)

        if write:
            write_path = os.path.join(self._base_path, '{0}.txt'.format(title))
            with open(write_path, 'w', encoding='utf8') as f:
                lines = [
                    "The number of {0} data: {1}\n".format(title, len(self._preds_list)),
                    "Loss: {0}\n".format(epoch_loss),
                    "MSE Score: {0}\n".format(epoch_mse)
                ]

                f.writelines(lines)

        print('Loss: {:.4f} MSE: {:.4f}'.format(epoch_loss, epoch_mse))
        self.reset_epoch()

        return epoch_loss, epoch_mse, math.sqrt(epoch_mse)


class GanScorer(Scorer):
    def __init__(self, base_path: str, gan_core, seed_shape: tuple[int, ...], show=False):
        super().__init__(base_path, show)

        self._gan_core = gan_core
        self._seed_shape = seed_shape

    def add_batch_result(self, outputs, labels, batch_loss):
        self._epoch_loss += batch_loss.item() * outputs.size(0)
        # update correct prediction summation
        self._output_list.extend(outputs.tolist())

        return self._epoch_loss / len(self._output_list)

    def get_epoch_result(self, draw: bool, is_merged: bool, write=False, title="val"):
        if draw:
            images = self._gan_core.generate(16)
            plt.figure(figsize=(10, 10))
            plt.axis('off')
            for i in range(images.shape[0]):
                plt.subplot(4, 4, i + 1)
                image = images[i, :, :, :]
                image = (image + 1) / 2
                image = image.clamp(0, 1)
                image = image.cpu().permute(1, 2, 0).numpy()
                plt.imshow(image)

            self.show(plt, os.path.join(self._base_path, "{}.png".format(str(self._save_num))))
            self._save_num += 1
        if len(self._output_list) > 0:
            epoch_loss = self._epoch_loss / len(self._output_list)
            print('Loss: {:.4f}'.format(epoch_loss))
        self.reset_epoch()
        return 0, 0, 0
