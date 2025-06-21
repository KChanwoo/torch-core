import csv
import itertools

import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
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
        self._epoch_loss += batch_loss.item() if torch.is_tensor(batch_loss) else batch_loss

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

        return epoch_loss, epoch_loss, epoch_loss

    def draw_total_result(self, title='Train'):
        fig, ax = plt.subplots(facecolor="w")

        if len(self._accuracy_list) > 0:
            ax.plot(range(len(self._accuracy_list)), self._accuracy_list, label="accuracy")

        ax.plot(range(len(self._loss_list)), self._loss_list, label="loss")

        if len(self._score_list) > 0:
            ax.plot(range(len(self._score_list)), self._score_list, label="score")
        plt.xlim(0, len(self._loss_list))

        ax.legend()
        save_fig = os.path.join(self._base_path, '{}_fig.png'.format(title))
        self.show(plt, save_fig)
        self._save_num = 0
        return self._loss_list, self._accuracy_list, self._score_list

    def get_outputs(self):
        return self._output_list

    def get_preds(self):
        return self._preds_list

    def get_loss(self):
        return self._loss_list


class BinaryScorer(Scorer):
    def __init__(self, base_path, show=False, do_sigmoid=True):
        super().__init__(base_path, show)
        self.do_sigmoid = do_sigmoid

    def add_batch_result(self, outputs, labels, batch_loss):
        reshaped = outputs.squeeze(1)
        if self.do_sigmoid:
            preds = torch.nn.functional.sigmoid(reshaped)
        else:
            preds = reshaped
        preds = (preds > 0.5).float().cpu()
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
            cm = confusion_matrix(self._labels_list, self._preds_list)

            # visualise it
            plt.figure(figsize=(8, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
            plt.title("Confusion Matrix")
            plt.colorbar()

            tick_marks = np.arange(2)
            plt.xticks(tick_marks, [str(label) for label in list(range(2))], rotation=45)
            plt.yticks(tick_marks, [str(label) for label in list(range(2))])

            thresh = cm.max() / 2
            for i, j in itertools.product(range(cm.shape[1]), range(cm.shape[0])):
                plt.text(i, j, cm[j, i], horizontalalignment='center', color='white' if cm[j, i] > thresh else 'red')

            plt.tight_layout()
            plt.xlabel('Predictions')
            plt.ylabel('Real Values')
            self.show(plt, os.path.join(self._base_path, "{0}_cm".format(title) + str(self._save_num) + ".png"))

            write_path = os.path.join(self._base_path, '{0}.txt'.format(title))
            with open(write_path, 'w', encoding='utf8') as f:
                lines = [
                    "The number of {0} data: {1}\n".format(title, len(self._preds_list)),
                    "Accuracy: {0}\n".format(epoch_acc),
                    "Loss: {0}\n".format(epoch_loss),
                    "F1 Score: {0}\n".format(epoch_f1),
                    "Confusion Matrix: {0}".format(cm)
                ]

                f.writelines(lines)

        print('Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))

        return epoch_loss, epoch_acc, epoch_f1

    def draw_total_result(self, title='Train'):
        """
        Draws classification‑specific visualisations:
         • Confusion‑matrix (2×2 heat‑map)
         • ROC curve with AUC
         • Precision‑Recall curve with Average Precision (AP)
        Images are saved into `self._base_path` and also shown if `self._show` is True.
        """

        # --- prepare data ----------------------------------------------------
        labels = [int(l[0]) if isinstance(l, (list, tuple)) else int(l) for l in self._labels_list]
        scores = self._output_list  # raw logits or probabilities

        # --- Precision‑Recall curve --------------------------------------
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{title} – Precision‑Recall")
        plt.legend(loc="lower left")
        self.show(plt, os.path.join(self._base_path, f"{title}_pr.png"))

        return super().draw_total_result(title)


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
            labels_total = self._labels_list
            labels_one_hot = one_hot(torch.tensor(self._labels_list), num_classes=n_class).tolist()
        else:
            labels_total = torch.argmax(torch.tensor(self._labels_list), dim=1).tolist()
            labels_one_hot = self._labels_list

        epoch_corrects = torch.sum(torch.tensor(self._preds_list).cpu() == torch.tensor(self._labels_list).cpu())
        epoch_f1 = f1_score(labels_total, self._preds_list, average='weighted')
        epoch_loss = self._epoch_loss / len(self._preds_list)
        epoch_acc = epoch_corrects.double() / len(self._preds_list)

        if draw:
            y_score = np.array(self._output_list)
            if isinstance(self._labels_list[0], int):
                y_true = one_hot(torch.tensor(self._labels_list), num_classes=n_class).numpy()
            else:
                y_true = np.array(self._labels_list)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_class):
                fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f"micro-avg AUC = {roc_auc['micro']:.3f}",
                     color="navy", linestyle=":", linewidth=2)
            for i in range(n_class):
                plt.plot(fpr[i], tpr[i], linewidth=1.2)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{title} – ROC Curves")
            self.show(plt, os.path.join(self._base_path, f"{title}_roc{self._save_num}.png"))
            # Remove detailed per-threshold ROC/PR CSV; instead, write summary CSV
            summary_csv_path = os.path.join(self._base_path, f"summary_scores{self._save_num}.csv")
            self._save_num += 1

            with open(summary_csv_path, "w", newline="") as summary_file:
                writer = csv.writer(summary_file)
                writer.writerow(["class", "auc", "average_precision", "best_f1", "best_f1_threshold"])
                for i in range(n_class):
                    # ROC AUC
                    fpr_vals, tpr_vals, roc_thresholds = roc_curve(y_true[:, i], y_score[:, i])
                    roc_auc_val = auc(fpr_vals, tpr_vals)

                    # PR AP
                    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true[:, i], y_score[:, i])
                    ap_val = average_precision_score(y_true[:, i], y_score[:, i])

                    # Best F1
                    f1_scores = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-8)
                    best_idx = np.argmax(f1_scores)
                    best_f1 = f1_scores[best_idx]
                    best_thresh = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 1.0

                    writer.writerow([i, roc_auc_val, ap_val, best_f1, best_thresh])

        if is_merged:
            self._accuracy_list.append(epoch_acc.item())
            self._loss_list.append(epoch_loss)
            self._score_list.append(epoch_f1)

        if write:
            cm = confusion_matrix(labels_total, self._preds_list, labels=list(range(n_class)))

            if n_class >= 100:
                group_count = 10
            else:
                group_count = None

            if group_count:
                group_size = n_class // group_count
                cm_grouped = np.zeros((group_count, group_count), dtype=int)
                for i in range(group_count):
                    for j in range(group_count):
                        cm_grouped[i, j] = cm[
                                           i * group_size: min((i + 1) * group_size, n_class),
                                           j * group_size: min((j + 1) * group_size, n_class)
                                           ].sum()
                cm_to_plot = cm_grouped
                tick_labels = [f"{i * group_size}-{min((i + 1) * group_size - 1, n_class - 1)}" for i in
                               range(group_count)]
            else:
                cm_to_plot = cm
                tick_labels = [str(label) for label in range(n_class)]

            # visualise it
            plt.figure(figsize=(8, 8))
            plt.imshow(cm_to_plot, interpolation='nearest', cmap=plt.cm.Greens)
            plt.title("Confusion Matrix (Grouped)" if group_count else "Confusion Matrix")
            plt.colorbar()

            tick_marks = np.arange(len(tick_labels))
            plt.xticks(tick_marks, tick_labels, rotation=45)
            plt.yticks(tick_marks, tick_labels)

            thresh = cm_to_plot.max() / 2
            for i, j in itertools.product(range(cm_to_plot.shape[1]), range(cm_to_plot.shape[0])):
                plt.text(i, j, cm_to_plot[j, i],
                         horizontalalignment='center',
                         color='white' if cm_to_plot[j, i] > thresh else 'red')

            plt.tight_layout()
            plt.xlabel('Predictions')
            plt.ylabel('Real Values')
            self.show(plt, os.path.join(self._base_path, "{0}_cm".format(title) + str(self._save_num) + ".png"))

            # Sparse confusion scatter plot
            rows, cols = np.where((cm > 0) & (np.arange(n_class)[:, None] != np.arange(n_class)))
            vals = cm[rows, cols]

            plt.figure(figsize=(10, 8))
            plt.scatter(rows, cols, s=np.clip(vals, 10, 300), c=vals, cmap='cool', alpha=0.6)
            plt.plot([0, n_class], [0, n_class], 'k--', alpha=0.3)
            plt.xlabel('True Class')
            plt.ylabel('Predicted Class')
            plt.title(f"{title} – Sparse Confusion Scatter Plot")
            plt.colorbar(label='Count')
            self.show(plt, os.path.join(self._base_path, f"{title}_sparse_cm{self._save_num}.png"))

            write_path = os.path.join(self._base_path, '{0}.txt'.format(title))
            with open(write_path, 'w', encoding='utf8') as f:
                lines = [
                    "The number of {0} data: {1}\n".format(title, len(self._preds_list)),
                    "Accuracy: {0}\n".format(epoch_acc),
                    "Loss: {0}\n".format(epoch_loss),
                    "F1 Score: {0}\n".format(epoch_f1),
                    "Confusion Matrix: {0}".format(cm_to_plot.tolist())
                ]
                f.writelines(lines)

        print('Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1))

        return epoch_loss, epoch_acc, epoch_f1

    def draw_total_result(self, title: str = "Train"):
        """
        Draws multi‑class visualisations:

         • Precision‑Recall curves per class + micro‑average
         • Confusion‑matrix + training curves (delegated to super)
        """

        # --- prepare -----------------------------------------------------------------
        # labels: index form or one‑hot
        if len(self._labels_list) == 0:
            return super().draw_total_result(title)

        n_class = len(self._output_list[0])
        y_score = np.array(self._output_list)  # shape (N, C)

        # convert labels to one‑hot array (N, C)
        if isinstance(self._labels_list[0], int):
            y_true = one_hot(torch.tensor(self._labels_list), num_classes=n_class).numpy()
        else:
            y_true = np.array(self._labels_list)

        # --- Precision‑Recall curves -------------------------------------------------
        precision = dict()
        recall = dict()
        avg_prec = dict()
        for i in range(n_class):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
            avg_prec[i] = average_precision_score(y_true[:, i], y_score[:, i])

        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true.ravel(), y_score.ravel())
        avg_prec["micro"] = average_precision_score(y_true, y_score, average="micro")

        plt.figure()
        plt.plot(recall["micro"], precision["micro"],
                 label=f"micro‑avg AP = {avg_prec['micro']:.3f}",
                 color="navy", linestyle=":", linewidth=2)

        for i in range(n_class):
            plt.plot(recall[i], precision[i], linewidth=1.2,
                     label=f"class {i} (AP = {avg_prec[i]:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{title} – Precision‑Recall Curves")
        # plt.legend(fontsize="small", loc="lower left")  # Legend removed as per instruction
        self.show(plt, os.path.join(self._base_path, f"{title}_pr.png"))

        pr_csv_path = os.path.join(self._base_path, "pr_scores.csv")

        # Export PR values
        with open(pr_csv_path, "w", newline="") as pr_file:
            writer = csv.writer(pr_file)
            writer.writerow(["class", "threshold", "precision", "recall"])
            for i in range(n_class):
                precision_vals, recall_vals, thresholds = precision_recall_curve(y_true[:, i], y_score[:, i])
                # thresholds has len n-1, precision/recall have len n; append 1.0 to thresholds for last
                for p, r, t in zip(precision_vals, recall_vals, list(thresholds) + [1.0]):
                    writer.writerow([i, t, p, r])

        # delegate to base for learning curves
        return super().draw_total_result(title)


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

        return epoch_loss, epoch_mse, math.sqrt(epoch_mse)

    def draw_total_result(self, title: str = "Train"):
        """
        Visualises regression performance:

         • Scatter plot (True vs. Predicted) with y = x reference
         • Residuals histogram
         • Absolute error over sample index

        Saves figures into `self._base_path` and returns the usual
        loss / accuracy / score lists from the base class.
        """
        if len(self._labels_list) == 0:
            return super().draw_total_result(title)

        # --- prepare data ----------------------------------------------------
        y_true = np.array([l[0] if isinstance(l, (list, tuple)) else l for l in self._labels_list])
        y_pred = np.array(self._preds_list)
        residuals = y_pred - y_true

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 1) Scatter – True vs Predicted
        plt.figure()
        plt.scatter(y_true, y_pred, s=10, alpha=0.5)
        min_v, max_v = y_true.min(), y_true.max()
        plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{title} – True vs Predicted\nMSE={mse:.3f}  R²={r2:.3f}")
        self.show(plt, os.path.join(self._base_path, f"{title}_scatter.png"))

        # 2) Residuals histogram
        plt.figure()
        plt.hist(residuals, bins=30, edgecolor="black")
        plt.xlabel("Residual (Pred − True)")
        plt.ylabel("Frequency")
        plt.title(f"{title} – Residual Distribution")
        self.show(plt, os.path.join(self._base_path, f"{title}_residual_hist.png"))

        # 3) Absolute error per sample
        plt.figure()
        plt.plot(np.abs(residuals))
        plt.xlabel("Sample Index")
        plt.ylabel("|Residual|")
        plt.title(f"{title} – Absolute Error per Sample")
        self.show(plt, os.path.join(self._base_path, f"{title}_abs_error.png"))

        # Delegate to base class for combined loss / acc / score curves
        return super().draw_total_result(title)


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
        return 0, 0, 0
