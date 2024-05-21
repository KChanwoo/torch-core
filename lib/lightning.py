import lightning as L
from torch.utils.data import Dataset, DataLoader
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from lib.core import GanCore


class PLModel(L.LightningModule):
    def __init__(self, core):
        super(PLModel, self).__init__()
        self.core = core

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, loss = self.core.train_step(self.core.get_model(), inputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, loss = self.core.train_step(self.core.get_model(), inputs, labels)
        self.core.get_scorer().add_batch_result(outputs, labels, loss)
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, loss = self.core.train_step(self.core.get_model(), inputs, labels)
        self.core.get_scorer().add_batch_result(outputs, labels, loss)
        self.log('test_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return [self.core.get_optim()], [self.core.get_scheduler()] if self.core.get_scheduler() is not None else []

    def on_train_end(self):
        self.core.get_scorer().draw_total_result()

    def on_validation_epoch_end(self):
        self.core.get_scorer().get_epoch_result(True, True)

    def on_test_end(self):
        self.core.get_scorer().get_epoch_result(True, True, title='Test')
        self.core.get_scorer().draw_total_result()


class GanPLModel(PLModel):
    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, labels = batch
        outputs, loss = self.core.train_step(self.core.get_model(), inputs, labels, optimizer_idx == 0)
        self.log('train_loss', loss)
        return loss

    def __init__(self, core: 'GanCore'):
        super(GanPLModel, self).__init__(core)
        self.core = core

    def configure_optimizers(self):
        return [self.core.get_d_model().get_optim(), self.core.get_optim()], []


class PLDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, train_dataset: Dataset = None, valid_dataset: Dataset = None, test_dataset: Dataset = None, collate_fn=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn) if self.train_dataset is not None else None

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn) if self.valid_dataset is not None else None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn) if self.test_dataset is not None else None
