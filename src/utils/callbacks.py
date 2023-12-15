import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)


class Callbacks(L.Callback):
    def __init__(self, ckpt_path, ckpt_file, early_stop_patience, lr_monitor):
        self.ckpt_path = ckpt_path
        self.ckpt_file = ckpt_file
        self.early_stop_patience = early_stop_patience
        self.lr_monitor = lr_monitor
        self.model_checkpoint = None
        self.early_stopping = None
        self.callbacks = None

    def on_fit_start(self, trainer, pl_module):
        self.model_checkpoint = ModelCheckpoint(
            dirpath=self.ckpt_path,
            filename=self.ckpt_file,
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

        self.early_stopping = EarlyStopping(
            patience=self.early_stop_patience,
            monitor="val_loss",
            mode="min",
            min_delta=1e-6,
        )

        self.callbacks = [self.model_checkpoint, self.early_stopping]

        if self.lr_monitor:
            lr_monitor = LearningRateMonitor(logging_interval="epoch")
            self.callbacks.append(lr_monitor)

    def on_train_epoch_end(self, trainer, pl_module):
        for callback in self.callbacks:
            callback.on_train_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        for callback in self.callbacks:
            callback.on_validation_epoch_end(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        pass
