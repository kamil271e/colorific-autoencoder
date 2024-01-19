import os
import sys

sys.path.append("..")
from src.utils.callbacks import Callbacks


class Config:
    def __init__(self):
        # DATA MODULE
        self.TRAIN_RATIO = 0.7
        self.BATCH_SIZE = 32
        self.IMAGE_RESOLUTION = (160, 160)  # x, y % 32 == 0
        self.PREDICTORS_DIR = "../data/gray/"
        self.TARGETS_DIR = "../data/color/"
        self.IMAGE_EXT = "jpg"

        # MODEL ARCHITECTURE
        self.IN_CHANNELS = 1
        self.OUT_CHANNELS = 3
        self.UNIT = 16

        # TRAINING DETAILS
        self.LR = 1e-3
        self.EPOCHS = 50
        self.EARLY_STOP_PATIENCE = 5
        self.LR_MONITOR = True
        self.SCHEDULER_STEP_SIZE = 10
        self.SCHEDULER_GAMMA = 0.9
        self.CKPT_DIR = "checkpoints"
        self.CKPT_NAME = f"b{self.BATCH_SIZE}_u{self.UNIT}_e{self.EPOCHS}"
        self.CKPT_PATH = os.path.join(self.CKPT_DIR, self.CKPT_NAME + ".ckpt")
        self.FAST_DEV_RUN = False
        self.NUM_SANITY_VAL_STEPS = 0
        self.LOG_STEPS = 20
        self.WEIGHT_DECAY = 0
        self.DROPOUT_RATE = 0

    def get_trainer_args(self):
        return {
            "max_epochs": self.EPOCHS,
            "fast_dev_run": self.FAST_DEV_RUN,
            "num_sanity_val_steps": self.NUM_SANITY_VAL_STEPS,
            "log_every_n_steps": self.LOG_STEPS,
            "callbacks": self.get_callbacks(),
        }

    def get_train_wrapper_args(self):
        return {
            "in_channels": self.IN_CHANNELS,
            "out_channels": self.OUT_CHANNELS,
            "unit": self.UNIT,
            "lr": self.LR,
            "weight_decay": self.WEIGHT_DECAY,
            "dropout_rate": self.DROPOUT_RATE,
        }

    def get_callbacks(self):
        callbacks = Callbacks(
            ckpt_path=self.CKPT_DIR,
            ckpt_file=self.CKPT_NAME,
            early_stop_patience=self.EARLY_STOP_PATIENCE,
            lr_monitor=self.LR_MONITOR,
        )
        return callbacks.callbacks
