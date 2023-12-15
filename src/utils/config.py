import os
import sys

sys.path.append("..")
from src.utils.callbacks import Callbacks


class Config:
    # DATA MODULE
    TRAIN_RATIO = 0.7
    BATCH_SIZE = 32
    IMAGE_RESOLUTION = (160, 160)  # x, y % 32 == 0
    PREDICTORS_DIR = "../data/gray/"
    TARGETS_DIR = "../data/color/"
    IMAGE_EXT = "jpg"

    # MODEL ARCHITECTURE
    IN_CHANNELS = 1
    OUT_CHANNELS = 3
    UNIT = 16

    # TRAINING DETAILS
    LR = 1e-3
    EPOCHS = 200
    EARLY_STOP_PATIENCE = 5
    LR_MONITOR = True
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.9
    CKPT_DIR = "checkpoints"
    CKPT_NAME = f"b{BATCH_SIZE}_u{UNIT}_e{EPOCHS}"
    CKPT_PATH = os.path.join(CKPT_DIR, CKPT_NAME + ".ckpt")
    FAST_DEV_RUN = False
    NUM_SANITY_VAL_STEPS = 0
    LOG_STEPS = 20

    @classmethod
    def get_trainer_args(cls):
        return {
            "max_epochs": cls.EPOCHS,
            "fast_dev_run": cls.FAST_DEV_RUN,
            "num_sanity_val_steps": cls.NUM_SANITY_VAL_STEPS,
            "log_every_n_steps": cls.LOG_STEPS,
            "callbacks": cls.get_callbacks(),
        }

    @classmethod
    def get_train_wrapper_args(cls):
        return {
            "in_channels": cls.IN_CHANNELS,
            "out_channels": cls.OUT_CHANNELS,
            "unit": cls.UNIT,
            "lr": cls.LR,
        }

    @classmethod
    def get_callbacks(cls):
        callbacks = Callbacks(
            ckpt_path=cls.CKPT_DIR,
            ckpt_file=cls.CKPT_NAME,
            early_stop_patience=cls.EARLY_STOP_PATIENCE,
            lr_monitor=cls.LR_MONITOR,
        )
        return callbacks.callbacks
