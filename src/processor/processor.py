import pytorch_lightning as L
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class ImgDataset(Dataset):
    def __init__(self, predictors_root, targets_root, transform=None):
        self.predictors_root = Path(predictors_root)
        self.targets_root = Path(targets_root)
        self.transform = transform
        self.predictors_dataset = list(self.predictors_root.glob("*.jpg"))
        self.targets_dataset = list(self.targets_root.glob("*.jpg"))
        assert len(self.predictors_dataset) == len(
            self.targets_dataset
        ), "Datasets must have the same length"

    def __len__(self):
        return len(self.predictors_dataset)

    def __getitem__(self, idx):
        predictors_path = self.predictors_dataset[idx]
        targets_path = self.targets_dataset[idx]
        predictors_img = Image.open(predictors_path).convert("L")
        targets_img = Image.open(targets_path).convert("RGB")
        if self.transform:
            predictors_img = self.transform(predictors_img)
            targets_img = self.transform(targets_img)
        return predictors_img, targets_img


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        predictors_dir="../data/gray/",
        targets_dir="../data/color/",
        resolution=(224, 224),
        batch_size=32,
        train_split_ratio=0.7,
    ):
        super(DataModule, self).__init__()
        self.predictors_dir = predictors_dir
        self.targets_dir = targets_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.train_split_ratio = train_split_ratio
        self.transform = self.get_transform()
        self.dataset = None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.setup()

    def setup(self):
        self.dataset = ImgDataset(
            predictors_root=self.predictors_dir,
            targets_root=self.targets_dir,
            transform=self.transform,
        )
        dataset_size = len(self.dataset)

        train_size = int(self.train_split_ratio * dataset_size)
        val_size = (dataset_size - train_size) // 2
        test_size = dataset_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def plot_sample(self, idx, figsize=(8, 8)):
        X, y = self.dataset[idx]
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(X[0], cmap="gray")
        ax[0].set_title("Predictor")
        ax[0].axis("off")
        ax[1].imshow(y.permute(1, 2, 0))
        ax[1].set_title("Target")
        ax[1].axis("off")
        plt.show()

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(self.resolution),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[...], std=[...]),
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
