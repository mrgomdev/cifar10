from typing import Tuple, Optional, Callable, List
import os
import glob
import enum
import pickle
from tqdm import tqdm

import einops
from PIL import Image

import wandb

import numpy as np
import torch.utils.data
import torchvision.transforms.functional


class Cifar10Label(enum.Enum):
    airplane = 0
    automobile = 1
    bird = 2
    cat = 3
    deer = 4
    dog = 5
    frog = 6
    horse = 7
    ship = 8
    truck = 9

    @classmethod
    def from_label(cls, label: int) -> 'Cifar10Label':
        return cls(label)

    @classmethod
    def from_name(cls, name: str) -> 'Cifar10Label':
        return cls[name]



class WandbLogKeys(str, enum.Enum):
    train_accuracy = 'train_accuracy'
    train_batch_accuracy = 'train_batch_accuracy'
    test_accuracy = 'test_accuracy'
    train_loss = 'train_loss'
    train_batch_loss = 'train_batch_loss'
    test_loss = 'test_loss'

    train_confusion_matrix = 'train_confusion_matrix'
    test_confusion_matrix = 'test_confusion_matrix'
WANDB_LOG_PERIOD_BATCHES = 50


def cutmix(image_batch: torch.Tensor, label_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert image_batch.ndim == 4
    assert label_batch.ndim == 1

    # 0. Shuffle
    # 1. combination ratio lamb = beta(alpha, alpha) = uniform(0, 1)
    # 2. bounding box B = (x, y, w, h) = (uniform(0, 1), uniform(0, 1), sqrt(1 - lamb), sqrt(1 - lamb))
    # 3. Get Mask.
    # 4. Mask
    shuffle_perm = torch.randperm(len(image_batch))

    lamb = np.random.uniform(0, 1)

    x, y = np.random.uniform(0, 1), np.random.uniform(0, 1)
    w, h = np.sqrt(1 - lamb), np.sqrt(1 - lamb)
    top, bottom = max(0, int(y * image_batch.shape[3])), min(image_batch.shape[3], int(y * image_batch.shape[3] + h * image_batch.shape[3]))
    left, right = max(0, int(x * image_batch.shape[2])), min(image_batch.shape[2], int(x * image_batch.shape[2] + w * image_batch.shape[2]))
    mask = torch.zeros_like(image_batch)
    mask[:, :, top:bottom, left:right] = 1

    image_batch = image_batch * mask + image_batch[shuffle_perm] * (1 - mask)

    lamb = (right - left) * (bottom - top) / (image_batch.shape[2] * image_batch.shape[3])
    label_batch = torch.nn.functional.one_hot(label_batch, num_classes=10)
    label_batch = label_batch * lamb + label_batch[shuffle_perm] * (1 - lamb)

    return image_batch, label_batch


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, images: List[Image.Image], transform: Optional[Callable[[Image.Image], torch.Tensor]] = torchvision.transforms.ToTensor()):
        super(ImagesDataset, self).__init__()
        self.images = images
        self.transform = transform

    def __getitem__(self, index: int) -> np.ndarray:
        image = self.images[index]
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self) -> int:
        return len(self.images)


class Cifar10Dataset(ImagesDataset):
    def __init__(self, data_paths: List[str], transform: Optional[Callable[[np.ndarray], torch.Tensor]] = torchvision.transforms.ToTensor(), target_transform: Optional[Callable[[int], torch.Tensor]] = None):
        self.data_paths = list(data_paths)
        images = []
        labels = []

        to_pil = torchvision.transforms.ToPILImage(mode='RGB')

        for path in self.data_paths:
            with open(path, 'rb') as file:
                batch = pickle.load(file, encoding='bytes')
                each_data = einops.rearrange(batch[b'data'], 'batch (channel height width) -> batch height width channel', channel=3, height=32, width=32)
                each_labels = np.asarray(batch[b'labels'], dtype=np.int64)

                #TODO: Filter specific class

                images.extend(map(to_pil, each_data))
                labels.extend(list(each_labels))
        assert len(images) == len(labels)

        super(Cifar10Dataset, self).__init__(images=images, transform=transform)
        self.target_transform = target_transform
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image, label = self.images[index], self.labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    @property
    def images_dataset(self) -> ImagesDataset:
        return ImagesDataset(images=self.images, transform=self.transform)


class TrainDataset(Cifar10Dataset):
    def __init__(self, data_dir_path: str, transform: Optional[Callable[[np.ndarray], torch.Tensor]] = torchvision.transforms.ToTensor(), target_transform: Optional[Callable[[int], torch.Tensor]] = None):
        paths = sorted(glob.glob(os.path.join(data_dir_path, 'data_batch_*')))
        super(TrainDataset, self).__init__(data_paths=paths, transform=transform, target_transform=target_transform)
        self.data_dir_path = data_dir_path


class TestDataset(Cifar10Dataset):
    def __init__(self, data_dir_path: str, transform: Optional[Callable[[np.ndarray], torch.Tensor]] = torchvision.transforms.ToTensor(), target_transform: Optional[Callable[[int], torch.Tensor]] = None):
        paths = sorted(glob.glob(os.path.join(data_dir_path, 'test_batch')))
        super(TestDataset, self).__init__(data_paths=paths, transform=transform, target_transform=target_transform)
        self.data_dir_path = data_dir_path


################

class CustomBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv1 = torch.nn.utils.spectral_norm(self.conv1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = torch.nn.utils.spectral_norm(self.conv2)

        self.conv_stride = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.relu = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.relu(identity + x)
        x = self.relu(self.conv_stride(x))

        return x


class CustomResNet(torch.nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.channels = channels

        self.conv1 = torch.nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)
        self.layer1 = CustomBlock(channels=channels, kernel_size=3, stride=2, padding=0)
        self.layer1_2 = CustomBlock(channels=channels, kernel_size=3, stride=1, padding=0)
        self.layer2 = CustomBlock(channels=channels, kernel_size=3, stride=2, padding=0)
        self.layer3 = CustomBlock(channels=channels, kernel_size=3, stride=1, padding=0)
        self.layer4 = CustomBlock(channels=channels, kernel_size=3, stride=1, padding=0)
        self.top = torch.nn.Linear(channels * 6 * 6, num_classes)

        self.relu = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.layer1(x)
        x = self.layer1_2(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = torch.flatten(x, 1)
        x = self.top(x)

        return x

class ShallowCNN(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.top = torch.nn.Linear(16 * 32 * 32, 10)

        print(f"total params: {sum(parameter.numel() for parameter in self.parameters())}")

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.top(x)
        assert x.shape == (len(x), 10)

        return x
class SpectralNormBasicBlock(torchvision.models.resnet.BasicBlock):
    def __init__(self, *args, **kwargs):
        super(SpectralNormBasicBlock, self).__init__(*args, **kwargs)
        self.conv1 = torch.nn.utils.spectral_norm(self.conv1)
        self.conv2 = torch.nn.utils.spectral_norm(self.conv2)

class SpectralNormLeakyBasicBlock(SpectralNormBasicBlock):
    def __init__(self, *args, **kwargs):
        super(SpectralNormLeakyBasicBlock, self).__init__(*args, **kwargs)

        self.relu = torch.nn.LeakyReLU(inplace=True)

def resnet18_spectral_norm_leaky() -> torchvision.models.resnet.ResNet:
    return torchvision.models.ResNet(SpectralNormLeakyBasicBlock, [2, 2, 2, 2], num_classes=10)
def resnet18_spectral_norm() -> torchvision.models.resnet.ResNet:
    return torchvision.models.ResNet(SpectralNormBasicBlock, [2, 2, 2, 2], num_classes=10)
def resnet18_no_pretrained() -> torchvision.models.resnet.ResNet:
    return torchvision.models.resnet18(num_classes=10)
def resnet34_no_pretrained() -> torchvision.models.resnet.ResNet:
    return torchvision.models.resnet34(num_classes=10)
def custom_resnet() -> CustomResNet:
    return CustomResNet(channels=24, num_classes=10)
Classifier = resnet18_no_pretrained

# Train
def train(classifier: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, epochs: int):
    global trained_batch_accumulated
    old_training = classifier.training

    classifier.train()
    # progress_bar = tqdm(range(epochs), position=1)
    progress_bar = range(epochs)
    for epoch in progress_bar:
        for i, (images, labels) in enumerate(train_dataloader):
            imagex, labels = cutmix(images, labels)
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            trained_batch_accumulated += 1

            if trained_batch_accumulated % WANDB_LOG_PERIOD_BATCHES == 0:
                wandb.log({WandbLogKeys.train_batch_accuracy: torch.sum(torch.max(outputs, 1).indices == torch.max(labels, 1).indices) / len(labels)}, step=trained_batch_accumulated)
                wandb.log({WandbLogKeys.train_batch_loss: loss.item()}, step=trained_batch_accumulated)

    classifier.train(mode=old_training)


def predict(classifier: torch.nn.Module, dataset: ImagesDataset, batch_size: int) -> torch.Tensor:
    if isinstance(dataset[0].shape[0], tuple):
        dataset = dataset.images_dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    outputs = []

    old_training = classifier.training
    classifier.eval()
    with torch.no_grad():
        for images_batch in dataloader:
            if isinstance(images_batch, tuple):
                images_batch = images_batch[0]
            images_batch = images_batch.cuda()
            outputs.extend(classifier(images_batch).cpu())

    classifier.train(mode=old_training)

    outputs = torch.stack(outputs)
    assert outputs.shape == (len(dataset), 10)

    return outputs


# Test
def eval(classifier: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader) -> float:
    old_training = classifier.training

    labels = []
    predicteds = []

    classifier.eval()
    with torch.no_grad():
        for images_batch, labels_batch in test_dataloader:
            images_batch = images_batch.cuda()
            labels_batch = labels_batch.cuda()

            outputs = classifier(images_batch)
            predicted = torch.max(outputs, 1).indices

            labels.extend(labels_batch.cpu().numpy())
            predicteds.extend(predicted.cpu().numpy())

    classifier.train(mode=old_training)

    labels = np.asarray(labels)
    predicteds = np.asarray(predicteds)
    accuracy = np.sum(labels == predicteds) / len(labels)

    return accuracy

# Confusion matrix
def get_confusion_matrix(labels: np.ndarray, predicteds: np.ndarray) -> np.ndarray:
    assert labels.shape == predicteds.shape
    assert len(labels.shape) == 1

    matrix = np.zeros((10, 10), dtype=np.int64)
    for label, predicted in zip(labels, predicteds):
        matrix[label, predicted] += 1

    return matrix


def log_confusion_matrix(key: str, classifier: torch.nn.Module, dataset: Cifar10Dataset, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    assert key.endswith('_confusion_matrix'), f"postfix must be '_confusion_matrix', but '{key}'"
    predicteds = predict(classifier=classifier, dataset=dataset.images_dataset, batch_size=batch_size)
    predicteds = torch.max(predicteds, 1).indices.cpu().numpy()
    labels = np.asarray(dataset.labels)
    wandb.log({key: wandb.plot.confusion_matrix(y_true=labels, preds=predicteds, class_names=[Cifar10Label(idx).name for idx in range(10)])}, step=trained_batch_accumulated)

    return predicteds, labels


if __name__ == '__main__':
    # for Classifier in [ShallowCNN, resnet18_no_pretrained, resnet34_no_pretrained, resnet18_spectral_norm, resnet18_spectral_norm_leaky, custom_resnet]:
    for Classifier in [custom_resnet]:
        classifier = Classifier()
        classifier = classifier.cuda()

        wandb.init(
            project='cifar10',
            config={
                "learning_rate": 1e-4,
                "model_name": str(Classifier.__name__),
                "batch_size": 128,
                "cutmix": 'Right',
                "activation": "leaky_relu",
                "classifier_total_params": sum(parameter.numel() for parameter in classifier.parameters())
            }
        )
        # soft_labeler =
        # classifier.load_state_dict(torch.load('/home/gimun-lee/PycharmProjects/cifar_classification/wandb/run-20230913_131945-a4ov9mb6/files/classifier.pth'))
        trained_batch_accumulated = 0 #global

        try:
            cifar_dir_path = './data/cifar-10-batches-py'
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.1)
            ])
            train_dataset = TrainDataset(cifar_dir_path, transform=transform)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=True, num_workers=0)
            train_eval_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=False, num_workers=0)

            test_dataset = TestDataset(cifar_dir_path, transform=transform)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=wandb.config['batch_size'], shuffle=False, num_workers=0)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(classifier.parameters(), lr=wandb.config['learning_rate'])

            wandb.log({WandbLogKeys.test_accuracy: eval(classifier=classifier, test_dataloader=test_dataloader)}, step=trained_batch_accumulated)
            progress_bar = tqdm(range(25), position=0, desc='epochs')
            for idx in progress_bar:
                train(classifier=classifier, train_dataloader=train_dataloader, epochs=1)
                wandb.log({WandbLogKeys.train_accuracy: eval(classifier=classifier, test_dataloader=train_eval_dataloader)}, step=trained_batch_accumulated)
                wandb.log({WandbLogKeys.test_accuracy: eval(classifier=classifier, test_dataloader=test_dataloader)}, step=trained_batch_accumulated)

                log_confusion_matrix(key=WandbLogKeys.train_confusion_matrix, classifier=classifier, dataset=train_dataset, batch_size=wandb.config['batch_size'])
                log_confusion_matrix(key=WandbLogKeys.test_confusion_matrix, classifier=classifier, dataset=test_dataset, batch_size=wandb.config['batch_size'])
        except (Exception, KeyboardInterrupt) as e:
            print(repr(e))
            torch.save(classifier.state_dict(), f"{wandb.run.dir}/classifier.pth")
            wandb.finish(exit_code=1)
            raise
        else:
            torch.save(classifier.state_dict(), f"{wandb.run.dir}/classifier.pth")
            wandb.finish()
