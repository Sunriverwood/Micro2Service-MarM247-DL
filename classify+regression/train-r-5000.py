import os
import sys
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from Alex_model import AlexNet  # 确保 AlexNet 输出层调整为回归任务（num_classes=2）


# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        # 遍历所有子文件夹
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            temp_time = folder_name.split("_")
            try:
                temperature = float(temp_time[0].replace("℃", ""))
                time_exposed = float(temp_time[1].replace("h", ""))
            except ValueError:
                print(f"Skipping invalid folder name: {folder_name}")
                continue
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if img_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.samples.append(img_path)
                    self.labels.append([temperature, time_exposed])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


# 自定义归一化损失函数
class NormalizedLoss(nn.Module):
    def __init__(self, temp_mean, temp_std, time_mean, time_std):
        super(NormalizedLoss, self).__init__()
        self.temp_mean = temp_mean
        self.temp_std = temp_std
        self.time_mean = time_mean
        self.time_std = time_std

    def forward(self, predictions, targets):
        temp_pred = (predictions[:, 0] - self.temp_mean) / self.temp_std
        temp_target = (targets[:, 0] - self.temp_mean) / self.temp_std
        time_pred = (predictions[:, 1] - self.time_mean) / self.time_std
        time_target = (targets[:, 1] - self.time_mean) / self.time_std

        temp_loss = nn.functional.mse_loss(temp_pred, temp_target)
        time_loss = nn.functional.mse_loss(time_pred, time_target)

        total_loss = (temp_loss + time_loss) / 2
        return total_loss


# 计算 R^2 (决定系数)
def calculate_r2(predictions, targets):
    ss_total = torch.sum((targets - torch.mean(targets, dim=0)) ** 2, dim=0)
    ss_residual = torch.sum((targets - predictions) ** 2, dim=0)
    r2 = 1 - ss_residual / ss_total
    return r2.mean().item()  # 返回平均 R^2 值


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(),".."))
    print(data_root)
    image_path = os.path.join(data_root, "superalloy_data", "bingxing1")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = CustomDataset(root_dir=os.path.join(image_path, "train"),
                                  transform=data_transform["train"])
    validate_dataset = CustomDataset(root_dir=os.path.join(image_path, "val"),
                                     transform=data_transform["val"])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=4, shuffle=False, num_workers=0)

    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(validate_dataset)))

    all_labels = torch.tensor(train_dataset.labels)
    temp_mean, temp_std = all_labels[:, 0].mean().item(), all_labels[:, 0].std().item()
    time_mean, time_std = all_labels[:, 1].mean().item(), all_labels[:, 1].std().item()
    print(f"Temperature mean: {temp_mean}, std: {temp_std}")
    print(f"Time mean: {time_mean}, std: {time_std}")

    net = AlexNet(num_outputs=2, init_weights=True)
    net.to(device)

    loss_function = NormalizedLoss(temp_mean, temp_std, time_mean, time_std)
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 100
    save_path = 'regression_5000.pth'
    best_loss = float('inf')
    train_steps = len(train_loader)

    for epoch in range(epochs):
        net.train()
        t1 = time.perf_counter()
        running_loss = 0.0
        all_preds, all_labels = [], []
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_labels.append(labels)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        train_r2 = calculate_r2(all_preds, all_labels)

        net.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels_batch = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels_batch.to(device))
                val_loss += loss.item()

                val_preds.append(outputs.cpu())
                val_labels.append(val_labels_batch)

        val_preds = torch.cat(val_preds, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        val_r2 = calculate_r2(val_preds, val_labels)

        val_loss /= len(validate_loader)
        print(f"[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}, train_r2: {train_r2:.3f}, "
              f"val_loss: {val_loss:.3f}, val_r2: {val_r2:.3f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), save_path)
        print(time.perf_counter() - t1)
        print()

    print('Finished Training')


if __name__ == '__main__':
    main()
