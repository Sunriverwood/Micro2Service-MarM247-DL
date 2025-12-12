import os
import sys
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import json
from Alex_model import AlexNet  # 确保 AlexNet 输出层调整为回归任务（num_classes=1）


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(),"..",".."))
    image_path = os.path.join(data_root, "superalloy_data", "heat_exposure_photos")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,"train"),transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path,"val"),transform=data_transform["val"])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=4, shuffle=False, num_workers=0)
    val_num = len(validate_dataset)
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(validate_dataset)))

    image_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in image_list.items())#将索引和值反过来
    json_str = json.dumps(cla_dict, indent=4)#indent是字符串格式化时缩进几个字符，使输出的JSON字符更易读
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    net = AlexNet(num_outputs=9, init_weights=True)
    net.to(device)



    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 100
    save_path = 'AlexNet_regression_1000CC.pth'
    best_loss = float('inf')

    for epoch in range(epochs):
        net.train()
        t1 = time.perf_counter()
        train_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=80)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(0)
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        train_accurate = correct / total  # 计算训练集的准确率
        train_loss /= len(train_loader)

        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()

        val_accurate = acc / val_num
        val_loss /= len(validate_loader)

        print(f"[epoch {epoch + 1}] train_loss: {train_loss:.3f}, train_accurate: {train_accurate:.3f}, "
              f"val_loss: {val_loss:.3f}, val_accurate: {val_accurate:.3f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), save_path)
        print(time.perf_counter() - t1)
        print()

    print('Finished Training')

if __name__ == '__main__':
    main()
