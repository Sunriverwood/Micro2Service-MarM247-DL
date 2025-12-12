import os
import sys
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model_Inception import InceptionV3


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd()))  # get data root path
    print(data_root)
    image_path = os.path.join(data_root, "..","superalloy_data", "heat_exposure_photos")  # data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)


    image_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in image_list.items())#将索引和值反过来
    json_str = json.dumps(cla_dict, indent=4)#indent是字符串格式化时缩进几个字符，使输出的JSON字符更易读
    with open('class_inception.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = 8
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw,pin_memory=True)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])

    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=True,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = InceptionV3(num_outputs=56, pretrained=True, dropout_p=0.5, task_type='multiclass_classification')

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 1000
    save_path = 'parameter/all-classify-inception.pth'
    best_loss = float('inf')
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        t1 = time.perf_counter()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels.to(device)).sum().item()
            total += labels.size(0)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        train_accuracy = correct / total  # 计算训练集的准确率
        train_loss = running_loss / train_steps

        # validate
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

        print(f"[epoch {epoch + 1}] train_loss: {train_loss:.3f}, train_accurate: {train_accuracy:.3f}, "
              f"val_loss: {val_loss:.3f}, val_accurate: {val_accurate:.3f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), save_path)
        print(time.perf_counter() - t1)
        print()
    print('Finished Training')


if __name__ == '__main__':
    main()
