import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from Alex_model import AlexNet
import shutil


class ImageDataset(Dataset):
    """自定义数据集，用于加载测试集图片"""
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)
                            if fname.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, img_path


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 读取 class_indict
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型
    model = AlexNet(num_outputs=10).to(device)

    # 加载模型权重
    weights_path = "AlexNet_regression_800C.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, weights_only=True))

    model.eval()

    # 加载测试集
    test_root = '../../superalloy_data/heat_exposure_photos/test'  # 测试集文件夹路径
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    error_images = {}  # 用于保存预测错误的图片路径

    # 错误图片存储的文件夹路径
    error_folder = 'error_images'
    # 每次运行前删除 error_images 文件夹及其内容
    if os.path.exists(error_folder):
        shutil.rmtree(error_folder)
    # 创建 error_images 文件夹
    os.makedirs(error_folder)

    # 遍历每个类别
    for cla in os.listdir(test_root):
        cla_path = os.path.join(test_root, cla)
        if os.path.isdir(cla_path):
            dataset = ImageDataset(cla_path, transform=data_transform)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            # 对每个图片进行预测
            with torch.no_grad():
                for img, img_path in dataloader:
                    img = img.to(device)
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()

                    # 收集真实标签和预测标签
                    all_labels.append(cla)
                    all_preds.append(class_indict[str(predict_cla)])

                    # 如果预测正确，增加正确的计数
                    if class_indict[str(predict_cla)] == cla:
                        correct += 1
                    else:
                        # 如果预测正确，增加正确的计数
                        if class_indict[str(predict_cla)] == cla:
                            correct += 1
                        else:
                            # 如果预测错误，按预测值和真实值分类
                            if class_indict[str(predict_cla)] not in error_images:
                                error_images[class_indict[str(predict_cla)]] = []

                            # 给每个错误的图片添加序号
                            error_image_name = f"{class_indict[str(predict_cla)]}_{cla}_{len(error_images[class_indict[str(predict_cla)]]) + 1}.jpg"
                            error_image_path = os.path.join(error_folder, error_image_name)

                            # 将图片复制到错误图片文件夹
                            shutil.copy(img_path[0], error_image_path)

                            # 将错误图片路径记录下来
                            error_images[class_indict[str(predict_cla)]].append(error_image_path)
                    total += 1

    # 输出准确率
    accuracy = correct / total * 100
    print(f'Total images: {total}, Correct predictions: {correct}')
    print(f'Accuracy: {accuracy:.2f}%')

    # # 生成混淆矩阵
    # cm = confusion_matrix(all_labels, all_preds, labels=list(class_indict.values()))
    # cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    #
    # # 可视化混淆矩阵
    # plt.figure(figsize=(10, 7))
    # plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix')
    # plt.colorbar()
    # tick_marks = np.arange(len(class_indict))
    # plt.xticks(tick_marks, class_indict.values(), rotation=45)
    # plt.yticks(tick_marks, class_indict.values())
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    #
    # # 在每个格子中显示数字
    # thresh = 0.5
    #
    # for i in range(len(class_indict)):
    #     for j in range(len(class_indict)):
    #         plt.text(j, i, format(cm[i][j], 'd'),
    #                  ha="center", va="center",
    #                  color="white" if cm_norm[i][j] > thresh else "black")
    #
    # plt.tight_layout()
    # plt.savefig('confusion_matrix_850C.png')
    # plt.show()

    # # 保存混淆矩阵到文件
    # np.savetxt('confusion_matrix_850C.txt', cm, fmt='%d', delimiter='\t')
    # print("Confusion matrix saved to confusion_matrix")

    # 输出预测错误的图片路径
    if error_images:
        print("Predicted error images:")
        for img_path in error_images:
            print(img_path)
    else:
        print("No prediction errors found.")



if __name__ == '__main__':
    main()
