import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 设置训练集、验证集和测试集的比例
    train_rate = 0.8
    val_rate = 0.1
    test_rate = 0.1

    # 指向你解压后的photos文件夹
    data_root = os.path.abspath(os.path.join(os.getcwd(),"../../"))
    origin_photo_path = os.path.join(data_root, "superalloy_data","heat_exposure_photos","800")
    assert os.path.exists(origin_photo_path), "path '{}' does not exist.".format(origin_photo_path)

    photo_class = [cla for cla in os.listdir(origin_photo_path)
                    if os.path.isdir(os.path.join(origin_photo_path, cla))]

    # 建立保存训练集的文件夹
    train_root = os.path.join(origin_photo_path,"..", "train")
    mk_file(train_root)
    for cla in photo_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(origin_photo_path,"..", "val")
    mk_file(val_root)
    for cla in photo_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    # 建立保存测试集的文件夹
    test_root = os.path.join(origin_photo_path,"..", "test")
    mk_file(test_root)
    for cla in photo_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(test_root, cla))

    for cla in photo_class:
        cla_path = os.path.join(origin_photo_path, cla)
        images = os.listdir(cla_path)
        num = len(images)

        # 随机划分数据集索引
        train_index = random.sample(images, k=int(num * train_rate))
        remaining_images = list(set(images) - set(train_index))
        val_index = random.sample(remaining_images, k=int(len(remaining_images) * val_rate / (val_rate + test_rate)))
        test_index = list(set(remaining_images) - set(val_index))

        for index, image in enumerate(images):
            image_path = os.path.join(cla_path, image)

            if image in train_index:
                new_path = os.path.join(train_root, cla)
            elif image in val_index:
                new_path = os.path.join(val_root, cla)
            else:
                new_path = os.path.join(test_root, cla)

            # 复制文件到相应目录
            copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
