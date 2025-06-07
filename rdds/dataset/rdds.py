import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import datasets, transforms
import numpy as np
import warnings
import json
from clip import clip

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

BASE = "."
DATASET_ROOT_DIR = os.path.join(BASE, "data")

mapping = {"lie": 0, "mesh": 1, "face": 2, "repair": 3, "Transformation": 4}

class CustomImageDataset(Dataset):
    def __init__(self, split, transform=None):
        self.data_dir = os.path.join(DATASET_ROOT_DIR, split)
        self.image_paths = []
        self.json_paths = []

        # 遍历每个种类文件夹
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                image_folder = os.path.join(class_path, "images")
                json_folder = os.path.join(class_path, "labels")

                # 收集图像和对应的 JSON 文件路径
                for fname in os.listdir(image_folder):
                    if fname.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(image_folder, fname)
                        base_name = os.path.splitext(fname)[0]
                        json_path = os.path.join(json_folder, f"{base_name}.json")

                        if os.path.exists(json_path):
                            self.image_paths.append(img_path)
                            self.json_paths.append(json_path)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        json_path = self.json_paths[idx]

        # 加载图像
        image = Image.open(img_path).convert("RGB")

        # 加载 JSON 文件
        with open(json_path, 'r') as f:
            try:
                label_data = json.load(f)
            except Exception:
                print(json_path)
                
        # 提取标签信息
        type_info = label_data["type"]
        yanzhong_info = label_data["yanzhong"]

        # 处理标签数据
        if isinstance(type_info, str):
            type_list = [type_info]
            yanzhong_list = [yanzhong_info]
        else:
            type_list = type_info
            yanzhong_list = yanzhong_info

        # 初始化6维的独热编码向量
        type_onehot = torch.zeros(5, dtype=torch.float)

        # 根据存在的类别设置对应位置为1
        for t in type_list:
            if t in mapping:
                type_onehot[mapping[t]] = 1

        # 从yanzhong_list中提取最大值
        max_yanzhong = max(yanzhong_list)
        yanzhong_tensor = torch.tensor(max_yanzhong, dtype=torch.long)

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        label_dict = {
            "type_label": type_onehot,    # 6维独热编码向量
            "yanzhong_label": yanzhong_tensor  # 严重程度最大值
        }

        # 构建返回字典
        ret_dict = {
            "image": image,
            "label": label_dict # 严重程度最大值
        }

        return ret_dict


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def getData(args):
    if args.is_pretrain:
        _, preprocess = clip.load("ViT-B/32")
        transform_train = preprocess
        transform_test = preprocess 
    else:
        transform_train = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
            transforms.RandomRotation(degrees=15),   # 随机旋转
            transforms.ColorJitter(                 # 随机颜色变换
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(                # 随机仿射变换
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            transforms.RandomPerspective(           # 随机透视变换
                distortion_scale=0.1,
                p=0.2
            ),
            transforms.ToTensor(),                  # 归一化到 [0,1]
            transforms.Normalize(                   # 标准化
                mean=[0.5296, 0.5323, 0.5292],
                std=[0.2044, 0.2043, 0.2183]
            )
        ])

        # 测试集的变换：仅包含基本的预处理
        transform_test = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),                  # 归一化到 [0,1]
            transforms.Normalize(                   # 标准化
                mean=[0.5226, 0.5311, 0.5317],
                std=[0.2126, 0.2150, 0.2329]
            )
        ])

    datasets = {}
    dataloaders = {}

    train_dataset = CustomImageDataset("train", transform=transform_train)
    test_dataset = CustomImageDataset("test", transform=transform_test)

    datasets["train"] = train_dataset
    datasets["test"] = test_dataset

    for split in ["train", "test"]:
        if split == "train":
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
        )

    return datasets, dataloaders


# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import os
# from torchvision import datasets, transforms
# import numpy as np
# import warnings
# import json

# warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

# BASE = "."
# DATASET_ROOT_DIR = os.path.join(BASE, "data")

# mapping = {"normal": 0, "lie": 1, "mesh": 2, "face": 3, "repair": 4, "Transformation": 5}

# class CustomImageDataset(Dataset):
#     def __init__(self, split, transform=None):
#         self.data_dir = os.path.join(DATASET_ROOT_DIR, split)
#         self.image_paths = []

#         # 遍历每个种类文件夹
#         for class_dir in os.listdir(self.data_dir):
#             class_path = os.path.join(self.data_dir, class_dir)
#             if os.path.isdir(class_path):
#                 image_folder = os.path.join(class_path, "images")

#                 # 收集图像路径
#                 for fname in os.listdir(image_folder):
#                     if fname.endswith(('.jpg', '.png', '.jpeg')):
#                         img_path = os.path.join(image_folder, fname)
#                         self.image_paths.append(img_path)

#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]

#         # 加载图像
#         image = Image.open(img_path).convert("RGB")

#         # 应用图像变换
#         if self.transform:
#             image = self.transform(image)

#         return image


# def calculate_mean_std(data_dir):
#     # 定义一个简单的变换，仅将图像转换为张量
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()  # 归一化到 [0,1]
#     ])

#     dataset = CustomImageDataset(data_dir, transform=transform)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

#     # 初始化变量以累积像素值
#     channels_sum, channels_squared_sum, num_batches = 0, 0, 0

#     for data in dataloader:
#         # 计算每个通道的像素值总和
#         channels_sum += torch.mean(data, dim=[0, 2, 3])
#         # 计算每个通道的像素值平方总和
#         channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#         num_batches += 1

#     # 计算均值
#     mean = channels_sum / num_batches

#     # 计算标准差
#     std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

#     return mean, std


# # 计算训练集和测试集的均值和标准差
# train_mean, train_std = calculate_mean_std("train")
# test_mean, test_std = calculate_mean_std("test")

# print(f"Train Mean: {train_mean}, Train Std: {train_std}")
# print(f"Test Mean: {test_mean}, Test Std: {test_std}")