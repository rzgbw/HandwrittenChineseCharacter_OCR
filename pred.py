import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from tqdm import tqdm
import json
from module.model import ResNet
import torch
from PIL import Image
import matplotlib.pyplot as plt
from module.model import resnet50
import cv2


def main():

    result_dict = {}  # 用于存储模型预测结果和真实图片类别的对应关系


    # Set device to cuda:1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    # 17W
    predict_image_path = '/mnt/big_disk/gbw/strokeExtractionDatasets_raw/handwritten_chinese_stroke_2021/train2021'
    print(predict_image_path)

    data_transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor()])

    # Define a custom dataset class to include image names
    class CustomImageDataset(Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform
            self.images = sorted(os.listdir(self.root))

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image_name = self.images[idx]
            image_path = os.path.join(self.root, image_name)
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, image_name
    predict_dataset = CustomImageDataset(root=predict_image_path, transform=data_transform)
    pred_num = len(predict_dataset)
    print(pred_num)

    batch_size = 4096
    predict_loader = torch.utils.data.DataLoader(predict_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=32)

    # create model
    model = resnet50(num_classes=3938)

    # Move the model to cuda:1
    model = model.to(device)

    # load model weights
    weights_path = "/mnt/big_disk/gbw/resnet_ocr/save_weight/10/137.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    # read class_indict
    json_path = '/mnt/big_disk/gbw/resnet_ocr/datasets_all/7_13/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    json_file = open(json_path, "r", encoding='UTF-8')
    class_indict = json.load(json_file)
    result_path = '/mnt/big_disk/gbw/resnet_ocr_3/predict/all/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"文件夹 '{result_path}' 创建成功")

    model.eval()
    
    with torch.no_grad():
        pred_bar = tqdm(predict_loader)
        image_names_list = []
        for pred_data in pred_bar:
            img, image_names = pred_data
            
            outputs = model(img.to(device))
            predict_y = torch.max(outputs, dim=1)[1] 

            for i in range(len(predict_y)):
                result_dict[image_names[i]] = class_indict[str(predict_y[i].item())]

                output_path = result_path + class_indict[str(predict_y[i].item())]
                
                img_ = img[i].clone().to(device)  # Move input tensor to cuda:1
                img_ = img_.permute(1, 2, 0)  # 调整维度顺序为 (H, W, C)
                img_ = (img_ * 255).clamp(0, 255).byte()  # 将图像像素值恢复到 0-255 范围内
                img_ = Image.fromarray(img_.cpu().numpy())  # 将Tensor转换为PIL图像对象
                
                if os.path.isdir(output_path):
                    # print('文件夹已存在')
                    k_path = os.path.join(output_path + '/' + image_names[i])  # Use original image name
                    img_.save(k_path)
                else:
                    os.makedirs(output_path)
                    k_path = os.path.join(output_path + '/' + image_names[i])  # Use original image name
                    img_.save(k_path)

    # 保存结果到txt文件
    with open("result.txt", "a") as f:
        for key, value in result_dict.items():
            f.write(f"{key}: {value}\n")
        
    print('Finished pred')

if __name__ == '__main__':
    main()            