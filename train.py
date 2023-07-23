import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from datetime import datetime

from module.model import resnet50

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((64, 64)),
            
            transforms.ToTensor()
        ]),
        "test": transforms.Compose([transforms.Resize((64,64)),
                                   transforms.ToTensor(),
                                   ])
    }


    batch_size = 3000
    image_path = os.path.join("/mnt/big_disk/gbw/resnet_ocr_/datasets/handwritten_Chinese_characters")  # 汉字数据集
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    train_num = len(train_dataset)
    
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    val_num = len(validate_dataset)

    data_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in data_list.items())

    # 将字典写入 JSON 文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('/mnt/big_disk/gbw/resnet_ocr_/datasets/handwritten_Chinese_characters/class_indices.json', 'w') as json_file:
        json_file.write(json_str)
  
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet50()

    # model_weight_path = "/mnt/big_disk/gbw/resnet_ocr/save_weight_hasbackgrounp_/198.pth" # 已训练80epoch
    # assert os.path.exists(model_weight_path), "文件 {} 不存在.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # print(f'使用预训练权重{model_weight_path}...')
    
    num_class = 3938
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, num_class)
    net.to(device)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    lr = 0.1
    # 构造优化器
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    

    epochs = 200
    save_path = '/mnt/big_disk/gbw/resnet_ocr_/save_weight/1/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"文件夹 '{save_path}' 创建成功")

    result_file = '/mnt/big_disk/gbw/resnet_ocr_/save_weight/1/result.txt'
    with open(result_file, 'w') as file:
        file.write('Epoch,Train Loss,Train Accuracy,Val Accuracy,current_time\n')

    train_steps = len(train_loader)
    
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            _, predicted = logits.max(1)

            # print(f'labels.size(0):{labels.size(0)},labels.size(-1)：{labels.size(-1)}')
            total += labels.size(0) # labels.size(0) = batch_size
            correct += predicted.eq(labels.to(device)).sum().item()
            # print(f'predicted:{predicted},labels:{labels}')

            train_bar.desc = "Train Epoch [{}/{}], Loss: {:.3f}, Accuracy: {:.2f}%".format(
                epoch + 1, epochs, running_loss / (step + 1), 100.0 * correct / total
            )
        train_acc = correct / total
            
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        total = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                total += val_labels.size(0)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}], Accuracy: {:.2f}%".format(epoch + 1,
                                                           epochs, 100 * acc / total)


        # print(f'total:{total},val_num:{val_num}')
        val_accurate = acc / val_num


        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print("\nEpoch [{}/{}], Loss: {:.3f}, train_Accuracy: {:.2f}%,  Time: {}\n".format(
        #         epoch + 1, epochs, running_loss / train_steps, 100.0 * correct / total, current_time
        #     ))

        with open(result_file, 'a') as file:
            file.write("{}/{},\t{:.6f},\t{:.6f},\t{:.6f},\t{}\n".format(
                epoch + 1, epochs, running_loss / train_steps, 100.0 * train_acc, 100.0 * val_accurate, current_time
            ))

        if epoch % 2 == 0:
            torch.save(net.state_dict(), str(save_path) + str(epoch+1) + '.pth')
        
    print('Finished Training')


if __name__ == '__main__':
    main()
