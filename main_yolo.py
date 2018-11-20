from __future__ import division
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import warnings
warnings.filterwarnings("ignore")

os.system('git clone https://github.com/eriklindernoren/PyTorch-YOLOv3')
print(os.getcwd())
if not os.path.isfile('yolov3.weights') : 
    os.system('wget https://pjreddie.com/media/files/yolov3.weights')


import os
import sys
import time
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator




if __name__ == '__main__': 
    # Training settings
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='./bird_dataset', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batchsize', type=int, default=64, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    
    parser.add_argument('--image_folder', type=str, default='./bird_dataset', help='path to dataset')
    parser.add_argument('--config_path', type=str, default='./PyTorch-YOLOv3/config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default='yolov3.weights', help='path to weights file')
    parser.add_argument('--class_path', type=str, default='./PyTorch-YOLOv3/data/coco.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size_cropped', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--output', type=str, default='./birds_output_cropped', help='path to dataset')
    parser.add_argument('--crop', type=bool, default=True, help='whether to crop data')
    parser.add_argument('--use_crop', type=bool, default=True, help='whether to use cropped data')
    parser.add_argument('--dont_crop', type=bool, default=False, help='whether to use cropped data')
    parser.add_argument('--dont_use_crop', type=bool, default=False, help='whether to use cropped data')
    args = parser.parse_args()
    
    from models import *
    from utils.utils import *
    from utils.datasets import *
    cuda = torch.cuda.is_available() 
    
    
    if args.crop and not args.dont_crop : 
        os.makedirs(args.output, exist_ok=True)

        # Set up model
        model = Darknet(args.config_path, img_size=args.img_size)
        model.load_weights(args.weights_path)

        if cuda:
            model.cuda()

        model.eval() # Set in evaluation mode

        direct=args.image_folder
        for data_type in list(os.walk(direct))[0][1] :
            data_type = '/'+ data_type 
            directory = args.image_folder+data_type
            print(directory)
            for folder in list(os.walk(directory))[0][1]:
                print("\nFolder : "+folder)

                if not os.path.isdir(args.output):
                    os.makedirs(args.output)
                if not os.path.isdir(args.output+data_type+'/'+folder):
                    os.makedirs(args.output+data_type+'/'+folder)


                dataloader = DataLoader(ImageFolder(directory+'/'+folder, img_size=args.img_size),
                                        batch_size=args.batch_size_cropped, shuffle=False, num_workers=args.n_cpu)

                classes = load_classes(args.class_path) # Extracts class labels from file

                Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

                imgs = []           # Stores image paths
                img_detections = [] # Stores detections for each image index

                print ('\n\tPerforming object detection:')
            #     prev_time = time.time()
                try:
                    list(dataloader)[0]
                except:
                    for file in os.listdir(directory+'/'+folder):
                        i=plt.imread(directory+'/'+folder+'/'+file)
                        if len(i.shape)==2:
                            plt.imsave(directory+'/'+folder+'/'+file,np.stack((i,)*3, axis=-1))
                        del i
                for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
                    # Configure input
                    input_imgs = Variable(input_imgs.type(Tensor))

                    # Get detections
                    with torch.no_grad():
                        detections = model(input_imgs)
                        detections = non_max_suppression(detections, 80, args.conf_thres, args.nms_thres)


                    # Log progress
            #         current_time = time.time()
            #         inference_time = datetime.timedelta(seconds=current_time - prev_time)
            #         prev_time = current_time
            #         #print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

                    # Save image and detections
                    imgs.extend(img_paths)
                    img_detections.extend(detections)

                print ('\n\tSaving images:')

                # Iterate through images and save plot of detections
                for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

                    # Create plot
                    img = np.array(Image.open(path))
                    path=path.split("/")[-1]
                    plt.imsave(args.output+data_type+'/'+folder+'/'+path, img)

                    # The amount of padding that was added
                    pad_x = max(img.shape[0] - img.shape[1], 0) * (args.img_size / max(img.shape))
                    pad_y = max(img.shape[1] - img.shape[0], 0) * (args.img_size / max(img.shape))
                    # Image height and width after padding is removed
                    unpad_h = args.img_size - pad_y
                    unpad_w = args.img_size - pad_x

                    # Draw bounding boxes and labels of detections
                    if detections is not None:
                        unique_labels = detections[:, -1].cpu().unique()
                        n_cls_preds = len(unique_labels)
                        #bbox_colors = random.sample(colors, n_cls_preds)
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                            if cls_pred == classes.index("bird"):

                                # Rescale coordinates to original dimensions
                                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                                x1, y1 = np.maximum(0,int(x1)-20), np.maximum(0,int(y1)-20)
                                x2, y2 = np.minimum(x1+box_w+40,img.shape[1]), np.minimum(y1+box_h+40,img.shape[0])

                                img1 = img[int(y1):int(y2), int(x1):int(x2), :].copy()

                                path=path.split("/")[-1]
                    # Save generated image with detections
                                if data_type == 'test_images' : 
                                    plt.imsave(args.output+data_type+'/'+folder+'/'+path, img1)
                                    plt.close()
                                else : 
                                    plt.imsave(args.output+data_type+'/'+folder+'/'+path[:-4]+'_cropped_20_.jpg', img1)
                                    plt.close()

                        
                        
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    # Data initialization and loading
    from data import data_transforms
    from data import data_transforms_test

    if args.use_crop and  not args.dont_use_crop : 
        train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.output + '/train_images',
                             transform=data_transforms),
        batch_size=args.batchsize, shuffle=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.output + '/val_images',
                             transform=data_transforms_test),
        batch_size=args.batchsize, shuffle=False, num_workers=1)
        
    else :
        train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images',
                             transform=data_transforms),
        batch_size=args.batchsize, shuffle=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms_test),
        batch_size=args.batchsize, shuffle=False, num_workers=1)

    # Neural network and optimizer
    # We define neural net in model.py so that it can be reused by the evaluate.py script
    from model import Net
    model = nn.DataParallel(Net())
#     print(model.module._modules.keys())
    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')

    #optimizer = optim.SGD(list(model.module.res.fc.parameters()) + list(model.module.res.layer4.parameters())  +list(model.module.res2.fc.parameters()) + list(model.module.res2.layer4.parameters())  #+ list(model.module.res2.layer4.parameters()) 
                          #, lr=args.lr, momentum=args.momentum )
    optimizer = optim.SGD(model.parameters() , lr=args.lr, momentum=args.momentum )
#     scheduler = MultiStepLR(optimizer, milestones=[25,40,65], gamma=0.1)



    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
            if isinstance(output, tuple):
                loss = sum((criterion(o,target) for o in output))
            else:
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))

    def validation():
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))


    for epoch in range(1, args.epochs + 1):
#         scheduler.step()
        train(epoch)
        validation()
        model_file = args.experiment + '/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')
