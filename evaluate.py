import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch

from model import Net

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset_cropped', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--proba', type=str, default='experiment/proba.csv', metavar='P',
                    help='name of the proba csv file')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

state_dict = torch.load(args.model)
model = torch.nn.DataParallel(Net())

# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k.replace('fc.2','fc.3') # remove `module.`
#     new_state_dict[name] = v
# # load params
# model.load_state_dict(new_state_dict)


model.load_state_dict(state_dict)


model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

from data import data_transforms_test

test_dir = args.data + '/test_images/mistery_category'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
proba_file = open(args.proba, "w")
output_file.write("Id,Category\n")
proba_file.write("Id,Proba1,Proba2,Proba3,Proba4,Proba5,Proba6,Proba7,Proba8,Proba9,Proba10,Proba11,Proba12,Proba13,Proba14,Proba15,Proba16,Proba17,Proba18,Proba19,Proba20\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms_test(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))
        proba_file.write("%s,%s\n" % (f[:-4], ', '.join([str(i) for i in output.data.cpu().numpy()[0]]))) 
        
output_file.close()
proba_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
        


