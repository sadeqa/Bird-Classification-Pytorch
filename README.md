
## Object recognition and computer vision 2018/2019

### Assignment 3: Image classification 

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18/assignment3/bird_dataset.zip). The test image labels are not provided.

#### Training and validating your model
Run the script `main_yolo.py` to train your model.

- By default the images are loaded and resized to 299x299 pixels and normalized to zero-mean and standard deviation of 1. See data.py for the `data_transforms`.
- The default arguments of main_yolo.py will git clone and use the yolo_v3 architecture to crop the image datasets and create a new folder and then train the model.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.

