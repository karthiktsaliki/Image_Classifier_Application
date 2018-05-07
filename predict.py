# Usage python predict.py --json_path cat_to_name.json --use_gpu False --topk 5 --image_path C:/Users/TsalikiK/Downloads/Kantar/Kantar_Python_Work/Notebooks/aipnd-project/test/1/image_06743.jpg

# Imports here
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import argparse

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # load the image
    img_pil = Image.open(image)

    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    
    return img_tensor

def load_json(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def predict(image_path, model, topk=5,use_gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    # TODO: Implement the code to predict the class from an image file
    img_tensor=process_image(image_path)
    img_tensor.requires_grad_(False)
    arr=np.array(img_tensor)
    img_tensor=Variable(torch.from_numpy(arr.reshape(1,3,224,224)))
    if use_gpu:
        img_tensor = Variable(img_tensor.cuda())
    else:
        img_tensor = Variable(img_tensor)
    output=model(img_tensor)
    probs= F.softmax(output.data)
    return probs.topk(topk)

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_path', type=str, default='cat_to_name.json', 
                        help='Json path of labels to categories')
    parser.add_argument('--use_gpu', type=bool, default=False, 
                        help='Run with GPU')
    parser.add_argument('--topk', type=int, default=5, 
                        help='Top N classes that needed to be displayed')
    parser.add_argument('--image_path', type=str, default='C:/Users/TsalikiK/Downloads/Kantar/Kantar_Python_Work/Notebooks/aipnd-project/test/1/image_06743.jpg', 
                        help='Image which you want to predict')

    # returns parsed argument collection
    return parser.parse_args()

# The function reads in an image and a checkpoint then prints the most likely image class and it's associated probability
# Also allows users to print out the top K classes along with associated probabilities
# Also allows users to use the GPU to calculate the predictions
# Allows users to load a JSON file that maps the class values to other category names

def main():
    in_arg = get_input_args()
    json_path=in_arg.json_path
    cat_to_name=load_json(json_path)
    use_gpu=in_arg.use_gpu
    topk=in_arg.topk
    image_path=in_arg.image_path
    model_ft_load=torch.load('flowers_transfer.pt')
    values=predict(image_path, model_ft_load, topk=topk,use_gpu=use_gpu)
    x_labels = [cat_to_name[str(ind+1)] for ind in np.array(values[1][0])]
    probs=np.array(values[0][0])
    print('Categories Predicted:',x_labels)
    print('Probabilities in respective manner:',probs)

if __name__=='__main__':
    main()
        
