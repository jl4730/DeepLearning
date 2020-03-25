import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torchvision import transforms
import torch
import torchvision.models as models

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to image')
    parser.add_argument('--save_dir', type=str, help='Directory for checkpoints')
    parser.add_argument('--json_dir', type=str, help='Directory for json file')   
    parser.add_argument('--topK', type=int, help='Return top K most likely classes')

    args = parser.parse_args()
    return args

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    image = PIL.Image.open(image)
    image_transform = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor()])
    image_transform = image_transform(image).float()
    
    np_image_transform = np.array(image_transform) 
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image_transform = (np.transpose(np_image_transform, (1, 2, 0)) - mean)/std    
    np_image_transform = np.transpose(np_image_transform, (2, 0, 1))
    
    return np_image_transform

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to("cpu")
    model.eval()
    
    # Load image
    image = process_image(image_path)
    
    # tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()
    
    # use model to predict
    output = model.forward(image)
    prob = torch.exp(output).data
    
    # Find the top topk results
    top_probs = torch.topk(prob, topk)[0].tolist()[0] 
    top_labels = torch.topk(prob, topk)[1].tolist()[0] 
    
    # Convert to class labels
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    
    return top_probs, top_labels
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = eval("models.{}(pretrained=True)".format(checkpoint['architecture']))
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def main():
     
    # Get Args
    args = arg_parser()
    
    # load model
    if type(args.save_dir) == type(None):
        save_dir = 'checkpoint.pth'
    else:
        save_dir = args.save_dir
    
    model_load = load_checkpoint(save_dir)
    print(model_load)
    
    # load image
    if type(args.data_dir) == type(None):
        image_path = 'flowers/train/100/image_07893.jpg'
    else:
        image_path = args.data_dir
        
    # make prediction
    if type(args.topK) == type(None):
        topK = 5
    else:
        topK = args.topK
        
    prob, classes = predict(image_path, model_load, topK)
    
    # load in Json file
    if type(args.json_dir) == type(None):
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    else:
        with open(args.json_dir, 'r') as f:
                cat_to_name = json.load(f)
            
    # convert predicted classes to label names
    labels = []
    for i in classes:
        labels.append(cat_to_name[i])   
        
    # Actual flower label
    flower_num = image_path.split('/')[2]
    actual_label = cat_to_name[flower_num]
    
    print("Actual label is", actual_label)
    print("Top 5 labels are:", labels)
    print("With Probabilty as:", prob)
 
# Run script
if __name__ == '__main__': main()