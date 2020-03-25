import argparse
from collections import OrderedDict
import torchvision.models as models
from torch import nn
from torchvision import datasets, transforms, models
import torch
from torch import optim

# parse keyword arguments from the command line
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, help='Directory for checkpoints')
    parser.add_argument('--arch', type=str, help='Model architecture')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, help='Number of epochs')

    args = parser.parse_args()
    return args

# Load model
def load_model(arch='vgg16', hidden_units=4096):
    # Load the pre-trained model     
    if arch =='vgg16' or type(arch) == type(None):
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    elif arch =='densenet121':
        model = models.densenet121(pretrained=True)
        model.name = "densenet121"
    else:
        raise ValueError('Sorry only support densenet121 and vgg16 for now. Pls change arch to densenet121 or vgg16')
        
    if arch =='vgg16' or type(arch) == type(None): 
        hidden_units = 4096
    elif arch =='densenet121':
        hidden_units = 500
    else:
        raise ValueError('Sorry only support densenet121 and vgg16 for now. Pls change arch to densenet121 or vgg16')
        
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    
    # Number of inputs  
    if arch=='vgg16' or type(arch) == type(None):
        num_inputs = model.classifier[0].in_features
    elif arch=='densenet121':
        num_inputs = model.classifier.in_features
    else:
        raise ValueError('Sorry only support densenet121 and vgg16 for now. Pls change arch to densenet121 or vgg16')
    
    # Define Classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_inputs, hidden_units, bias=True)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)), 
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    return model

def train_model(model, trainloaders, testloaders, device, 
                  criterion, optimizer, epochs, print_every, steps):
    
    running_loss = 0
    model.to(device)
    
    # Train model
    for epoch in range(epochs):
        for inputs, labels in trainloaders:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloaders:  # use validation set to get accuracy
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloaders):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloaders):.3f}")
                running_loss = 0
                model.train()
                
    return model

def main():
     
    # Get Args
    args = arg_parser()
    
    # Load Model
    model = load_model(arch=args.arch, hidden_units=args.hidden_units)
    
    # Load data
    if type(args.data_dir) == type(None): 
        data_dir = 'flowers'
    else:
        data_dir = args.data_dir
        
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()
    
    if type(args.learning_rate) == type(None): 
        lr = 0.003
    else:
        lr = args.learning_rate
    
    optimizer = optim.Adam(model.classifier.parameters(), lr)
        
    if type(args.epochs) == type(None):
        epochs = 10
        
    # Train Model
    trained_model = train_model(model, trainloaders, testloaders, 
                                  device, criterion, optimizer, epochs, 
                                  print_every=32, steps=0)
    
    # Save the checkpoint 
    trained_model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'architecture': trained_model.name,
                  'classifier' : trained_model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': trained_model.state_dict(),
                  'class_to_idx': trained_model.class_to_idx}

    if type(args.save_dir) == type(None):
        save_dir = 'checkpoint.pth'
    else:
        save_dir = args.save_dir
    
    torch.save(checkpoint, save_dir)
  
# Run script
if __name__ == '__main__': main()