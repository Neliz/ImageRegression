import torch
import torchvision
import torchvision.transforms as transforms
from dataset import CustomImageDataset
import pandas as pd 
import time
from tqdm import tqdm
import numpy as np 
import os 
import timm


def train(learning_rate, num_epochs=30,batch_size=128):   

    # Set device
    device = 'mps'    

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
        # transforms.Normalize([0.5,], [0.5,])
        # transforms.Normalize(np.mean([0.485, 0.456, 0.406]), np.mean([0.229, 0.224, 0.225]))
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Load the ImageNet Object Localization Challenge dataset
    # train_dataset = CustomImageDataset(df[['filename','mass']],'.',transform=transform)
    df_train = pd.read_csv('data.csv').head(2000)
    train_dataset = CustomImageDataset(df_train[['filename','mass']],'.',transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                               num_workers=4,multiprocessing_context="forkserver",persistent_workers=True)
    
    df_val = pd.read_csv('data.csv').tail(500)
    val_dataset = CustomImageDataset(df_val[['filename','mass']],'.',transform=transform)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, 
                                               num_workers=4,multiprocessing_context="forkserver",persistent_workers=True)

    # Load the ResNet50 model
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, 1)
    
    # model = timm.create_model('resnet18',pretrained=True)
    # conv_weight = model.conv1.weight
    # model.conv1.in_channels=1
    # model.conv1.weight = torch.nn.Parameter(conv_weight.sum(dim=1,keepdim=True))
    # model.fc = torch.nn.Linear(512, 1)
    # did not converge 
    
    # Load the ResNet50 model
    # model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # model = list(model.children())
    # w = model[0].weight
    # model[0] = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=2, bias=False)
    # model[0].weight = torch.nn.Parameter(torch.mean(w, dim=1, keepdim=True))
    # model = torch.nn.Sequential(*model)
    # model.fc = torch.nn.Linear(512, 1)
    # did not work yet
    
    # model = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
    # model = list(model.children())
    # model[-1] = torch.nn.Linear(512, 1)
    # model = torch.nn.Sequential(*model)
    
    # Parallelize training across multiple GPUs
    model = torch.nn.DataParallel(model)

    # Set the model to run on the device
    model = model.to(device)

    # Define the loss function and optimizer
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss_total = 1e12
    print('Start epochs')
    # Train the model...
    for epoch in range(num_epochs):
        time_start = time.time()
        model.train()
        for inputs, labels in tqdm(train_loader):
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            # Backward pass
            loss.backward()
            optimizer.step()
            
        
        # validate the model
        model.eval()
        val_loss_all = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                #   Move input and label tensors to the device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                val_loss = criterion(outputs, labels.unsqueeze(1))
                val_loss_all.append(val_loss.item())


        val_loss_total = np.mean(val_loss_all)
        # Print the loss for every epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Duration: {time.time()-time_start}, Loss: {loss.item():.4f}, Validation Loss: {val_loss_total:.4f},' )
        
        if val_loss_total < best_val_loss_total:
            best_val_loss_total = val_loss_total

            filename = os.path.join(
                './outputs/model', f'model_{learning_rate}.pt')
            # save model state related to this epoch
            os.makedirs('./outputs/model', exist_ok=True)
            torch.save(model, filename)



    print(f'Finished Training, Best validation Loss: {best_val_loss_total:.4f}')
    
    # torch.save(model,'model.pt')
    return filename



if __name__ == '__main__':
    train(learning_rate = 0.07)


    
    