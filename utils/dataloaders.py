import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import random

def cifar10_c_dataloader(severity, noise_type="gaussian_noise"):
   labels = torch.from_numpy(np.load("../datasets/CIFAR10/CIFAR-10-C/labels.npy"))
   labels = labels.to(torch.long)
   test_images = torch.from_numpy(np.load("../datasets/CIFAR10/CIFAR-10-C/"+noise_type+".npy"))
   images_size = 32  # 32 x 32

   # Filter based on Severity
   labels = labels[10000*(severity-1):(10000*(severity))]
   images = test_images[10000*(severity-1):(10000*(severity))]
    
   # Rearranging tensor for corrupte data to match format of un-corrupt.
   images = images.reshape((images.size()[0], images_size, images_size,3))
   images = np.transpose(images,(0,3,1,2)) # Custom transpose needed for ouput of courrputed and non courrputed to match (found emprically)
   
   images = images.float()/255

   test_set_c = torch.utils.data.TensorDataset(images,labels)
   testloader_c = torch.utils.data.DataLoader(test_set_c, batch_size=64, shuffle=False)

   return testloader_c, images, labels

def mnist_c_dataloader(noise_type="gaussian_noise"):

   path_corrupt_dataset = '../datasets/MNIST/mnist_c/'

   train_images  = torch.from_numpy(np.load(path_corrupt_dataset+noise_type+"/train_images.npy"))
   train_labels  = torch.from_numpy(np.load(path_corrupt_dataset+noise_type+"/train_labels.npy"))
   test_images   = torch.from_numpy(np.load(path_corrupt_dataset+noise_type+"/test_images.npy"))
   test_labels   = torch.from_numpy(np.load(path_corrupt_dataset+noise_type+"/test_labels.npy"))

   # Rearranging tensor for corrupted data to match format of un-corrupt.
   train_images  = train_images.reshape((train_images.size()[0], 1, 28, 28)) 
   train_images  = train_images.float()/255
   test_images   = test_images.reshape((test_images.size()[0], 1, 28, 28)) 
   test_images   = test_images.float()/255
   # print(train_images.type())
   
   # == Load batches
   train_set_c   = torch.utils.data.TensorDataset(train_images,train_labels)
   trainloader_c = torch.utils.data.DataLoader(train_set_c, batch_size=64, shuffle=True)
   test_set_c    = torch.utils.data.TensorDataset(test_images,test_labels)
   testloader_c  = torch.utils.data.DataLoader(test_set_c, batch_size=64, shuffle=True)

   return testloader_c, trainloader_c

def pytorch_dataloader(dataset_name="", dataset_dir="", images_size=32, batch_size=64):

    transform = transforms.Compose([
              transforms.Resize((images_size, images_size)),
              transforms.ToTensor()
              ])

    # Check which dataset to load.
    if dataset_name == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform) 
        test_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform)
    
    elif dataset_name =="CIFAR100": 
        train_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform) 
        test_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform)

    elif dataset_name == "TinyImageNet":
        train_set = torchvision.datasets.ImageFolder(root=dataset_dir+"tiny-imagenet-200/train", transform=transform)
        test_set = torchvision.datasets.ImageFolder(root=dataset_dir+"tiny-imagenet-200/val", transform=transform)

    else:
        print("ERROR: dataset name is not integrated into NETZIP yet.")


    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=True)

    return train_loader, test_loader


def samples_dataloader(N_T, noisy_images, noisy_labels):
    # Get length of images
    max_num_noisy_samples = len(noisy_labels)-1
    samples_indices_array = []
    for _ in range(N_T): 
        # Select a random number from the max number of images
        i = random.randint(0,max_num_noisy_samples)
        samples_indices_array.append(i)
    
    selected_noisy_images = np.take(noisy_images,samples_indices_array, axis=0)
    selected_noisy_labels = np.take(noisy_labels,samples_indices_array, axis=0)
    # print("samples_indices_array = ",samples_indices_array)
    # print("selected_noisy_images = ", np.shape(selected_noisy_images))
    # print("selected_noisy_labels = ", np.shape(selected_noisy_labels))
    # input("press enter")
        
    N_T_test_set_c = torch.utils.data.TensorDataset(selected_noisy_images,selected_noisy_labels)
    N_T_testloader_c = torch.utils.data.DataLoader(N_T_test_set_c, batch_size=64, shuffle=True)

    # print("selected_noisy_images shape = ",selected_noisy_images.shape)
    # print("selected_noisy_images type = ",selected_noisy_images.dtype)
    # print("selected_noisy_labels shape = ",selected_noisy_labels.shape)
    # print("selected_noisy_labels type = ",selected_noisy_labels.dtype)
    

    # input("enter to continue")
    # == TODO: Data Augmentation: To do it we need to implement our custom dataset class like done here: https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
    #   # load noisy dataset
    #   train_transform = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomRotation(),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ])

    return N_T_testloader_c