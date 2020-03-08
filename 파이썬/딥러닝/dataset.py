# MNIST DATASET loader
# mini_batch

def MNIST_dataloader(train_batch_size, val_batch_size):
    train_dataset = datasets.MNIST('./MNIST_Dataset', train = True, download = True, transform = transforms.ToTensor())
    validation_dataset = datasets.MNIST("./MNIST_Dataset", train = False, download = True, transform = transforms.ToTensor())
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = val_batch_size, shuffle = True)
    return train_loader, validation_loader


# numpy regression noise dataset generator
def numpy_noise_dataset_generator:
    n_data = 1000
    data_mean, data_std = 2., 1.
    
    ##### Your Code(Dataset Generation/Start) #####
    x_data = np.random.normal(data_mean,data_std,size = n_data)
    y_data = 0.2*np.random.normal(loc = 0, scale = 1., size = n_data) + x_data +1
    ##### Your Code(Dataset Generation/End) #####
    print(x_data.shape)
    print(y_data.shape)
    fig, ax1 = plt.subplots(figsize = (15,5))
    ax1.plot(x_data, y_data, 'bo')
    ax1.grid()
