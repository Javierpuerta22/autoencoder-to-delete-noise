import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#importar dataloader
import torch.utils.data.dataloader as dataloader
import torchvision.datasets as datasets
from model import Autoencoder
from config import *


#Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformed = transforms.Compose([transforms.ToTensor()])

#creamos el set de train y de test
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transformed)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transformed)

def add_gaussian_noise(images, mean=0, std=0.1):
    """
    Añade ruido gaussiano a las imágenes.

    Parameters:
        images (torch.Tensor): Tensor de imágenes.
        mean (float): Media del ruido gaussiano.
        std (float): Desviación estándar del ruido gaussiano.

    Returns:
        torch.Tensor: Tensor de imágenes con ruido gaussiano.
    """
    noise = torch.randn(images.size()) * std + mean
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)  # Asegurarse de que los valores estén en el rango [0, 1]

train_data = dataloader.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) 
test_data = dataloader.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    

model = Autoencoder(input_dim=IMAGE_DIMS[0]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
images = []

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, _) in enumerate(train_data):
        data_noisse = add_gaussian_noise(data)
        data_noisse = data_noisse.to(device)
        data = data.to(device)
        recon = model(data_noisse)
        loss = criterion(recon, data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Epoch {}, Batch idx {}, loss {}'.format(epoch, batch_idx, loss.item()))
            
            with torch.no_grad():
                recon = model(data)
                #guardamos las imagenes
                torchvision.utils.save_image(data_noisse, './images_autoencoder/data_images_{}.png'.format(epoch), nrow=4)
                torchvision.utils.save_image(recon, './images_autoencoder/recon_images_{}.png'.format(epoch), nrow=4)

# test loop
with torch.no_grad():
    for batch_idx, (data, _) in enumerate(test_data):
        data_noisse = add_gaussian_noise(data)
        data_noisse = data_noisse.to(device)
        data = data.to(device)
        recon = model(data_noisse)
        loss = criterion(recon, data)
        
        if batch_idx % 100 == 0:
            print('Batch idx {}, loss {}'.format(batch_idx, loss.item()))
            
            with torch.no_grad():
                recon = model(data)
                #guardamos las imagenes
                torchvision.utils.save_image(data_noisse, './images_autoencoder/data_images_test_{}.png'.format(epoch), nrow=4)
                torchvision.utils.save_image(recon, './images_autoencoder/recon_images_test_{}.png'.format(epoch), nrow=4) 
        