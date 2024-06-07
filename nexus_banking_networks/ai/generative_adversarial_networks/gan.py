import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, num_features):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, num_features)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = self.fc2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_features):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.001)

    def train(self, real_data, num_epochs):
        for epoch in range(num_epochs):
            # Train discriminator
            self.optimizer_d.zero_grad()
            real_output = self.discriminator(real_data)
            fake_data = self.generator(torch.randn(real_data.shape[0], 100))
            fake_output = self.discriminator(fake_data)
            loss_d = -(torch.mean(real_output) - torch.mean(fake_output))
            loss_d.backward()
            self.optimizer_d.step()

            # Train generator
            self.optimizer_g.zero_grad()
            fake_data = self.generator(torch.randn(real_data.shape[0], 100))
            fake_output = self.discriminator(fake_data)
            loss_g = -torch.mean(fake_output)
            loss_g.backward()
            self.optimizer_g.step()

    def generate_synthetic_data(self, num_samples):
        synthetic_data = self.generator(torch.randn(num_samples, 100))
        return synthetic_data
