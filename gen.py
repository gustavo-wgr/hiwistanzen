import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

# If needed, adjust path to where helper_functions is located.
sys.path.append(r"C:\Users\gus07\Desktop\data hiwi")


##############################################################################
# 1. Define the Conditional Generator and Discriminator
##############################################################################

class ConditionalGenerator(nn.Module):
    """
    A generator that takes noise and labels, embeds the labels, and generates data
    conditioned on these labels.
    """
    def __init__(self, noise_dim, label_dim, embed_dim, output_dim):
        super().__init__()
        # Embedding for labels
        self.label_emb = nn.Embedding(num_embeddings=label_dim, embedding_dim=embed_dim)
        
        # The actual generator network
        # We add noise_dim and embed_dim to get the total input dimension
        self.model = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        """
        noise:  (batch_size, noise_dim)
        labels: (batch_size,) integer class labels
        """
        # Embed labels to shape (batch_size, embed_dim)
        label_embeddings = self.label_emb(labels)  # => (batch_size, embed_dim)

        # Concatenate noise and embedded labels
        x = torch.cat([noise, label_embeddings], dim=1)
        return self.model(x)


class ConditionalDiscriminator(nn.Module):
    """
    A discriminator that sees both the input data and the label embedding.
    """
    def __init__(self, input_dim, label_dim, embed_dim):
        super().__init__()
        # Embedding for labels
        self.label_emb = nn.Embedding(num_embeddings=label_dim, embedding_dim=embed_dim)
        
        # The discriminator network
        # input_dim + embed_dim is the concatenation dimension
        self.model = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, data, labels):
        """
        data:   (batch_size, input_dim)
        labels: (batch_size,) integer class labels
        """
        # Embed labels
        label_embeddings = self.label_emb(labels)  # => (batch_size, embed_dim)

        # Concatenate input data with label embeddings
        x = torch.cat([data, label_embeddings], dim=1)
        return self.model(x)

##############################################################################
# 2. Main CGAN training routine
##############################################################################

def generate(
    train_loader,
    noise_dim=100,
    label_dim=10,        # number of classes
    embed_dim=50,        # dimension used to embed labels
    input_dim=2800,
    lr=0.0002,
    num_epochs=50,
    num_samples=500,     # Amount of artificial samples to be generated
    save_path="synth_data/generated_data.pkl", 
    pretrained_generator_path=None,
    save_new_generator_path=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generator training on {len(train_loader.dataset)} samples. Will generate {num_samples} synthetic samples.")

    # 2a. Initialize generator & discriminator
    generator = ConditionalGenerator(
        noise_dim=noise_dim,
        label_dim=label_dim,
        embed_dim=embed_dim,
        output_dim=input_dim
    ).to(device)

    discriminator = ConditionalDiscriminator(
        input_dim=input_dim,
        label_dim=label_dim,
        embed_dim=embed_dim
    ).to(device)

    # 2b. Optionally load a pretrained generator
    if pretrained_generator_path:
        generator.load_state_dict(torch.load(pretrained_generator_path, map_location=device))
        print(f"Loaded pretrained generator from {pretrained_generator_path}")

    # 2c. Always train (fine-tune if a pretrained generator was loaded)
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for batch_idx, (real_inputs, real_labels_onehot) in enumerate(train_loader):
            real_labels = torch.argmax(real_labels_onehot, dim=1).long()
            real_inputs = real_inputs.squeeze(1).to(device)
            real_labels = real_labels.to(device)
            
            batch_size_current = real_inputs.size(0)
            real_targets = torch.ones(batch_size_current, 1, device=device)
            fake_targets = torch.zeros(batch_size_current, 1, device=device)

            ##########################
            # ----- Train D ---------
            ##########################
            discriminator.zero_grad()

            # Forward pass real batch
            out_real = discriminator(real_inputs, real_labels)
            loss_D_real = criterion(out_real, real_targets)

            # Forward pass fake batch
            noise = torch.randn(batch_size_current, noise_dim, device=device)
            # Random integer labels for generation
            random_labels = torch.randint(0, label_dim, (batch_size_current,), device=device)
            fake_inputs = generator(noise, random_labels)

            out_fake = discriminator(fake_inputs.detach(), random_labels)
            loss_D_fake = criterion(out_fake, fake_targets)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()

            ##########################
            # ----- Train G ---------
            ##########################
            generator.zero_grad()
            # We re-run fake data through D, but *do not* detach so we can backprop into G
            out_fake_for_G = discriminator(fake_inputs, random_labels)
            loss_G = criterion(out_fake_for_G, real_targets)
            loss_G.backward()
            optimizer_G.step()

    # Optionally save newly trained (or fine-tuned) generator
    if save_new_generator_path:
        torch.save(generator.state_dict(), save_new_generator_path)
        print(f"Generator saved to {save_new_generator_path}")

    ###############################
    # 2d. Generate synthetic data
    ###############################
    # 1. Compute how many samples per class
    assert num_samples % label_dim == 0, (
        "For an even distribution, num_samples should be divisible by label_dim."
    )
    samples_per_class = num_samples // label_dim

    # 2. Build a label tensor: each class label repeated 'samples_per_class' times
    all_labels = []
    for class_id in range(label_dim):
        all_labels.extend([class_id] * samples_per_class)

    # Convert to a torch tensor on the appropriate device
    all_labels = torch.tensor(all_labels, dtype=torch.long, device=device)

    # 3. Generate matching noise
    noise = torch.randn(num_samples, noise_dim, device=device)

    # (Optional) Shuffle them so data order isn't strictly classwise
    perm = torch.randperm(num_samples, device=device)
    all_labels = all_labels[perm]
    noise = noise[perm]

    # 4. Generate synthetic data
    with torch.no_grad():
        synthetic_inputs = generator(noise, all_labels).cpu()
        synthetic_labels = all_labels.cpu()

    # 5. Save the generated data
    synthetic_data = {
        "inputs": synthetic_inputs.numpy(),
        "labels": synthetic_labels.numpy()
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(synthetic_data, f)

    print(f"Generated data saved to {save_path}")
