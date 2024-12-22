from model import *


latent_size = 32
IMAGE_SIZE = 512
device = torch.device("cpu")
G = GeneratorNetwork(latent_size, IMAGE_SIZE)
G.load_state_dict(torch.load("./out/final_G.pth"))
plt.imshow(sample_image(G, latent_size, device))
plt.axis("off")
plt.show()
