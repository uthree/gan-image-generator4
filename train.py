from model import GAN
from dataset import ImageDataset

ds = ImageDataset(["/mnt/d/local-develop/lineart2image_data_generator/colorized_1024x"], max_len=100)
model = GAN()
model.train(ds, num_epoch=1)
