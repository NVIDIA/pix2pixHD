from curses import use_default_colors
import fractions
import torch
from torchvision import utils as vutils
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader

def fc(input_tensor, filename):
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    # filename = "./test_dataset/" + filename
    vutils.save_image(input_tensor, filename)

opt = TrainOptions().parse()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
# print('#training images = %d' % dataset_size)

for i, data in enumerate(dataset):
    print("/n--------------------------------------/n")
    fc(data['s_inst'], "%d_s_inst.png"%i)
    fc(data['s_image'], "%d_s_image.png"%i)
    fc(data['t_inst'], "%d_t_inst.png"%i)
    fc(data['t_image'], "%d_t_image.png"%i)
    if i == 2:
        break