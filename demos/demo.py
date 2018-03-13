import torch
# import torchvision.utils as vutils
import numpy as np
# import torchvision.models as models
from c2board import SummaryWriter

# resnet18 = models.resnet18(False)
writer = SummaryWriter(tag='default')

import pdb
pdb.set_trace()

for n_iter in range(100):
    s1 = torch.rand(1) # value to keep
    s2 = torch.rand(1)
    writer.add_scalar('data/scalar1', s1[0], n_iter) # data grouping by `slash`
    # writer.add_scalars('data/scalar_group', {"xsinx":n_iter*np.sin(n_iter),
    #                                          "xcosx":n_iter*np.cos(n_iter),
    #                                          "arctanx": np.arctan(n_iter)}, 
    #                                          n_iter)
    # x = torch.rand(32, 3, 64, 64) # output from network
    if n_iter % 10 == 0:
        # x = vutils.make_grid(x, normalize=True, scale_each=True)   
        writer.add_image('Image', x, n_iter) # Tensor
        writer.add_text('Text', 'text logged at step:'+str(n_iter), n_iter)
        writer.add_text('markdown Text', '''a|b\n-|-\nc|d''', n_iter)
        # for name, param in resnet18.named_parameters():
        #     writer.add_histogram(name, param, n_iter)

# export scalar data to JSON for external processing
writer.close()
