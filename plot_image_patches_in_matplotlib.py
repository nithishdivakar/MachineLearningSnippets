import matplotlib.pyplot as plt
from tqdm import tqdm

# gray scale images. Add third dim if color.
def Plot(xc,plt_name):
    def merge(images, size):
        h, w = images.shape[2], images.shape[3]
        img = numpy.zeros((h * size[0], w * size[1]))

        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w] = image[0,:,:]

        return img
    
    def plot_data(xx):
        fig = plt.figure()
        
        img = merge(xx,(8,8)) # assuming no of patches are 64 
        plt.imshow(img,cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
        plt.axis('off')
        plt.show()
        fig.savefig(plt_name)
    
    plot_data(xc)
