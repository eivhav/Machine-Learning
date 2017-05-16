
from matplotlib import pyplot as plt



def plot_Image(image):
    fig, ax = plt.subplots(ncols=1, nrows= 1, figsize = (20,20))
    ax.imshow(image)
    plt.show()

