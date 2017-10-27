#coding:utf8

import numpy as np
from skimage import transform
from pylab import plt
from PIL import Image

def show_paf(img,paf,stride = 5,img_size=384):
    """
    @param img: ndarry, HxWx3
    @param paf: ndarry, HxwX3
    """
    paf = transform.rescale(paf,img.shape[0]/paf.shape[0])
    X,Y = np.meshgrid(np.arange(0,img_size),np.arange(0,img_size))
    
    plt.imshow(transform.rescale(img,(1)),alpha=0.5)
    res = plt.quiver( X[::stride,::stride],
                Y[::stride,::stride],
                paf[::stride,::stride,::2].sum(axis=2),
                paf[::stride,::stride,1::2].sum(axis=2),            
                scale=20,
                units='width', headaxislength=0.01,
                    alpha=.8,
                    width=0.001,
                    color='r')
    return  res

def fig2np(fig):
    fig.canvas.draw()
    data = np.fromstring(ccc.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(ccc.canvas.get_width_height()[::-1] + (3,))
    return data

 
def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    h,w, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf.reshape(h,w,4)



def PIL2array(img):
    return numpy.array(img.getdata(),
                    numpy.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


