import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import *
from mpl_toolkits.mplot3d import Axes3D

# from scipy.fftpack import *

def main():

    # init_param
    default_shift = 10
    default_angle = 45
    default_scale = 0.9

    # load images
    img_dir = 'lena/'
    f = np.asarray(cv2.imread(img_dir+'lena512.png',0),dtype=np.float32)
    g = np.asarray(cv2.imread(img_dir+'lena512.png',0),dtype=np.float32)

    # translation, rotation, scaling
    center = tuple(np.array(g.shape)/2)
    rotMat = cv2.getRotationMatrix2D(center, default_angle, 1.0)
    g = cv2.warpAffine(g, rotMat, g.shape, flags=cv2.INTER_LINEAR)

    # row & col size
    row = f.shape[0]
    col = f.shape[1]

    # fft module
    F = fft2(f)
    G = fft2(g)

    # highpass module
    X1 = np.cos(np.pi*(np.arange(row)/row-0.5))
    X2 = np.cos(np.pi*(np.arange(col)/col-0.5))
    X1 = np.reshape(X1,(row,1))
    X2 = np.reshape(X2,(1,col))
    X1 = np.tile(X1,(1,col))
    X2 = np.tile(X2,(row,1))
    X = X1*X2
    H = (1.0-X)*(2.0-X)

    F = H * F
    G = H * G
    M1 = np.log(np.abs(F))
    M2 = np.log(np.abs(G))

    # Log-Polar module
    N = row/np.log(row)
    M1 = cv2.logPolar(M1, (M1.shape[0]/2, M1.shape[1]/2), N, cv2.WARP_FILL_OUTLIERS)
    M2 = cv2.logPolar(M2, (M2.shape[0]/2, M2.shape[1]/2), N, cv2.WARP_FILL_OUTLIERS)

    # plot figures
    # xx = np.linspace(-row/2, row/2, row)
    # yy = np.linspace(-col/2, col/2, col)
    # XX, YY = np.meshgrid(xx, yy)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(XX, YY, H, cmap='jet', linewidth=0)
    # fig.colorbar(surf)
    # ax.set_title("Highpass Module")
    # plt.show()

    # plt.figure()
    # plt.imshow(np.uint8(f), cmap=plt.get_cmap('gray'))
    # plt.show()

    # plt.figure()
    # plt.imshow(np.uint8(g), cmap=plt.get_cmap('gray'))
    # plt.show()

    plt.figure()
    plt.imshow(np.uint8(M1/np.max(M1)*255), cmap=plt.get_cmap('gray'))
    plt.show()

if __name__ == '__main__':
    main()
