import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import *
from mpl_toolkits.mplot3d import Axes3D

def fft_coreg(master,slave):

    # hunning
    hy = np.hanning(master.shape[0])
    hx = np.hanning(master.shape[1])
    hw = hy.reshape(hy.shape[0],1) * hx
    master = master * hw
    slave = slave * hw

    # fft2
    master_fd = fft2(master)
    slave_fd = fft2(slave)

    # normalization
    master_nfd = master_fd/np.abs(master_fd)
    slave_nfd = slave_fd/np.abs(slave_fd)

    usfac = 1000
    output, Nc, Nr, peak_map = dftregistration(master_nfd,slave_nfd,usfac)

    nr, nc = slave.shape
    diffphase = output[1]
    row_shift = output[2]
    col_shift = output[3]

    return row_shift[0], col_shift[0], peak_map

def dftregistration(buf1ft,buf2ft,usfac):

    nr,nc = buf2ft.shape
    Nr = ifftshift(np.arange(-np.fix(nr/2),np.ceil(nr/2)))
    Nc = ifftshift(np.arange(-np.fix(nc/2),np.ceil(nc/2)))

    if usfac == 0:
        # Simple computation of error and phase difference without registration
        CCmax = np.sum(buf1ft*np.conjugate(buf2ft))
        row_shift = 0
        col_shift = 0
    elif usfac == 1:
        # Single pixel registration
        CC = ifft2(buf1ft*np.conjugate(buf2ft))
        CCabs = np.abs(CC)
        row_shift, col_shift = np.where(CCabs == np.max(CCabs))
        CCmax = CC[row_shift,col_shift]*nr*nc
        # Now change shifts so that they represent relative shifts and not indices
        row_shift = Nr[row_shift]
        col_shift = Nc[col_shift]
    elif usfac > 1:
        # Start with usfac == 2
        CC = ifft2(FTpad(buf1ft*np.conjugate(buf2ft),(2*nr,2*nc)))
        CCabs = np.abs(CC)
        
        ##
        row_shift, col_shift = np.where(CCabs == np.max(CCabs))
        peak_map = ifftshift(CCabs)
        peak_map = np.roll(peak_map,-row_shift,axis=0)
        peak_map = np.roll(peak_map,-col_shift,axis=1)
        ##
        
        # row_shift, col_shift = row_shift[0], col_shift[0]
        CCmax = CC[row_shift,col_shift]*nr*nc
        # Now change shifts so that they represent relative shifts and not indices
        Nr2 = ifftshift(np.arange(-np.fix(nr),np.ceil(nr)))
        Nc2 = ifftshift(np.arange(-np.fix(nc),np.ceil(nc)))
        row_shift = Nr2[row_shift]/2
        col_shift = Nc2[col_shift]/2

        # If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2:
            ### DFT computation ###
            # Initial shift estimate in upsampled grid
            row_shift = np.round(row_shift*usfac)/usfac
            col_shift = np.round(col_shift*usfac)/usfac
            dftshift = np.fix(np.ceil(usfac*1.5)/2) ## Center of output array at dftshift+1
            # Matrix multiply DFT around the current shift estimate
            CC = np.conjugate(dftups(buf2ft*np.conjugate(buf1ft),np.ceil(usfac*1.5),np.ceil(usfac*1.5),usfac,dftshift-row_shift*usfac,dftshift-col_shift*usfac))
            # Locate maximum and map back to original pixel grid 
            CCabs = np.abs(CC)
            rloc, cloc = np.where(CCabs == np.max(CCabs))
            # rloc, cloc = rloc[0], cloc[0]
            CCmax = CC[rloc,cloc]
            rloc = rloc - dftshift
            cloc = cloc - dftshift
            row_shift = row_shift + rloc/usfac
            col_shift = col_shift + cloc/usfac

        # If its only one row or column the mportift along that dimension has no
        # effect. Set to zero.
        if nr == 1:
            row_shift = 0

        if nc == 1:
            col_shift = 0

    rg00 = np.sum(np.abs(buf1ft)**2)
    rf00 = np.sum(np.abs(buf2ft)**2)
    error = 1.0 - np.abs(CCmax)**2/(rg00*rf00)
    error = np.sqrt(np.abs(error))
    diffphase = np.angle(CCmax)

    output=[error,diffphase,row_shift,col_shift]

    Nc,Nr = np.meshgrid(Nc,Nr)

    return output, Nc, Nr, peak_map

def dftups(in_arr,nor,noc,usfac,roff,coff):

    nr,nc=in_arr.shape

    # Compute kernels and obtain DFT by matrix products
    kernc=np.exp((-1j*2*np.pi/(nc*usfac))*( ifftshift(np.arange(0,nc)[:, np.newaxis]) - np.floor(nc/2) )*( np.arange(0,noc) - coff ))
    kernr=np.exp((-1j*2*np.pi/(nr*usfac))*( np.arange(0,nor)[:, np.newaxis] - roff )*( ifftshift(np.arange(0,nr)) - np.floor(nr/2)  ))

    out=np.dot(np.dot(kernr,in_arr),kernc)
    return out

def FTpad(imFT,outsize):

    Nin = np.array(imFT.shape)
    Nout = np.asarray(outsize)
    imFT = fftshift(imFT)
    center = np.floor(Nin/2)

    imFTout = np.zeros(outsize).astype('complex64')
    centerout = np.floor(Nout/2)

    cenout_cen = (centerout - center).astype(int)

    imFTout[slice(cenout_cen[0],cenout_cen[0]+Nin[0]), slice(cenout_cen[1],cenout_cen[1]+Nin[1])] = imFT
    imFTout = ifftshift(imFTout)*Nout[0]*Nout[1]/(Nin[0]*Nin[1])

    return imFTout

def main():

    ## 
    trans_true = [0, 0]
    angle_true = 1
    scale_true = 1.0
    mag_scale = 100

    ## load images
    img_dir = 'lena/'
    f = np.asarray(cv2.imread(img_dir+'lena512.png',0),dtype=np.float64)
    g = np.asarray(cv2.imread(img_dir+'lena512.png',0),dtype=np.float64)
    row = f.shape[0]; col = f.shape[1] # row & col size
    hrow = int(row/2); hcol = int(col/2)
    center = tuple(np.array(g.shape)/2)

    ## scale, rotate, translate
    g_tmp = cv2.resize(g,None,fx=scale_true,fy=scale_true, interpolation = cv2.INTER_CUBIC)
    g = g*0
    if scale_true < 1.0:
        row_tmp = g_tmp.shape[0]; col_tmp = g_tmp.shape[1] # row & col size
        hrow_tmp = int(row_tmp/2); hcol_tmp = int(col_tmp/2)
        row_slice = slice(int(center[0]-hrow_tmp),int(center[0]+hrow_tmp))
        col_slice = slice(int(center[1]-hcol_tmp),int(center[1]+hcol_tmp))
        g[row_slice,col_slice] = g_tmp
    else:
        center_tmp = tuple(np.array(g_tmp.shape)/2)
        row_slice = slice(int(center_tmp[0]-hrow),int(center_tmp[0]+hrow))
        col_slice = slice(int(center_tmp[1]-hcol),int(center_tmp[1]+hcol))
        g = g_tmp[row_slice,col_slice]

    rotMat = cv2.getRotationMatrix2D(center, angle_true, 1.0)
    g = cv2.warpAffine(g, rotMat, g.shape, flags=cv2.INTER_CUBIC)

    transMat = np.float32([[1,0,trans_true[0]],[0,1,trans_true[1]]])
    g = cv2.warpAffine(g,transMat,(col,row))

    ## hanning window
    hy = np.hanning(row)
    hx = np.hanning(col)
    hw = hy.reshape(row, 1) * hx.reshape(1, col)
    f = f * hw
    g = g * hw

    # fft
    F = fftshift(fft2(f))
    G = fftshift(fft2(g))

    # highpass filter
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

    ## Log-Polar transform
    F = np.abs(F)
    G = np.abs(G)
    FLP = cv2.logPolar(F, (F.shape[0]/2, F.shape[1]/2), mag_scale, cv2.INTER_NEAREST)
    GLP = cv2.logPolar(G, (G.shape[0]/2, G.shape[1]/2), mag_scale, cv2.INTER_NEAREST)

    ## roll and slice
    FLP = np.roll(FLP,int(hcol),axis=1)
    GLP = np.roll(GLP,int(hcol),axis=1)
    FLP = FLP[slice(int(hrow)),slice(int(hcol))]
    GLP = GLP[slice(int(hrow)),slice(int(hcol))]

    ## estimation
    row_shift, col_shift, peak_map = fft_coreg(FLP,GLP)
    angle_est = - row_shift/(hrow) * 180
    scale_est = 1.0 - col_shift/mag_scale

    print(angle_true,scale_true)
    print(angle_est,scale_est)

    ## rescale, rerotate, retranslate
    # g_tmp = cv2.resize(g,None,fx=scale_est,fy=scale_est, interpolation = cv2.INTER_CUBIC)
    # g = g*0
    # if scale_est < 1.0:
    #     row_tmp = g_tmp.shape[0]; col_tmp = g_tmp.shape[1] # row & col size
    #     hrow_tmp = int(row_tmp/2); hcol_tmp = int(col_tmp/2)
    #     row_slice = slice(int(center[0]-hrow_tmp),int(center[0]+hrow_tmp))
    #     col_slice = slice(int(center[1]-hcol_tmp),int(center[1]+hcol_tmp))
    #     g[row_slice,col_slice] = g_tmp
    # else:
    #     center = tuple(np.array(g_tmp.shape)/2)
    #     row_slice = slice(int(center[0]-hrow),int(center[0]+hrow))
    #     col_slice = slice(int(center[1]-hcol),int(center[1]+hcol))
    #     g = g_tmp[row_slice,col_slice]

    rotMat = cv2.getRotationMatrix2D(center, angle_est, 1.0)
    g = cv2.warpAffine(g, rotMat, g.shape, flags=cv2.INTER_CUBIC)
    
    ## plot figures
    xx = np.linspace(-hrow, hrow, row)
    yy = np.linspace(-hcol, hcol, col)
    XX, YY = np.meshgrid(xx, yy)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XX, YY, hw, cmap='jet', linewidth=0)
    fig.colorbar(surf)
    ax.set_title("Hanning Module")

    plt.figure()
    plt.imshow(np.uint8(np.abs(f-g)), cmap=plt.get_cmap('gray'))
    plt.figure()
    plt.imshow(np.uint8(F/np.max(F)*255), cmap=plt.get_cmap('gray'))
    plt.figure()
    plt.imshow(np.uint8(G/np.max(G)*255), cmap=plt.get_cmap('gray'))
    plt.figure()
    plt.imshow(np.uint8(FLP/np.max(FLP)*255), cmap=plt.get_cmap('gray'))
    plt.figure()
    plt.imshow(np.uint8(GLP/np.max(GLP)*255), cmap=plt.get_cmap('gray'))
    plt.show()

if __name__ == '__main__':
    main()
