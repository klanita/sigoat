import numpy as np
from scipy.sparse import load_npz

from sklearn import linear_model
from scipy.sparse.linalg import lsqr
# import pylops
from ReconstructionBP import backProject

import matplotlib.pyplot as plt
import time

class ReconstructionBenchmarks:
    def __init__(self, geometry='multi'):
        assert (geometry in ['multi', 'linear'])

        self.geometry = geometry
        if geometry == 'multi':
            self.mm = load_npz('./data/ModelMatrixMultiCropped.npz')
        else:
            self.mm = load_npz('./data/ModelMatrixLinearCropped.npz')

        # normalize model matrix (mm) for numerical stability
        self.norm_mm = np.max(abs(self.mm))
        self.mm /= self.norm_mm

    def plot(
        self,
        img,
        cmap='gray',
        figsize=(5, 5),
        filename=None
        ):

        plt.figure(figsize=figsize)
        plt.imshow(img, cmap=cmap)
        plt.colorbar()

        if not (filename is None):
            plt.savefig(filename, format='png', dpi=300)

    def reconstruction_Atb(
        self,
        signal, 
        show=False,
        return_img=True
        ):
        
        signal_vec = np.transpose(signal).reshape(-1, 1)

        start = time.time()
        img_pred =\
            (np.transpose(self.mm)*signal_vec).reshape(256, 256)
        end = time.time()
        # print(f"Atb: {end - start:.2f} sec")

        if show:
            self.plot(img_pred)
        
        if return_img:
            return img_pred

    def reconstruction_lsqr(
        self,
        signal, 
        iter_lim=50,
        show=False,
        return_img=True
        ):
        
        signal_vec = np.transpose(signal).reshape(-1, 1)

        start = time.time()
        img_pred =\
            lsqr(self.mm, signal_vec, iter_lim=iter_lim)[0].reshape(256, 256)
        end = time.time()
        # print(f"lsqr: {end - start:.2f} sec")

        if show:
            self.plot(img_pred)

        if return_img:
            return img_pred

    def reconstruction_linreg(
        self,
        signal,
        model,
        alpha=1e-5,
        max_iter=50,
        show=False,
        return_img=True        
        ):

        signal_vec = np.transpose(signal).reshape(-1, 1)

        start = time.time()
        if model == 'Lasso':
            clf = linear_model.Lasso(
                alpha=alpha, fit_intercept=False, max_iter=max_iter)
        elif model == 'Ridge':
            clf = linear_model.Ridge(
                alpha=alpha, fit_intercept=False, max_iter=max_iter)
        elif model == 'ElasticNet':
            clf = linear_model.ElasticNet(
                alpha=alpha, fit_intercept=False, max_iter=max_iter)
        else:
            raise NotImplementedError

        clf.fit(self.mm, signal_vec)
        img_pred = clf.coef_.reshape(256, 256)
        end = time.time()
        # print(f"{model}: {end - start:.2f} sec")

        if show:
            self.plot(img_pred)

        if return_img:
            return img_pred

    def reconstruction_TV(
        self,
        signal,
        alpha=1e-4,
        iter_lim=50,
        show=False,
        return_img=True
    ):
        signal_vec = np.transpose(signal).reshape(-1, 1)
        nx = self.mm.shape[1]
        
        start = time.time()
        D2op = pylops.SecondDerivative(nx, edge=True)
        img_pred =\
            pylops.optimization.leastsquares.RegularizedInversion(
                self.mm, [D2op], signal_vec.reshape(-1),
                epsRs=[np.sqrt(alpha/2)],
                **dict(iter_lim=iter_lim)).reshape(256, 256)
        end = time.time()
        # print(f"TV: {end - start:.2f} sec")

        if show:
            self.plot(img_pred)

        if return_img:
            return img_pred

    def reconstruction_BP(
        self,
        signal,
        alpha=1e-4,
        show=False,
        return_img=True
    ):  
        n_trans = signal.shape[1]

        if signal.shape[0] < 1000:
            idx_use=list(range(524, 1480))
            signalNotCropped = np.zeros([2030, n_trans])
            signalNotCropped[idx_use] = signal
        else:
            signalNotCropped = signal
        
        if self.geometry == 'multi':
            idx = None
        else:
            idx=range(64, 64+128)

        start = time.time()
        img_pred = backProject(
            signalNotCropped,
            geometry='multisegmentCup',
            resolutionXY=256,
            array_dir='./arrayInfo/',
            idx=idx
            )
        
        end = time.time()
        # print(f"BP: {end - start:.2f} sec")

        if show:
            self.plot(img_pred)

        if return_img:
            return img_pred





def show_resonstruction(
    reconstrutctions,
    cmap='gray',
    fname=None,
    figsize=(7, 3)):
    
    imgs_list = list(reconstrutctions.values())
    titels = list(reconstrutctions.keys())
    
    if len(titels) == 1:
        fig = plt.figure(figsize=figsize)
        img = reconstrutctions[titels[0]]
        if img.shape[0] == 1:
            plt.imshow(reconstrutctions[titels[0]][0], cmap=cmap)
        else:
            plt.imshow(reconstrutctions[titels[0]], cmap=cmap)
        # , cmap='seismic'
        plt.title(titels[0], fontsize=5, fontweight='bold')

        # ax[i].xaxis.set_tick_params(labelsize=3)
        # ax[i].yaxis.set_tick_params(labelsize=3)
    
    elif len(imgs_list) < 4:
        fig, ax = plt.subplots(1, len(imgs_list), figsize=figsize)
        for i in range(len(imgs_list)):
            img = reconstrutctions[titels[i]]
        #     print(img)
        # quit()
        #     # if img.shape[0] == 1:
        #     #     img = img[0]
            ax[i].imshow(img, cmap=cmap)
            # , cmap='seismic'
            ax[i].set_title(titels[i], fontsize=5, fontweight='bold')
            ax[i].xaxis.set_tick_params(labelsize=3)
            ax[i].yaxis.set_tick_params(labelsize=3)
            
    elif len(titels) > 1: 
        n1 = int(np.ceil(len(imgs_list)/2))
        n2 = 2
        figsize = (n2*2.25, n1*2.25)
        
        fig, ax = plt.subplots(n1, n2, figsize=figsize)
        l = 0
        for i in range(n1):
            for j in range(n2):
                if l < len(imgs_list):
                    ax[i, j].imshow(imgs_list[l], cmap=cmap)
                    # , cmap='seismic'
                    ax[i, j].set_title(titels[l], fontsize=5, fontweight='bold')
                    ax[i, j].xaxis.set_tick_params(labelsize=3)
                    ax[i, j].yaxis.set_tick_params(labelsize=3)
                else:
                    ax[i, j].axis('off')
                    ax[i, j].axis('equal')     

                l += 1
                
    elif len(titels) == 1: 
        n1 = int(np.ceil(len(imgs_list)/2))
        figsize = (n1*2.25, 2.25)
        
        fig, ax = plt.subplots(1, n1, figsize=figsize)
        l = 0
        for i in range(n1):
            if l < len(imgs_list):
                ax[i].imshow(imgs_list[l], cmap=cmap)
                # , cmap='seismic'
                ax[i].set_title(titels[l], fontsize=5, fontweight='bold')
                ax[i].xaxis.set_tick_params(labelsize=3)
                ax[i].yaxis.set_tick_params(labelsize=3)
            else:
                ax[i].axis('off')
                ax[i].axis('equal')     

                l += 1
    
    plt.tight_layout()
    if fname:
        plt.savefig(f'{fname}.png', format='png', dpi=300)
