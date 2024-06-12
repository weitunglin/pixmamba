from typing import Optional
import torch
import math
import numpy as np
from scipy import ndimage
import time


from mmagic.registry import METRICS
from .base_sample_wise_metric import BaseSampleWiseMetric


@METRICS.register_module()
class UIQM(BaseSampleWiseMetric):
    metric = 'UIQM'

    def __init__(self,
            gt_key: str = 'gt_img',
            pred_key: str = 'pred_img',
            collect_device: str = 'cpu',
            prefix: Optional[str] = None,
            crop_border=0,
            input_order='CHW',
            convert_to=None) -> None:
        self.name = "UIQM"
        super().__init__(
            gt_key=gt_key,
            pred_key=pred_key,
            mask_key=None,
            collect_device=collect_device,
            prefix=prefix)

        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to = convert_to

    def sobel(self, x):
        dx = ndimage.sobel(x,0)
        dy = ndimage.sobel(x,1)
        mag = np.hypot(dx, dy)
        mag *= 255.0 / np.max(mag)
        return mag
    
    def get_values(self,blocks,top,bot,k1,k2,alpha):
        values = alpha * torch.pow((top / bot), alpha) * torch.log(top / bot)
        values[blocks[k1:k1+10, k2:k2+10, :] == 0] = 0.0
        return values
        
    def _uiconm(self, x, window_size):
        k1x = x.shape[1] / window_size
        k2x = x.shape[0] / window_size
        w = -1. / (k1x * k2x)
        blocksize_x = window_size
        blocksize_y = window_size
        blocks = x[0:int(blocksize_y * k2x), 0:int(blocksize_x * k1x)]
        k1, k2 = int(k1x), int(k2x)
        alpha = 1

        val = 0.0
        for l in range(k1):
            for k in range(k2):
                block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
                max_ = np.max(block)
                min_ = np.min(block)
                top = max_-min_
                bot = max_+min_
                if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
                else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
                #try: val += plip_multiplication((top/bot),math.log(top/bot))
        return w*val

    def mu_a(self, x, alpha_l = 0.1, alpha_r = 0.1):
        """
        Calculates the asymetric alpha-trimmed mean
        """
        # sort pixels by intensity - for clipping
        x = sorted(x)
        # get number of pixels
        K = len(x)
        # calculate T alpha L and T alpha R
        T_a_L = math.ceil(alpha_l*K)
        T_a_R = math.floor(alpha_r*K)
        # calculate mu_alpha weight
        weight = (1/(K-T_a_L-T_a_R))
        # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
        s   = int(T_a_L+1)
        e   = int(K-T_a_R)
        val = sum(x[s:e])
        val = weight*val
        return val


    def s_a(self, x, mu):
        val = 0
        for pixel in x:
            val += math.pow((pixel-mu), 2)
        return val/len(x)

    def _uicm(self, x):
        r = x[:,:,0].flatten()
        g = x[:,:,1].flatten()
        b = x[:,:,2].flatten()
        rg = r - g
        yb = ((r + g) / 2) - b
        mu_a_rg = self.mu_a(rg)
        mu_a_yb = self.mu_a(yb)
        s_a_rg = self.s_a(rg, mu_a_rg)
        s_a_yb = self.s_a(yb, mu_a_yb)
        l = math.sqrt((mu_a_rg ** 2) + (mu_a_yb ** 2))
        r = math.sqrt(s_a_rg + s_a_yb)
        return (-0.0268 * l) + (0.1586 * r)
    
    def _eme1(self, x, window_size):
        """
        Enhancement measure estimation
        x.shape[0] = height
        x.shape[1] = width
        """
        # if 4 blocks, then 2x2...etc.
        k1 = x.shape[1]/window_size
        k2 = x.shape[0]/window_size
        # weight
        w = 2./(k1*k2)
        blocksize_x = window_size
        blocksize_y = window_size
        # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
        x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
        val = 0
        k1 = int(k1)
        k2 = int(k2)
        for l in range(k1):
            for k in range(k2):
                block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
                max_ = np.max(block)
                min_ = np.min(block)
                # bound checks, can't do log(0)
                if min_ == 0.0: val += 0
                elif max_ == 0.0: val += 0
                else: val += math.log(max_/min_)
        return w*val

    def _uism(self, x):
        r = x[:,:,0]
        g = x[:,:,1]
        b = x[:,:,2]
        rs = self.sobel(r)
        gs = self.sobel(g)
        bs = self.sobel(b)
        r_edge_map = np.multiply(rs, r)
        g_edge_map = np.multiply(gs, g)
        b_edge_map = np.multiply(bs, b)
        r_eme = self._eme1(r_edge_map, 10)
        g_eme = self._eme1(g_edge_map, 10)
        b_eme = self._eme1(b_edge_map, 10)
        lambda_r = 0.299
        lambda_g = 0.587
        lambda_b = 0.144
        return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)

    def calculate(self, x):
        t0 = time.time()
        x = x.permute(1, 2, 0).numpy().astype(np.float32)

        c1 = 0.0282
        c2 = 0.2953
        c3 = 3.5753
        uicm = self._uicm(x)
        uism = self._uism(x)
        uiconm = self._uiconm(x, 10)
        uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
        t1 = time.time()
        print(t1-t0)
        return uiqm
    
    def process_image(self, gt, pred, mask):
        return self.calculate(pred)