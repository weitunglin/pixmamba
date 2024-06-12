from typing import Optional
import time
import cv2
import numpy as np
import torch
from mmagic.registry import METRICS
from .base_sample_wise_metric import BaseSampleWiseMetric


@METRICS.register_module()
class UCIQE(BaseSampleWiseMetric):
    metric = 'UCIQE'

    def __init__(self,
                 gt_key: str = 'gt_img',
                 pred_key: str = 'pred_img',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(
            gt_key=gt_key,
            pred_key=pred_key,
            collect_device=collect_device,
            prefix=prefix)

    def process_image(self, gt, pred, mask):
        t0 = time.time()
        # UCIQE_values = []
        # for res, gt_img in zip(results, gt):
        img_lab = np.array(cv2.cvtColor(pred.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2LAB),dtype=np.float32)
        # Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576 according to paper.
        coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
        img_lum = img_lab[..., 0]/255
        img_a = img_lab[..., 1]/255
        img_b = img_lab[..., 2]/255

        img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

        img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
        aver_sat = np.mean(img_sat)                                       # Average of saturation

        aver_chr = np.mean(img_chr)                                       # Average of Chroma

        var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

        dtype = img_lum.dtype                                             # Determine the type of img_lum
        if dtype == 'uint8':
            nbins = 256
        else:
            nbins = 65536

        hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
        cdf = np.cumsum(hist)/np.sum(hist)

        ilow = np.where(cdf > 0.0100)
        ihigh = np.where(cdf >= 0.9900)
        tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
        con_lum = tol[1]-tol[0]

        uciqe = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    
        return uciqe
        # UCIQE_values.append(uciqe.cpu().numpy())
        t1 = time.time()
        print(t1-t0)

        return uciqe