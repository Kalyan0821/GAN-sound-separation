import numpy as np
import torch
import librosa

def warpgrid(B, f, T, warp=True):
    x = np.linspace(-1, 1, num=T)
    y = np.linspace(-1, 1, num=f)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((B, f, T, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)

    return grid  # B x f x T x 2


def resample_logscale(raw_mags, opt, f=256):
    """ raw_mags: B x 1 x F x T (F=512, T=256)
        out: B x 1 x f x T """
    B, _, F, T = raw_mags.shape
    grid_warp = torch.from_numpy(warpgrid(B, f, T, warp=True)).to(opt.device)       
    mags = torch.nn.functional.grid_sample(raw_mags, grid_warp, align_corners=False)  # B x 1 x f x T   
    return mags

    


def istft_reconstruction(mag, phase, hop_length=256, length=65535):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length, length=length)
    return np.clip(wav, -1., 1.)
