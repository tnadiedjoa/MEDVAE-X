import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import laplace, uniform_filter, convolve
from skimage.measure import shannon_entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SOBEL_H  = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_V  = SOBEL_H.T
SOBEL_D1 = np.array([[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]], dtype=np.float32)
SOBEL_D2 = np.array([[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]], dtype=np.float32)
_KERNELS = {"H": SOBEL_H, "V": SOBEL_V, "D1": SOBEL_D1, "D2": SOBEL_D2}

WEIGHTS = np.array([0.25, 0.15, 0.15, 0.30, 0.10, 0.05], dtype=np.float32)

EPS       = 1e-8
GRID_SIZE = 4
MSCN_WIN  = 7


class EngineeredScore:

    def __init__(self, grid_size: int = GRID_SIZE, mscn_window: int = MSCN_WIN, method: str = "weighted"):
        self.G      = grid_size
        self.win    = mscn_window
        self.method = method  # "weighted" ou "pca"
        self._scaler     = None
        self._entropy_max = None
        self._pca        = None
        self._pca_sign   = 1

    def fit(self, imgs: list):
        X_raw = np.stack([self._raw_features(self._to_gray(img)) for img in imgs])
        lo = np.percentile(X_raw, 1, axis=0)
        hi = np.percentile(X_raw, 99, axis=0)
        X_clipped = np.clip(X_raw, lo, hi)
        self._clip_lo = lo
        self._clip_hi = hi
        self._scaler = MinMaxScaler()
        self._scaler.fit(X_clipped)
        if self.method == "pca":
            X_std = StandardScaler().fit_transform(X_clipped)
            self._std_scaler = StandardScaler().fit(X_clipped)
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(self._std_scaler.transform(X_clipped)).squeeze()
            ten_idx = 3
            corr = np.corrcoef(pc1, X_clipped[:, ten_idx])[0, 1]
            self._pca_sign = 1 if corr >= 0 else -1
            self._pca = pca
            pc1_signed = pc1 * self._pca_sign
            self._pca_min = pc1_signed.min()
            self._pca_max = pc1_signed.max()
        return self

    def __call__(self, img) -> float:
        assert self._scaler is not None, "Appeler fit() sur un batch d'images propres avant d'utiliser __call__"
        gray  = self._to_gray(img)
        X_raw = self._raw_features(gray).reshape(1, -1)
        X_clipped = np.clip(X_raw, self._clip_lo, self._clip_hi)
        X_norm = self._scaler.transform(X_clipped)
        if self.method == "pca":
            X_std = self._std_scaler.transform(X_clipped)
            pc1 = self._pca.transform(X_std).squeeze() * self._pca_sign
            return float(np.clip((pc1 - self._pca_min) / (self._pca_max - self._pca_min + EPS), 0, 1))
        return float(np.clip((X_norm * WEIGHTS).sum(), 0, 1))

    def _to_gray(self, img) -> np.ndarray:
        if isinstance(img, (str, Path)):
            return np.asarray(Image.open(img).convert("L"), dtype=np.float32) / 255.0
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        if img.ndim == 3:
            img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        return img

    def _mscn(self, img: np.ndarray) -> np.ndarray:
        img64 = img.astype(np.float64)
        mu    = uniform_filter(img64, size=self.win)
        sigma = np.sqrt(np.maximum(uniform_filter(img64 ** 2, size=self.win) - mu ** 2, 0))
        return ((img64 - mu) / (sigma + 1.0)).astype(np.float32)

    def _patches(self, img: np.ndarray) -> np.ndarray:
        H, W   = img.shape
        ph, pw = H // self.G, W // self.G
        out    = np.zeros((self.G, self.G, ph, pw), dtype=img.dtype)
        for i in range(self.G):
            for j in range(self.G):
                out[i, j] = img[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
        return out

    def _raw_features(self, gray: np.ndarray) -> np.ndarray:
        img     = self._mscn(gray)
        patches = self._patches(img)
        lv, rc, ent, ten = [], [], [], []
        ten_dirs = {k: [] for k in _KERNELS}

        for i in range(self.G):
            for j in range(self.G):
                p = patches[i, j]
                lv.append(float(laplace(p).var()))
                rc.append(float(p.std()))
                p_norm = (p - p.min()) / (p.max() - p.min() + EPS)
                ent.append(float(shannon_entropy(p_norm)))
                from scipy.ndimage import sobel
                ten.append(float(np.sqrt(sobel(p, axis=1)**2 + sobel(p, axis=0)**2).mean()))
                for k, kern in _KERNELS.items():
                    ten_dirs[k].append(float(np.abs(convolve(p, kern)).mean()))

        ten_mean = float(np.mean(ten))
        ten_std  = float(np.std(ten))
        dir_vals = [float(np.mean(v)) for v in ten_dirs.values()]

        entropy_inv        = max(ent) - float(np.mean(ent))
        spatial_homogeneity = float(np.clip(1.0 - ten_std / (ten_mean + EPS), 0, 1))
        directional_balance = float(np.clip(min(dir_vals) / (max(dir_vals) + EPS), 0, 1))

        return np.array([
            float(np.mean(lv)),
            float(np.mean(rc)),
            entropy_inv,
            ten_mean,
            spatial_homogeneity,
            directional_balance,
        ], dtype=np.float32)