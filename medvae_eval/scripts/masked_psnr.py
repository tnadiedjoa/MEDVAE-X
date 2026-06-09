import json
import numpy as np
import cv2
import torch
import torch.nn as nn


class MaskedPSNR(nn.Module):

    def __init__(self, ann_path, data_range=1.0):
        super().__init__()
        self.data_range = data_range
        ann = json.load(open(ann_path))
        self._img_info = {im["file_name"]: (im["id"], im["height"], im["width"]) for im in ann["images"]}
        self._polys = {}
        for a in ann["annotations"]:
            self._polys.setdefault(a["image_id"], []).append(a["segmentation"])
        self._cache = {}

    def _mask(self, file_name):
        if file_name not in self._cache:
            img_id, h, w = self._img_info[file_name]
            mask = np.zeros((h, w), dtype=np.uint8)
            for segs in self._polys.get(img_id, []):
                for poly in segs:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
            self._cache[file_name] = mask
        return self._cache[file_name]

    def set_image(self, file_name):
        self._current = file_name
        return self

    def forward(self, decoded_img, img):
        H, W = img.shape[-2], img.shape[-1]
        m = cv2.resize(self._mask(self._current), (W, H), interpolation=cv2.INTER_NEAREST)
        m = torch.as_tensor(m, dtype=torch.bool, device=img.device)
        mse = ((decoded_img[..., m] - img[..., m]) ** 2).mean()
        if mse == 0:
            return torch.tensor(float("inf"))
        return 10 * torch.log10(self.data_range ** 2 / mse)