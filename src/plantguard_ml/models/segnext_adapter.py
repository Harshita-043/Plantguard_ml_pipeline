from __future__ import annotations
from pathlib import Path
import numpy as np

class SegNeXtAdapter:
    def __init__(self, repo_dir, config, checkpoint):
        self.repo_dir=Path(repo_dir); self.config=config; self.checkpoint=Path(checkpoint)
        self.model=None
        self._init()
    def _init(self):
        try:
            from mmseg.apis import init_model
            self.model=init_model(str(self.config),str(self.checkpoint),device='cuda:0' if self._cuda() else 'cpu')
        except Exception as e:
            raise RuntimeError('SegNeXt/MMsegmentation is not installed or config/checkpoint is invalid. Run scripts/08_setup_segnext.py and training first.') from e
    @staticmethod
    def _cuda():
        import torch; return torch.cuda.is_available()
    def predict(self,image):
        from mmseg.apis import inference_model
        result=inference_model(self.model,image)
        pred=result.pred_sem_seg.data[0].detach().cpu().numpy().astype(np.uint8)
        return pred
