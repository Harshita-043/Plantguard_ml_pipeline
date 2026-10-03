from __future__ import annotations
import io, os
from pathlib import Path
import yaml
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from ..pipeline.inference import PlantGuardPipeline
from ..models.sam2_adapter import SAM2LeafSegmenter
from ..models.segnext_adapter import SegNeXtAdapter

ROOT=Path(__file__).resolve().parents[3]
cfg=yaml.safe_load((ROOT/'configs/plantguard.yaml').read_text())
app=FastAPI(title='PlantGuard ML Inference',version='1.0.0')
pipeline=PlantGuardPipeline(cfg)

@app.on_event('startup')
def startup():
    try:
        sam=SAM2LeafSegmenter(str(ROOT/cfg['paths']['sam2_checkpoint']),str(ROOT/'external/sam2'/cfg['sam2']['config']),pipeline.device)
        seg=SegNeXtAdapter(cfg['paths']['segnext_repo'],cfg['segnext']['config'],cfg['paths']['artifacts'] if False else cfg['segnext']['checkpoint'])
        pipeline.attach_heavy_models(sam,seg)
    except Exception as e:
        # Keep API importable for health checks; inference endpoint will explain the missing artifact.
        app.state.model_error=str(e)

@app.get('/health')
def health():
    return {'status':'ok','model_error':getattr(app.state,'model_error',None),'device':pipeline.device}

@app.post('/v1/inference')
async def inference(file:UploadFile=File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(415,'Upload a JPEG/PNG/WebP image')
    if getattr(app.state,'model_error',None): raise HTTPException(503,getattr(app.state,'model_error'))
    data=await file.read()
    try: img=Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as e: raise HTTPException(400,f'Invalid image: {e}')
    return pipeline.run(img)
