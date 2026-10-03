import sys, importlib.util
print('Python:',sys.version)
print('Executable:',sys.executable)
for m in ['torch','torchvision','PIL','numpy','sklearn','xgboost','fastapi','onnx','onnxruntime']:
    try:
        mod=__import__(m); print(f'{m}: {getattr(mod,"__version__","installed")}')
    except Exception as e: print(f'{m}: MISSING ({e})')
try:
 import torch; print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
except: pass
