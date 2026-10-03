"""Launch the official SegNeXt/MMsegmentation trainer with the project config.
You must edit configs/segnext_plantseg.py after inspecting the exact PlantSeg release you downloaded.
"""
import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--repo',default='external/SegNeXt');p.add_argument('--config',default='configs/segnext_plantseg.py');p.add_argument('--gpus',default='1');p.add_argument('--checkpoint',default='');a=p.parse_args();
cmd=['python',f'{a.repo}/tools/train.py',a.config]
if a.checkpoint: cmd += ['--cfg-options',f'load_from={a.checkpoint}']
print('RUNNING:', ' '.join(cmd)); subprocess.check_call(cmd)
