import subprocess,sys
from pathlib import Path
root=Path('external');root.mkdir(exist_ok=True)
repo=root/'SegNeXt'
if not repo.exists(): subprocess.check_call(['git','clone','https://github.com/Visual-Attention-Network/SegNeXt.git',str(repo)])
subprocess.check_call([sys.executable,'-m','pip','install','-e',str(repo)])
print('SegNeXt installed. Official implementation is based on MMSegmentation; see docs/MANUAL.md for PlantSeg config/data conversion.')
