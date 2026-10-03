import subprocess,sys
from pathlib import Path
root=Path('external');root.mkdir(exist_ok=True);repo=root/'sam2'
if not repo.exists(): subprocess.check_call(['git','clone','https://github.com/facebookresearch/sam2.git',str(repo)])
subprocess.check_call([sys.executable,'-m','pip','install','-e',str(repo)])
print('SAM2 installed. Download sam2.1_hiera_base_plus.pt into artifacts/sam2/. Official checkpoints are listed in the SAM2 repository.')
