from pathlib import Path
from urllib.request import urlretrieve
out=Path('artifacts/sam2/sam2.1_hiera_base_plus.pt');out.parent.mkdir(parents=True,exist_ok=True)
url='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt'
if out.exists():print('Already exists:',out)
else:
 print('Downloading official SAM 2.1 Base+ checkpoint...');urlretrieve(url,out);print('Saved',out)
