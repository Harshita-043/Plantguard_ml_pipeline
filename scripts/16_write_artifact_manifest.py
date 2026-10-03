from pathlib import Path
import hashlib,json,datetime
root=Path('artifacts');rows=[]
for f in root.rglob('*'):
 if f.is_file():
  h=hashlib.sha256();
  with open(f,'rb') as x:
   for chunk in iter(lambda:x.read(1024*1024),b''):h.update(chunk)
  rows.append({'path':str(f),'bytes':f.stat().st_size,'sha256':h.hexdigest()})
out=root/'ARTIFACT_MANIFEST.json';out.write_text(json.dumps({'created_utc':datetime.datetime.utcnow().isoformat()+'Z','artifacts':rows},indent=2));print(out)
