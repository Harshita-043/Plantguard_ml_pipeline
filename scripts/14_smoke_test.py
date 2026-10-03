import argparse,requests
p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--url',default='http://127.0.0.1:8000/v1/inference');a=p.parse_args()
with open(a.image,'rb') as f:r=requests.post(a.url,files={'file':(a.image,f,'image/jpeg')},timeout=300)
print(r.status_code);print(r.text)
