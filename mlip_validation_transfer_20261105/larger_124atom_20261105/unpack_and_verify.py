"""Verify and unpack the immutable 124-atom raw-data archive. No MD."""
from pathlib import Path
import hashlib,json,io,zipfile
R=Path(__file__).resolve().parent
m=json.loads((R/'archive_manifest.json').read_text());chunks=[]
for item in m['parts']:
 data=(R/item['name']).read_bytes()
 assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],item['name']
 chunks.append(data)
payload=b''.join(chunks);assert hashlib.sha256(payload).hexdigest()==m['zip_sha256']
with zipfile.ZipFile(io.BytesIO(payload)) as z:
 assert set(z.namelist())==set(m['files'])
 for name,sha in m['files'].items():
  dest=(R/name).resolve();assert dest.is_relative_to(R.resolve())
  data=z.read(name);assert hashlib.sha256(data).hexdigest()==sha,name
  if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==sha,'Existing file differs: '+name
  else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
print('Verified and extracted',len(m['files']),'original files; no MD started.')
