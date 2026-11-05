"""Fetch official weights or copy an existing checkpoint; verify exact SHA256."""
import argparse, hashlib, shutil, urllib.request
from pathlib import Path
R=Path(__file__).resolve().parents[1]
URL='https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model'
SHA='75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638'
p=argparse.ArgumentParser();p.add_argument('--from-file',type=Path);a=p.parse_args()
dest=R/'models/mace-mpa-0-medium.model';dest.parent.mkdir(exist_ok=True)
if not dest.exists():
 tmp=dest.with_suffix('.download')
 if a.from_file:shutil.copyfile(a.from_file,tmp)
 else:urllib.request.urlretrieve(URL,tmp)
 if hashlib.sha256(tmp.read_bytes()).hexdigest()!=SHA:raise SystemExit('Weight checksum mismatch; download retained, not activated')
 tmp.replace(dest)
if hashlib.sha256(dest.read_bytes()).hexdigest()!=SHA:raise SystemExit('Weight checksum mismatch')
print('Model verified:',dest)
