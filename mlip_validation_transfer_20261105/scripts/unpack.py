"""Expand losslessly compressed trajectories, without replacing existing data."""
import gzip,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def unpack():
 expected=json.loads((R/'trajectory_hashes.json').read_text())
 for name,sha in expected.items():
  dest=R/name
  if not dest.exists():
   payload=gzip.decompress(dest.with_suffix(dest.suffix+'.gz').read_bytes())
   if hashlib.sha256(payload).hexdigest()!=sha:raise ValueError('Trajectory hash mismatch: '+name)
   temp=dest.with_suffix('.unpacking');temp.write_bytes(payload);temp.replace(dest)
  if hashlib.sha256(dest.read_bytes()).hexdigest()!=sha:raise ValueError('Existing trajectory hash mismatch: '+name)
if __name__=='__main__':unpack()
