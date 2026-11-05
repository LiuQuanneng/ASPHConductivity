"""Offline integrity and checkpoint/trajectory consistency check; no MD."""
import hashlib,json
from pathlib import Path
import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory
R=Path(__file__).resolve().parents[1]
from unpack import unpack
unpack()
for name,sha in json.loads((R/'manifest.json').read_text()).items():
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha,name
for case,info in json.loads((R/'snapshot.json').read_text())['cases'].items():
 f=R/'inputs'/case;meta=json.loads(next(f.glob('checkpoint_*.json')).read_text());a=read(f/meta['trajectory'])
 with Trajectory(str(f/'segment_001.traj')) as t:
  assert len(t)==info['frames'];last=t[-1]
  steps=[int(x.info.get('total_step',x.info.get('thermal_step',x.info.get('time_fs',-1)))) for x in t]
 assert np.all(np.diff(steps)==20) and steps[-1]==meta['step']
 assert np.allclose(a.positions,last.positions,atol=1e-10)
 assert np.allclose(a.get_momenta(),last.get_momenta(),atol=1e-10)
 r=meta['rng'];rng=np.random.RandomState();rng.set_state((r[0],np.array(r[1],dtype=np.uint32),r[2],r[3],r[4]))
 print(case,info['snapshot_ps'],'ps;',len(a),'atoms; positions, momenta, RNG and continuous history verified')
print('All package hashes verified')
