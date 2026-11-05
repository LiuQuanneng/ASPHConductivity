"""Bounded sequential continuation, Windows/Linux; outputs isolated from inputs."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
R=Path(__file__).resolve().parents[1]
def main():
 from unpack import unpack
 unpack()
 p=argparse.ArgumentParser()
 p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
 p.add_argument('--threads',type=int,default=2)
 p.add_argument('--hours',type=float,default=8)
 p.add_argument('--case',choices=['both','history_a','history_b'],default='both')
 p.add_argument('--model',type=Path,default=R/'models/mace-mpa-0-medium.model')
 p.add_argument('--smoke',action='store_true',help='20 integration steps per case, separate output')
 a=p.parse_args()
 if not 1<=a.threads<=4 or not 0<a.hours<=8:p.error('threads must be 1..4, hours must be >0 and <=8')
 out=R/'runs'/('smoke' if a.smoke else 'production');out.mkdir(parents=True,exist_ok=True)
 lock=(out/'queue.lock').open('a+b')
 if os.name=='nt':
  import msvcrt
  lock.seek(0)
  if not lock.read(1):lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 else:
  import fcntl
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 end=time.time()+a.hours*3600
 deadline=datetime.fromtimestamp(end,timezone.utc).isoformat()
 env=os.environ.copy();env['MLIP_THREADS']=str(a.threads)
 jobs={};failed=False
 try:
  for case in (['history_a','history_b'] if a.case=='both' else [a.case]):
   target=50
   if a.smoke:target=json.loads((R/'snapshot.json').read_text())['cases'][case]['snapshot_ps']+.02
   if time.time()>=end:failed=True;break
   cmd=[sys.executable,str(R/'scripts/worker.py'),'--case',case,'--deadline',deadline,'--device',a.device,'--model',str(a.model.resolve()),'--target-ps',str(target),'--output-root',str(out)]
   with (out/(case+'.log')).open('a',encoding='utf-8') as log:
    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    jobs[case]={'pid':child.pid,'state':'running'};(out/'queue.json').write_text(json.dumps(jobs,indent=2))
    try:code=child.wait(timeout=max(1,end-time.time()))
    except subprocess.TimeoutExpired:child.kill();child.wait();code=124
    except KeyboardInterrupt:child.terminate();child.wait();raise
    jobs[case].update(state='completed' if code==0 else 'stopped',exit_code=code)
    (out/'queue.json').write_text(json.dumps(jobs,indent=2))
    print(case,jobs[case],flush=True)
    if code:failed=True;break
 finally:lock.close()
 if failed:raise SystemExit(1)
if __name__=='__main__':main()
