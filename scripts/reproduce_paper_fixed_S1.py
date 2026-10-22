"""Reproduce the revised fixed-S1 experiments from the saved ordered preprocessing."""
from pathlib import Path
import argparse,os,sys,json,ast,time,hashlib
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--output-dir',type=Path)
    p.add_argument('--combination',default='all',choices=['all','An','Can','AnCan','Magpie','Magpie-An','Magpie-Can','Magpie-AnCan'])
    p.add_argument('--models',nargs='+',default=['Decision Tree','Random Forest','Gradient Boosting','Support Vector Machine','XGBoost','CatBoost'])
    p.add_argument('--resume',action='store_true');p.add_argument('--fit-final',action='store_true');p.add_argument('--max-seconds',type=float,default=7200)
    args=p.parse_args()
    if not args.execute:print('No training. Add --execute --output-dir <new folder>. Fixed S1 only; no Optuna.');return
    if args.output_dir is None:p.error('--output-dir is required')
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    import numpy as np,pandas as pd,joblib
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error,r2_score
    from sklearn.base import clone
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    root=Path(__file__).resolve().parents[1];bundle=root/'publication_fixed_S1';out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:p.error('Existing output preserved; choose a new folder or use --resume.')
    out.mkdir(parents=True,exist_ok=True)
    classes={'Decision Tree':DecisionTreeRegressor,'Random Forest':RandomForestRegressor,'Gradient Boosting':GradientBoostingRegressor,'Support Vector Machine':SVR,'XGBoost':XGBRegressor,'CatBoost':CatBoostRegressor}
    if set(args.models)-set(classes):p.error('Unknown model name; enclose names containing spaces in quotes.')
    paramfile=root/'config/manuscript_table_S1_hyperparameters.json'
    params={k:ast.literal_eval(v) for k,v in json.loads(paramfile.read_text())['parameters_as_reported'].items()}
    table=root/'data/features/Magpie-AnCan.csv';frame=pd.read_csv(table);y=frame['Ionic Conductivity (Log)'].to_numpy()
    keys=['An','Can','AnCan','Magpie','Magpie-An','Magpie-Can','Magpie-AnCan'] if args.combination=='all' else [args.combination]
    hashes={}
    for f in [table,paramfile]+[bundle/'preprocessing'/k/'preprocessing.joblib' for k in keys]:hashes[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
    statefile=out/'manifest.json'
    if args.resume and statefile.exists() and json.loads(statefile.read_text())['input_hashes']!=hashes:p.error('Inputs differ from existing run; choose a new output folder.')
    state={'protocol':'Fixed S1, saved full-data preprocessing, held-out eval_set, 5x5 internal evaluation; no Optuna','input_hashes':hashes,'status':'running','models':args.models,'combinations':keys,'max_seconds':args.max_seconds}
    def save():statefile.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save();started=time.monotonic()
    try:
        for key in keys:
            folder=out/key;folder.mkdir(exist_ok=True);pre=joblib.load(bundle/'preprocessing'/key/'preprocessing.joblib');x=pre['x']
            cols=pre['remaining'];regenerated=pre['scaler'].transform(frame[cols].to_numpy())[:,[cols.index(n) for n in pre['selected']]]
            assert np.array_equal(x,regenerated)
            for name in args.models:
                kw=params[name].copy()
                if name!='Support Vector Machine':kw['random_state']=42
                if name=='XGBoost':kw.update(n_jobs=2,verbosity=0,eval_metric='rmse')
                if name=='CatBoost':kw.update(thread_count=2,verbose=False,allow_writing_files=False)
                base=classes[name](**kw)
                ff=folder/('folds_'+name.replace(' ','_')+'.csv')
                rows=pd.read_csv(ff).to_dict('records') if ff.exists() and args.resume else []
                for rep in range(5):
                    seed=42+rep
                    for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(x),1):
                        if any(r['repeat']==rep+1 and r['fold']==fold for r in rows):continue
                        if time.monotonic()-started>args.max_seconds:raise TimeoutError('Time budget reached; saved fold checkpoints can be resumed.')
                        model=clone(base)
                        if hasattr(model,'random_state'):model.random_state=seed+fold
                        if name in ['XGBoost','CatBoost']:
                            ev=(x[te],y[te]) if name=='CatBoost' else [(x[te],y[te])]
                            model.fit(x[tr],y[tr],eval_set=ev,verbose=False)
                        else:model.fit(x[tr],y[tr])
                        pred=model.predict(x[te]);rows.append({'repeat':rep+1,'fold':fold,'MAE':mean_absolute_error(y[te],pred),'R2':r2_score(y[te],pred)})
                        pd.DataFrame(rows).to_csv(ff,index=False)
                assert len(rows)==25
                metric={'combination':key,'model':name,'MAE':float(np.mean([r['MAE'] for r in rows])),'R2':float(np.mean([r['R2'] for r in rows])),'folds':25,'parameters':params[name]}
                (folder/('metrics_'+name.replace(' ','_')+'.json')).write_text(json.dumps(metric,indent=2),encoding='utf-8')
                print(json.dumps(metric),flush=True)
                if args.fit_final:
                    model=clone(base);model.fit(x,y)
                    joblib.dump(model,folder/('final_all_data_'+name.replace(' ','_')+'.joblib'))
        state['status']='complete'
    except TimeoutError as e:state['status']='time_budget_stopped';print(e)
    finally:state['elapsed_seconds']=time.monotonic()-started;save()
if __name__=='__main__':main()
