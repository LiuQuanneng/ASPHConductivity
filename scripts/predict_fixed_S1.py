"""Predict with the revised all-data CatBoost model and explicit feature order."""
from pathlib import Path
import argparse,os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import joblib,numpy as np,pandas as pd
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1]

def predict(features,model_dir=None):
    folder=Path(model_dir) if model_dir else ROOT/'publication_fixed_S1/model'
    info=joblib.load(folder/'feature_info.joblib')
    remaining=info['remaining_feature_names'];selected=info['selected_feature_names']
    missing=set(remaining)-set(features.columns)
    if missing:raise ValueError('Missing required features: '+', '.join(sorted(missing)))
    values=features.loc[:,remaining].to_numpy(float)
    if not np.isfinite(values).all():raise ValueError('Required features contain NaN or infinity; no automatic imputation is applied.')
    scaler=joblib.load(folder/'scaler.joblib')
    x=scaler.transform(values)[:,[remaining.index(n) for n in selected]]
    model=CatBoostRegressor();model.load_model(str(folder/'catboost_fixed_S1_all497.cbm'))
    pred=model.predict(x,thread_count=2)
    ids=features['material_id'] if 'material_id' in features else (features['ID'] if 'ID' in features else pd.Series(np.arange(len(features))))
    return pd.DataFrame({'material_id':ids.to_numpy(),'Predicted_log10_conductivity_S_cm':pred,'Predicted_conductivity_mS_cm':1000*np.power(10.,pred)})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists; choose a new filename to preserve existing results.')
    frame=pd.read_csv(args.features);out=predict(frame)
    args.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(args.output,index=False)
    print('Predicted',len(out),'rows ->',args.output)
if __name__=='__main__':main()
