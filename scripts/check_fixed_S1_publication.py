"""Check the revised publication evidence without training any models."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import pandas as pd,numpy as np,joblib,json
from predict_fixed_S1 import predict
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'publication_fixed_S1';data=out/'data'
    summary=pd.read_csv(data/'Table_S2_all42_fixed_S1.csv');folds=pd.read_csv(data/'all42_fold_scores_1050.csv')
    assert len(summary)==42 and len(folds)==1050
    assert not summary.duplicated(['combination','model']).any()
    for (key,model),g in folds.groupby(['combination','model']):
        row=summary[summary.combination.eq(key)&summary.model.eq(model)].iloc[0]
        assert len(g)==25 and not g.duplicated(['repeat','fold']).any()
        assert np.allclose([g.MAE.mean(),g.R2.mean()],[row.MAE,row.R2],rtol=0,atol=1e-12)
    f3=pd.read_csv(data/'Fig3_CatBoost_fixed_S1.csv')
    merged=f3.merge(summary[summary.model.eq('CatBoost')],on=['combination','model'],validate='one_to_one')
    assert np.allclose(merged[['MAE_x','R2_x']],merged[['MAE_y','R2_y']],rtol=0,atol=1e-12)
    features=pd.read_csv(data/'candidate_features_307.csv');expected=pd.read_csv(data/'candidate_predictions_fixed_S1.csv')
    fresh=predict(features);check=expected.merge(fresh,on='material_id',validate='one_to_one',suffixes=('_saved','_fresh'))
    assert len(check)==3082 and np.allclose(check.Predicted_log10_conductivity_S_cm_saved,check.Predicted_log10_conductivity_S_cm_fresh,rtol=0,atol=1e-11)
    assert np.allclose(expected.Predicted_conductivity_mS_cm,1000*10.**expected.Predicted_log10_conductivity_S_cm,rtol=1e-12)
    assert expected.Predicted_conductivity_mS_cm.is_monotonic_decreasing
    top=pd.read_csv(data/'Table_S4_top50_fixed_S1.csv');assert top.material_id.tolist()==expected.head(50).material_id.tolist()
    audit=pd.read_csv(data/'training_barcode_reconstruction_audit.csv');assert len(audit)==497 and audit.status.eq('matched').all()
    pre=joblib.load(out/'preprocessing/Magpie-AnCan/preprocessing.joblib');sv=np.load(data/'catboost_shap_values.npy');m=joblib.load(out/'model/catboost_fixed_S1_all497.joblib')
    assert np.allclose(sv[:,:-1].sum(1)+sv[:,-1],m.predict(pre['x']),atol=1e-8)
    print('PASS: 42 mean scores, 1050 folds, Fig3, 497 barcode reconstructions, 3082 saved-model predictions, TableS4 ranking, conductivity units, SHAP additivity.')
if __name__=='__main__':main()
