"""Generate plots from the fixed-parameter results and final CatBoost model."""
from pathlib import Path
import os, sys, json, ast, hashlib, time
os.environ.update(MPLBACKEND='Agg', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from scipy.stats import gaussian_kde

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'publication_fixed_S1'
PUB=ROOT
RUN=OUT/'preprocessing'
OUT.mkdir(exist_ok=True)
for d in ['figures','data','model','qa']: (OUT/d).mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(obj,p): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def savefig(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=400,facecolor='white',bbox_inches='tight')
    plt.close(fig)

def main():
    started=time.monotonic()
    data=PUB/'data/features/Magpie-AnCan.csv'
    df=pd.read_csv(data); y=df['Ionic Conductivity (Log)'].to_numpy()
    pre=joblib.load(RUN/'Magpie-AnCan/preprocessing.joblib')
    names=pre['selected']; x=pre['x']; remain=pre['remaining']
    # Explicit ordered selection must reproduce the exact CV input array.
    regenerated=pre['scaler'].transform(df[remain].to_numpy())[:,[remain.index(n) for n in names]]
    assert np.array_equal(x,regenerated)
    params=ast.literal_eval(json.loads((PUB/'config/manuscript_table_S1_hyperparameters.json').read_text())['parameters_as_reported']['CatBoost'])
    modelpath=OUT/'model/catboost_fixed_S1_all497.joblib'
    if modelpath.exists(): model=joblib.load(modelpath)
    else:
        model=CatBoostRegressor(**params,random_state=42,thread_count=2,verbose=False,allow_writing_files=False)
        model.fit(x,y)
        joblib.dump(model,modelpath)
        model.save_model(str(OUT/'model/catboost_fixed_S1_all497.cbm'))
    joblib.dump(pre['scaler'],OUT/'model/scaler.joblib')
    info=dict(original_feature_names=df.columns[2:].tolist(),remaining_feature_names=remain,selected_feature_names=names,dropped_features=pre['dropped'],selection_order='SHAP ranking; explicit name reindexing required',training_samples=497,model_role='All-data deployment/explanation fit; CV metrics reported separately')
    joblib.dump(info,OUT/'model/feature_info.joblib')
    dump(info,OUT/'model/feature_info.json')
    shap_path=OUT/'data/catboost_shap_values.npy'
    if shap_path.exists(): sv=np.load(shap_path)
    else:
        sv=model.get_feature_importance(Pool(x,y),type='ShapValues',thread_count=2)
        np.save(shap_path,sv)
    assert np.allclose(sv[:,:-1].sum(axis=1)+sv[:,-1],model.predict(x),atol=1e-8)
    imp=pd.DataFrame({'Feature':names,'mean_abs_SHAP':np.abs(sv[:,:-1]).mean(axis=0)}).sort_values('mean_abs_SHAP',ascending=False)
    imp.to_csv(OUT/'data/final_catboost_SHAP_importance.csv',index=False)
    top=imp.head(20).Feature.tolist()
    corr=pd.DataFrame(x,columns=names)[top].corr()
    corr.to_csv(OUT/'data/Fig4_correlation.csv')
    all42=pd.read_csv(OUT/'data/Table_S2_all42_fixed_S1.csv')[['combination','model','MAE','R2','run_origin']].copy()
    all42.to_csv(OUT/'data/Table_S2_all42_fixed_S1.csv',index=False)
    keys=['An','Can','AnCan','Magpie','Magpie-An','Magpie-Can','Magpie-AnCan']
    cat=all42[all42.model.eq('CatBoost')].set_index('combination').loc[keys].reset_index()
    cat.to_csv(OUT/'data/Fig3_CatBoost_fixed_S1.csv',index=False)
    labels={'An':'An','Can':'Can','AnCan':'An + Can','Magpie':'Magpie','Magpie-An':'Magpie + An','Magpie-Can':'Magpie + Can','Magpie-AnCan':'Magpie + An + Can'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def box(ax,xy,w,h,text,color='#edf3f9',size=10):
        ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.02',fc=color,ec='#68809b',lw=1))
        ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=size)
    def arrow(ax,a,b): ax.annotate('',b,a,arrowprops=dict(arrowstyle='->',lw=1.4,color='#426682'))
    fig,ax=plt.subplots(figsize=(9,6));ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
    box(ax,(.3,5.8),3.1,.85,'497 structure–conductivity\nrecords · stored log₁₀ target')
    box(ax,(4,5.8),2.5,.85,'Crystal structure\nAtom-specific\npoint clouds',size=9)
    box(ax,(7.1,5.8),2.5,.85,'Chemical composition\nMagpie descriptors')
    box(ax,(4,4.55),2.5,.8,'Persistent homology\nGlobal and Li/O/P/S\nstatistics',size=9)
    box(ax,(7.1,4.55),2.5,.8,'132 composition\ndescriptors')
    arrow(ax,(5.25,5.8),(5.25,5.35));arrow(ax,(8.35,5.8),(8.35,5.35))
    box(ax,(.5,3.05),9,.95,'Seven released descriptor sets · Six regression algorithms\n|Pearson r| ≥ 0.95 filtering → StandardScaler → XGBoost SHAP top 30%',size=11)
    arrow(ax,(1.85,5.8),(1.85,4));arrow(ax,(5.25,4.55),(5.25,4));arrow(ax,(8.35,4.55),(8.35,4))
    box(ax,(.5,1.55),3.9,.95,'Fixed Table S1 parameters\n5 repeats × 5-fold evaluation\nPreprocessing before CV',size=10)
    box(ax,(5,1.55),4.5,.95,'Full CatBoost: MAE 0.804 · R² 0.721\nvs retrained Magpie: MAE −5.7%\nRelative R² +5.8%',color='#f6f2d8',size=10)
    arrow(ax,(2.45,3.05),(2.45,2.5));arrow(ax,(4.4,2.025),(5,2.025))
    box(ax,(.5,.15),3.9,.85,'All-data CatBoost fit\nSHAP interpretation')
    box(ax,(5,.15),4.5,.85,'3,082 archived MP candidates\nE₉ > 1.5 eV · Eₕᵤₗₗ < 0.03 eV\nPredicted ranking → top 50',size=10)
    arrow(ax,(7.25,1.55),(7.25,1));arrow(ax,(5,1.95),(2.45,1))
    savefig(fig,'Fig1_fixed_S1_workflow')
    fig,ax=plt.subplots(figsize=(8.2,5.2));ax.set(xlim=(0,8),ylim=(0,6.5));ax.axis('off')
    ax.text(4,6.15,'497 samples · preprocessing fitted before cross-validation',ha='center',fontsize=12)
    ax.text(4,5.7,'Correlation filtering → scaling → SHAP selection',ha='center',fontsize=10)
    for fold in range(5):
        yy=4.65-fold*.78
        ax.text(.35,yy+.2,f'Fold {fold+1}',ha='left',va='center')
        for j in range(5):
            ax.add_patch(Rectangle((1.6+j*1.15,yy),1.02,.45,fc='#e77e75' if j==fold else '#6ba6ca',ec='white'))
    ax.text(4,5.15,'One repeat shown',ha='center',fontsize=10)
    ax.text(4,.48,'5 repeats, split seeds 42–46 → mean of 25 fold metrics',ha='center',fontsize=11)
    ax.text(4,.05,'Blue: training folds     Red: held-out validation fold\nHeld-out folds also used as eval_set for boosted models; no independent test set',ha='center',fontsize=9)
    savefig(fig,'FigS1_repeated_CV_protocol')
    fig,ax=plt.subplots(figsize=(8.2,5.6))
    for i,r in cat.iterrows():
        k=r.combination
        marker,color,size=('^','#db5444',90) if i<3 else (('o','#3b82c4',90) if i==3 else (('D','#8353b7',75) if i<6 else ('*','#ffd23f',320)))
        ax.scatter(r.MAE,r.R2,marker=marker,c=color,s=size,edgecolor='#333333',linewidth=.8,zorder=5)
        offsets={'An':(-15,-20),'Can':(10,-5),'AnCan':(10,-5),'Magpie':(15,-10),'Magpie-An':(55,12),'Magpie-Can':(55,-8),'Magpie-AnCan':(12,38)}
        dx,dy=offsets[k]
        ax.annotate(labels[k]+f' ({r.MAE:.3f}, {r.R2:.3f})',(r.MAE,r.R2),xytext=(dx,dy),textcoords='offset points',ha='left' if dx>=0 else 'right',fontsize=9,
            arrowprops=dict(arrowstyle='-',color='#888888',lw=.7) if k.startswith('Magpie-') else None)
    ax.set(xlabel='Mean absolute error in log₁₀(σ / S cm⁻¹)',ylabel='Coefficient of determination, R²',xlim=(.77,1.40),ylim=(.33,.815))
    ax.grid(alpha=.25)
    ax.text(.025,.04,'CatBoost · 5 repeats × 5 folds\nMagpie baseline trained in this study\nFull vs Magpie: MAE −5.7%; relative R² +5.8%',ha='left',va='bottom',transform=ax.transAxes,fontsize=9)
    savefig(fig,'Fig3_fixed_S1')
    fig,ax=plt.subplots(figsize=(10.2,9.0))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(20),top,rotation=65,ha='right',fontsize=8)
    ax.set_yticks(range(20),top,fontsize=8)
    for i in range(20):
        for j in range(20):
            ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=5.6,color='white' if abs(corr.iloc[i,j])>.65 else '#222222')
    fig.colorbar(im,ax=ax,fraction=.042,pad=.03,label='Pearson r')
    savefig(fig,'Fig4_final_CatBoost_correlation')
    # Deterministic SHAP scatter summary, raw feature names from the final model.
    fig,ax=plt.subplots(figsize=(9,6.8));rng=np.random.default_rng(42)
    last=None
    for j,n in enumerate(top[::-1]):
        idx=names.index(n);vals=x[:,idx]
        lo,hi=np.percentile(vals,[5,95]); colors=np.clip((vals-lo)/(hi-lo if hi>lo else 1),0,1)
        last=ax.scatter(sv[:,idx],j+rng.uniform(-.27,.27,len(x)),c=colors,cmap='coolwarm',s=9,alpha=.65,vmin=0,vmax=1,edgecolors='none',rasterized=True)
    ax.set_yticks(range(20),top[::-1],fontsize=9);ax.axvline(0,color='#888888',lw=.8)
    ax.set_xlabel('SHAP value in log₁₀(σ / S cm⁻¹)');ax.set_ylim(-.7,19.7)
    cb=fig.colorbar(last,ax=ax,fraction=.025,pad=.03);cb.set_ticks([0,1],labels=['Low','High']);cb.set_label('Standardized feature value')
    savefig(fig,'Fig5_final_CatBoost_SHAP')
    # One explicit fold used only for distribution illustration, never as a new test result.
    tr,te=next(KFold(5,shuffle=True,random_state=42).split(x))
    pd.DataFrame({'ID':df.ID,'partition':['Validation fold 1' if i in set(te) else 'Training folds 2–5' for i in range(len(df))]}).to_csv(OUT/'data/Fig6_split_seed42_fold1.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(10.2,6.1))
    for ax,n in zip(axes.flat,top[:6]):
        a=x[:,names.index(n)];grid=np.linspace(min(a)-.5,max(a)+.5,220)
        for indices,color,label in [(tr,'#3b82c4','Training folds'),(te,'#d95d55','Validation fold')]:
            vals=a[indices]
            if np.std(vals)>0: ax.plot(grid,gaussian_kde(vals)(grid),color=color,lw=1.5,label=label)
        ax.set_title(n,fontsize=9);ax.set_xlabel('Standardized value',fontsize=8);ax.set_ylabel('Density',fontsize=8);ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8);fig.tight_layout()
    savefig(fig,'Fig6_fold1_feature_distributions')
    manifest={'date':'2026-10-22','source_data_sha256':sha(data),'fixed_parameters':params,'samples':len(df),'original_features':307,'after_correlation':len(remain),'selected':len(names),'feature_order_verified_against_CV':True,'SHAP_additivity_verified':True,'top20_final_CatBoost':top,'MAE_reduction_percent':100*(1-cat.iloc[-1].MAE/cat.iloc[3].MAE),'R2_relative_increase_percent':100*(cat.iloc[-1].R2/cat.iloc[3].R2-1),'top5_importance':imp.head(5).to_dict('records')}
    dump(manifest,OUT/'revision_manifest.json')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
    print('FIGURES_AND_FINAL_MODEL_COMPLETE',time.monotonic()-started,flush=True)

if __name__=='__main__': main()
