from __future__ import annotations
from pathlib import Path
import numpy as np
import xgboost as xgb

FEATURE_NAMES=['disease_area_pct','leaf_area_px','disease_components','mean_disease_prob','edge_contact_pct','leaf_count','classifier_confidence']

def make_features(leaf_mask, disease_mask, disease_prob, leaf_count, classifier_confidence):
    leaf_area=float(leaf_mask.sum())+1e-6
    disease_area=float((disease_mask>0).sum())
    ratio=100.0*disease_area/leaf_area
    ncomp=0
    try:
        import cv2
        ncomp,_=cv2.connectedComponents((disease_mask>0).astype('uint8'))
        ncomp=max(0,ncomp-1)
        border=np.concatenate([disease_mask[0,:],disease_mask[-1,:],disease_mask[:,0],disease_mask[:,-1]])>0
        edge=float(border.mean())*100
    except Exception:
        edge=0.0
    return np.array([ratio,leaf_area,ncomp,float(np.mean(disease_prob)),edge,leaf_count,classifier_confidence],dtype=np.float32)

def train(csv_path, out_path):
    import pandas as pd
    df=pd.read_csv(csv_path)
    X=df[FEATURE_NAMES].values; y=df['severity'].values
    model=xgb.XGBRegressor(n_estimators=500,max_depth=5,learning_rate=0.03,subsample=.8,colsample_bytree=.9,
                           objective='reg:squarederror',tree_method='hist',random_state=42)
    model.fit(X,y)
    Path(out_path).parent.mkdir(parents=True,exist_ok=True); model.save_model(out_path)
    return model
