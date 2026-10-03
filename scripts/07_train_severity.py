import argparse,yaml
from plantguard_ml.models.severity import train
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/plantguard.yaml');p.add_argument('--csv');a=p.parse_args();c=yaml.safe_load(open(a.config));csv=a.csv or c['severity']['labels_csv'];print('Training severity model from',csv);train(csv,'artifacts/severity/xgb_regressor.json');print('saved artifacts/severity/xgb_regressor.json')
