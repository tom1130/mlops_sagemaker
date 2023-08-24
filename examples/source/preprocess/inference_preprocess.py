import os
import argparse
from dateutil.relativedelta import relativedelta
from datetime import datetime

import pandas as pd

from preprocess import Preprocess
'''
processing input
1. today 설정 -> strptime으로 변경
2. input
    (1) input : strDatapath(integration)
        destination : prefix/integrate/
    (2) input : strDatapath/today
        destination : prefix/today/input
    (3) input : strDatapath/yesterday
        destination : prefix/yesterday/input
    (4) input : strdatapath/etc
        destination : prefix/etc
3. output
    (1) input : strDataPath, output
        destination : prefix/inference/today
    (2) input : strDataPath, integrate
        destination : prefix/train/raw

processing output
'''
class inference_preprocess(Preprocess):

    def __init__(self, args):
        super().__init__(args)
        self.today = args.today_date
        self.yesterday = datetime.strftime(datetime.strptime(args.today_date,'%Y%m%d')-datetime.timedelta(days=1),'%Y%m%d')

    def logic(self, df):
        # total process
        df = self._transform_base_data(df)

        # pr data logic
        pr = self._make_pr_features(df)
        pr = self._get_pr_dataset(pr)

        # fr data logic
        fr = self._get_fr_dataset(df)

        # mr data logic
        mr = self._make_mr_features(df)
        mr = self._get_mr_dataset(mr)

        # save data
        fr_path = os.path.join(self.args.strDataPath, 'output','fr','pnr.csv')
        mr_path = os.path.join(self.args.strDataPath, 'output','mr','pnr.csv')
        pr_path = os.path.join(self.args.strDataPath, 'output', 'pr','pnr.csv')
        
        fr.to_csv(fr_path, index=False, header=False)
        mr.to_csv(mr_path, index=False, header=False)
        pr.to_csv(pr_path, index=False, header=False)

    def execution(self):
        df = pd.read_csv(os.path.join(self.args.strDataPath,self.today,'input',self.args.strDataName))
        label = pd.read_csv(os.path.join(self.args.strDataPath,self.yesterday,'input',self.args.strLabelName))
        
        self._integrate_data(df, label)
        self.logic(df)
        
    def _integrate_data(self, df, label):
        # read integrated data
        intg_df = pd.read_csv(os.path.join(self.args.strDataPath,'integrate',self.args.strDataName))
        intg_label = pd.read_csv(os.path.join(self.args.strDataPath,'integrate', self.args.strDataName))
        # str -> date
        intg_df['std'] = pd.to_datetime(intg_df['std']).dt.date
        intg_label['_date'] = pd.to_datetime(intg_label['_date']).dt.date
        
        df['std'] = pd.to_datetime(df['std']).dt.date
        label['_date'] = pd.to_datetime(label['_date']).dt.date
        # concat 
        intg_df = pd.concat([intg_df, df[df['std']==df['std'].min()]])
        intg_label = pd.concat([intg_label, label])
        # cut data
        cut_date = intg_label['_date'].max() - relativedelta(months=30)
        
        intg_df = intg_df[intg_df['std']>=cut_date]
        intg_label = intg_label[intg_label['_date']>=cut_date]
        # save data
        intg_df.to_csv(os.path.join(self.args.strDataPath,'output','integrate',self.args.strDataName))
        intg_label.to_csv(os.path.join(self.args.strDataPath,'output','integrate',self.args.strLabelName))
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='inference_preprocessing')

    AWS_yn = True

    if AWS_yn:
        parser.add_argument('--strDataPath', default='/opt/ml/processing')
        parser.add_argument('--strDataName', default='pnr.csv')
        parser.add_argument('--strLabelName', default='lounge.csv')
        parser.add_argument('--strHoliday', default='holiday.csv')
        parser.add_argument('--listYears', type=list, default=[2021,2022,2023])
        parser.add_argument('--today', type=str, default='20230725')
    else:
        parser.add_argument('--strDataPath', default='c://Users/고기호/Desktop/vscode/mlops/examples/data/raw')
        parser.add_argument('--strDataName', default='pnr_agg_data_20230815.csv')
        parser.add_argument('--strLabelName', default='lng_agg_data.csv')
        parser.add_argument('--strHoliday', default='holiday.csv')
        parser.add_argument('--listYears', type=list, default=[2021,2022,2023])
    
    args = parser.parse_args()
    prep = inference_preprocess(args)
    prep.execution()