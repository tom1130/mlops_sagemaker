import os
import argparse
from dateutil.relativedelta import relativedelta
from datetime import datetime

import pandas as pd

from preprocess import Preprocess
from config.config import config_handler

class inference_preprocess(Preprocess):

    def __init__(self, args):
        super().__init__(args)

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
        fr_path = os.path.join(self.args.data_path, 'output','fr',self.args.data_name)
        mr_path = os.path.join(self.args.data_path, 'output','mr',self.args.data_name)
        pr_path = os.path.join(self.args.data_path, 'output','pr',self.args.data_name)
        
        fr.to_csv(fr_path, index=False, header=False)
        mr.to_csv(mr_path, index=False, header=False)
        pr.to_csv(pr_path, index=False, header=False)

    def execution(self):
        df = pd.read_csv(os.path.join(self.args.data_path,self.args.today,'input',self.args.daily_data_name))
        
        self._integrate_data(df)
        self.logic(df)
        
    def _integrate_data(self, df):
        # read integrated data
        intg_df = pd.read_csv(os.path.join(self.args.data_path,'integrate',self.args.data_name))
        # str -> date
        intg_df['std'] = pd.to_datetime(intg_df['std']).dt.date
        
        df['std'] = pd.to_datetime(df['std']).dt.date
        # data concat 
        intg_df = intg_df[~intg_df['std'].isin(df['std'].unique())]
        intg_df = pd.concat([intg_df, df])
        # cut data
        cut_date = intg_df['std'].max() - relativedelta(months=30)
        
        intg_df = intg_df[intg_df['std']>=cut_date]
        # save data
        intg_df.to_csv(os.path.join(self.args.data_path,'output','integrate',self.args.data_name), index=False)
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='inference_preprocessing')

    parser.add_argument('--data_path', default='/opt/ml/processing')
    parser.add_argument('--data_name', default='pnr.csv')
    parser.add_argument('--daily_data_name', default='pnr.000')
    parser.add_argument('--holiday_name', default='holiday.csv')
    parser.add_argument('--today', type=str, default='20230725')
    
    args, _ = parser.parse_known_args()
    # get config file
    args.config = config_handler('preprocess_config.ini')
    
    prep = inference_preprocess(args)
    prep.execution()