import os
import argparse
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

import boto3
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

class monitoring:
    
    def __init__(self, args):
        self.args = args
        
        
    def logic(self):
        # label column = ['std','group','lounge', 'label]
        self.df = self.df.merge(self.label, on=['std','group','lounge'], how='inner')

        # fr
        fr = self.df[self.df['lounge']=='FR']
        fr_r2 = r2_score(fr['predictions'], fr['lounge'])
        # mr
        mr = self.df[self.df['lounge']=='MR']
        mr_r2 = r2_score(mr['predictions'], mr['lounge'])
        # pr
        pr = self.df[self.df['lounge']=='PR']
        pr_r2 = r2_score(pr['predictions'], pr['lounge'])

        report_dict = {
            "r2" :
                {
                    'FR' : fr_r2,
                    'MR' : mr_r2,
                    'PR' : pr_r2
                }
        }

        # fr pipeline start
        if fr_r2 < self.args.floatFrThreshold:
            pass

        if mr_r2 < self.args.floatMrThreshold:
            pass

        if pr_r2 < self.args.floatPrThreshold:
            pass

        # save monitoring data
        output_path = os.path.join(self.args.strDataPath, 'output', self.args.strOutputName)
        with open(output_path, 'w') as f:
            f.write(json.dumps(report_dict))

    def execution(self):
        self._read_predictions()
        self._read_label()

    def _read_predictions(self):
        list_dir = os.listdir(os.path.join(self.args.strDataPath,'input'))
        min_date = self.min_date.replace('-','')
        max_date = self.max_date.replace('-','')
        df = pd.DataFrame(columns=['std','group','predictions','lounge'])

        for dir in list_dir:
            if dir>=min_date and dir<max_date:
                prediction = pd.read_csv(os.path.join(self.args.strDataPath,'input',dir,self.args.strDataName))
                prediction = prediction[prediction['std']==prediction['std'].min()]
                df = pd.concat([df, prediction])
        self.df = df

    def _read_label(self):
        # read csv file
        self.label = pd.read_csv(self.args.strDataPath, 'input', 'label', 'lounge.csv')
        
        # get date
        self.max_date = self.label['_date'].max()
        self.min_date = datetime.strftime(datetime.strptime(self.max_date,'%Y-%m-%d')-relativedelta(days=7),'%Y-%m-%d')

        self.label = self.label[(self.label['_date']>self.min_date)&(self.label['_date']<=self.max_date)]
        self.label = self._preprocess_label(self.label)

    def _preprocess_label(self, df):
        df['std'] = pd.to_datetime(df['_date']).dt.date
        
        df['FR'] = df.loc[(df['lng_type'] == 'FR'), 'ke'].astype('int16')
        df['MR'] = df.loc[(df['lng_type'] == 'MR'), 'ke'].astype('int16')
        df['PR'] = df.loc[(df['lng_type'] == 'PR'), 'ke'].astype('int16')

        df_agg = df.groupby(['time_group', 'std']).agg({
            c : 'sum' for c in ['FR', 'MR', 'PR']
        }).astype('int16').reset_index().set_index('std')

        df_agg.index.name = 'std'
        df_agg = df_agg.rename(columns = {'time_group' : 'group'})

        # unpivoting lounge column
        df_agg = pd.melt(df_agg, id_vars=['std','group'], var_name='lounge', value_name='label')
        return df_agg
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='monitoring')

    parser.add_argument('--strDataPath', default='/opt/ml/processing')
    parser.add_argument('--strDataName', default='pnr.csv')
    parser.add_argument('--strOutputName', default='monitoring.json')
    # fr
    parser.add_argument('--floatFrThreshold', default=0.7)
    # mr
    parser.add_argument('--floatMrThreshold', default=0.95)
    # pr
    parser.add_argument('--floatPrThreshold', default=0.95)