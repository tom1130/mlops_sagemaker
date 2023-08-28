import os
import argparse
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
        # label column = ['std','group','FR_LNG",'MR_LNG,PR_LNG]
        self.df = self.df.merge(self.label, on=['std','group','lounge'], how='inner')
        
        pass

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

        # melt 함수 사용

        return df_agg
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='monitoring')
    # fr
    parser.add_argument('--strFrLoungePath', default='')
    # mr
    parser.add_argument('--strMrLoungePath', default='')
    # pr
    parser.add_argument('--strPrLoungePath', default='')