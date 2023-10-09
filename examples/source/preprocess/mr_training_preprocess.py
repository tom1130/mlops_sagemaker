import os
import argparse

import pandas as pd

from preprocess import Preprocess
from config.config import config_handler

from utils.notification import *

class MrTrainingProcess(Preprocess):

    def __init__(self, args):
        super().__init__(args)
        # 휴일 데이터 불러오기 및 type 변경
        hol_df = pd.read_csv(os.path.join(self.args.data_path, 'etc', self.args.holiday_name))
        hol_df['hol_date'] = pd.to_datetime(hol_df['hol_date']).dt.date
        self.hol_df = hol_df.set_index('hol_date')


    def logic(self, df, label):
        
        # pr data logic
        df = self._transform_base_data(df)
        df = self._drop_useless_date(df)
        df = self._make_mr_features(df)
        df = self._get_mr_dataset(df)
        label = self._preprocess_label(label)
        df = self._merge_data_label(df, label)

        # split train, valid, test
        train, validation, test = self._train_test_split(df)        

        # save train, valid, test
        train_path = os.path.join(self.args.data_path, 'output','train','pnr.csv')
        validation_path = os.path.join(self.args.data_path, 'output','validation','pnr.csv')
        test_pnr_path = os.path.join(self.args.data_path, 'output','test','pnr.csv') # features
        test_lounge_path = os.path.join(self.args.data_path, 'output','test','lounge.csv') # label

        train.to_csv(train_path, index=False)
        validation.to_csv(validation_path, index=False)
        test_pnr = test.drop(['target'], axis=1) # features
        test_pnr.to_csv(test_pnr_path, index=False, header=None)
        test_lounge = test[['target']] # label
        test_lounge.to_csv(test_lounge_path, index=False, header=None)
        print('Save train, validation, test data')

    def execution(self):

        df = pd.read_csv(os.path.join(self.args.data_path, 'input', self.args.data_name))
        label = pd.read_csv(os.path.join(self.args.data_path, 'input', self.args.label_name))

        self.logic(df, label)
        print('mr training preprocess process is completed')

if __name__=='__main__':
    try:
        # set path
        strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
        os.chdir(strBasePath)
        # arguments
        parser = argparse.ArgumentParser(description='fr_train_preprocessing')
        
        parser.add_argument('--lounge_name', default='MR')
        parser.add_argument('--data_path', default='/opt/ml/processing')
        parser.add_argument('--data_name', default='pnr.csv')
        parser.add_argument('--label_name', default='lounge.csv')
        parser.add_argument('--holiday_name', default='holiday.csv')
        parser.add_argument('--region', type=str, default="ap-northeast-2")
        parser.add_argument('--sns_arn', default='arn:aws:sns:ap-northeast-2:441848216151:awsdc-sns-dlk-dev-topic-ml-2023p01-lounge')
        parser.add_argument('--project_name', type=str)
        parser.add_argument('--pipeline_name', type=str)

        parser.add_argument('--env', type=str)
        
        args, _ = parser.parse_known_args()
        # get config file
        args.config = config_handler('preprocess_config.ini')
        
        prep = MrTrainingProcess(args)
        prep.execution()

    except Exception as e:

        publish_sns(
            region_name=args.region,
            sns_arn=args.sns_arn,
            project_name=args.project_name,
            pipeline_name=args.pipeline_name,
            error_type="MR Preprocess Step 실행 중 에러",
            error_message=e,
            env=args.env,
        )

        raise Exception('step error')