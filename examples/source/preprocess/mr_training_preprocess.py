import os
import argparse

import pandas as pd

from preprocess import Preprocess



class mr_training_preprocess(Preprocess):

    def __init__(self, args):
        super().__init__(args)

    def logic(self, df, label):
        
        # pr data logic
        df = self._transform_base_data(df)
        df = self._drop_useless_date(df)
        df = self._make_mr_features(df)
        df = self._get_mr_dataset(df)
        df = self._merge_data_label(df, label)

        # split train, valid, test
        train, validation, test = self._train_test_split(df)        

        # save train, valid, test
        train_path = os.path.join(self.args.strDataPath, 'output','train','lounge.csv')
        validation_path = os.path.join(self.args.strDataPath, 'output','validation','lounge.csv')
        test_path = os.path.join(self.args.strDataPath, 'output','test','lounge.csv')
        train.to_csv(train_path, index=False)
        validation.to_csv(validation_path, index=False)
        test.to_csv(test_path, index=False)

    def execution(self):
        df = pd.read_csv(os.path.join(self.args.strDataPath, 'input', self.args.strDataName))
        label = pd.read_csv(os.path.join(self.args.strDataPath, 'input', self.args.strLabelName))
        label = self._preprocess_label(label)

        self.logic(df, label)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='mr_train_preprocessing')

    AWS_yn = True

    if AWS_yn:
        parser.add_argument('--strLoungeName', default='MR')
        parser.add_argument('--strDataPath', default='/opt/ml/processing')
        parser.add_argument('--strDataName', default='pnr_agg_data_20230815.csv')
        parser.add_argument('--strLabelName', default='lng_agg_data.csv')
        parser.add_argument('--strHoliday', default='holiday.csv')
        parser.add_argument('--listYears', type=list, default=[2021,2022,2023])
    
    else:
        parser.add_argument('--strLoungeName', default='MR')
        parser.add_argument('--strDataPath', default='c://Users/고기호/Desktop/vscode/mlops/examples/data/raw')
        parser.add_argument('--strDataName', default='pnr_agg_data_20230815.csv')
        parser.add_argument('--strLabelName', default='lng_agg_data.csv')
        parser.add_argument('--listYears', type=list, default=[2021,2022,2023])
    
    args = parser.parse_args()
    prep = mr_training_preprocess(args)
    prep.execution()
    