import os
import argparse

import pandas as pd

from preprocess import Preprocess



class pr_training_preprocess(Preprocess):

    def __init__(self, args):
        super().__init__(args)

    def logic(self, df, label):
        
        # pr data logic
        df = self._transform_base_data(df)
        df = self._drop_useless_date(df)
        df = self._make_pr_features(df)
        df = self._get_pr_dataset(df)
        df = self._merge_data_label(df, label)

        # save pr data
        self.train_test_split(df)        

    def execution(self):
        df = pd.read_csv(os.path.join(self.args.strDataPath, 'input', self.args.strDataName))
        label = pd.read_csv(os.path.join(self.args.strDataPath, 'input', self.args.strLabelName))
        label = self._preprocess_label(label)

        self.logic(df, label)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='pr_train_preprocessing')

    parser.add_argument('--strLoungeName', default='PR')
    parser.add_argument('--prefix_prep', type=str, default='/opt/ml/processing')
    parser.add_argument('--strDataPath', default='/opt/ml/processing')
    # parser.add_argument('--strDataPath', default='c:\\Users\\고기호\\Desktop\\vscode\\mlops\\examples\\data\\raw')
    parser.add_argument('--strDataName', default='pnr_agg_data_20230815.csv')
    parser.add_argument('--strLabelName', default='lng_agg_data.csv')
    parser.add_argument('--strHoliday', default='holiday.csv')
    parser.add_argument('--listYears', type=list, default=[2021,2022,2023])

    parser.add_argument('--strPreprocessPath', default='train_preprocess/')
    args = parser.parse_args()

    prep = pr_training_preprocess(args)
    prep.execution()
    