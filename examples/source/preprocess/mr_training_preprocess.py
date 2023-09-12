import os
import argparse

import pandas as pd

from preprocess import Preprocess
from config.config import config_handler

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
        train_path = os.path.join(self.args.data_path, 'output','train','pnr.csv')
        validation_path = os.path.join(self.args.data_path, 'output','validation','pnr.csv')
        test_path = os.path.join(self.args.data_path, 'output','test','pnr.csv')
        train.to_csv(train_path, index=False)
        validation.to_csv(validation_path, index=False)
        test.to_csv(test_path, index=False)

    def execution(self):
        df = pd.read_csv(os.path.join(self.args.data_path, 'input', self.args.data_name))
        label = pd.read_csv(os.path.join(self.args.data_path, 'input', self.args.label_name))
        label = self._preprocess_label(label)

        self.logic(df, label)

if __name__=='__main__':
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
    parser.add_argument('--today', default='20230725')
    
    args, _ = parser.parse_known_args()
    # get config file
    args.config = config_handler('preprocess_config.ini')
    
    prep = mr_training_preprocess(args)
    prep.execution()
    