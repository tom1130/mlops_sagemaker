import os
import argparse

import pandas as pd

class Postprocess:

    def __init__(self, args):
        self.args = args

    def logic(self):
        # fr
        self.fr_result = self.fr_result[[0, self._find_grp_columns(self.fr_result), len(self.fr_result.columns)-1]]
        self.fr_result.columns = ['std', 'group', 'predictions']
        self.fr_result['lounge'] = 'FR'
        
        # mr
        self.mr_result = self.mr_result[[0, self._find_grp_columns(self.mr_result), len(self.mr_result.columns)-1]]
        self.mr_result.columns = ['std', 'group', 'predictions']
        self.mr_result['lounge'] = 'MR'

        # pr
        self.pr_result = self.pr_result[[0, self._find_grp_columns(self.pr_result), len(self.pr_result.columns)-1]]
        self.pr_result.columns = ['std', 'group', 'predictions']
        self.pr_result['lounge'] = 'PR'

        # union
        self.result = pd.concat([self.fr_result, self.mr_result, self.pr_result])

        # change column name
        self.result.columns = ['date','time_group','pred','lounge_type']
        self.result = self.result[['date','lounge_type','time_group','pred']]

        # save
        self.result.to_csv(os.path.join(self.args.data_path, 'output', 'pnr.csv'), index=False)

    def execution(self):
        # define file path
        fr_path = os.path.join(self.args.data_path, 'input', 'fr', self.args.data_name)
        mr_path = os.path.join(self.args.data_path, 'input', 'mr', self.args.data_name)
        pr_path = os.path.join(self.args.data_path, 'input', 'pr', self.args.data_name)
        # read file
        self.fr_result = pd.read_csv(fr_path, header=None)
        self.mr_result = pd.read_csv(mr_path, header=None)
        self.pr_result = pd.read_csv(pr_path, header=None)

        self.logic()

    def _find_grp_columns(self, df):
        for column in df.columns:
            if df[column].unique().tolist()==['BK','LN','DN']:
                return column

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='inference-postprocess')

    parser.add_argument('--data_path', default='/opt/ml/processing')
    parser.add_argument('--data_name', default='pnr.csv.out')

    args = parser.parse_args()
    postp = Postprocess(args)
    postp.execution()