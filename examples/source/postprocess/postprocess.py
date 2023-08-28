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

        # save
        self.result.to_csv(os.path.join(self.args.strDataPath, 'output', 'pnr.csv'), index=False)

    def execution(self):
        # define file path
        fr_path = os.path.join(self.args.strDataPath, 'input', 'fr', self.args.strDataName)
        mr_path = os.path.join(self.args.strDataPath, 'input', 'mr', self.args.strDataName)
        pr_path = os.path.join(self.args.strDataPath, 'input', 'pr', self.args.strDataName)
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

    parser.add_argument('--strDataPath', default='/opt/ml/processing')
    parser.add_argument('--strDataName', default='pnr.csv.out')

    args = parser.parse_args()
    postp = Postprocess(args)
    postp.execution()