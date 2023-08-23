import os
import argparse

from preprocess import Preprocess



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
        fr_path = os.path.join(self.args.strDataPath, 'output','fr','lounge.csv')
        mr_path = os.path.join(self.args.strDataPath, 'output','mr','lounge.csv')
        pr_path = os.path.join(self.args.strDataPath, 'output', 'pr','lounge.csv')
        fr.to_csv(fr_path, index=False)
        mr.to_csv(mr_path, index=False)
        pr.to_csv(pr_path, index=False)

    def execution(self):
        df = self._read_data()

        self.logic(df)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='inference_preprocessing')

    AWS_yn = False

    if AWS_yn:
        parser.add_argument('--strDataPath', default='/opt/ml/processing')
        parser.add_argument('--strDataName', default='pnr_agg_data_20230815.csv')
        parser.add_argument('--strLabelName', default='lng_agg_data.csv')
        parser.add_argument('--strHoliday', default='holiday.csv')
        parser.add_argument('--listYears', type=list, default=[2021,2022,2023])
    
    else:
        parser.add_argument('--strDataPath', default='c://Users/고기호/Desktop/vscode/mlops/examples/data/raw')
        parser.add_argument('--strDataName', default='pnr_agg_data_20230815.csv')
        parser.add_argument('--strLabelName', default='lng_agg_data.csv')
        parser.add_argument('--listYears', type=list, default=[2021,2022,2023])
    
    args = parser.parse_args()
    prep = inference_preprocess(args)
    prep.execution()