import argparse

from preprocess import Preprocess



class pr_training_preprocess(Preprocess):

    def __init__(self, args):
        super().__init__(args)

    def logic(self, df, label):
        
        df = self._transform_base_data(df)
        df = self._drop_useless_date(df)
        df = self._make_pr_features(df)
        df = self._get_pr_dataset(df)

        df = self._merge_data_label(df, label)

        return df

    def execution(self):
        df = self._read_data()
        label = self._read_label()

        return self.logic(df, label)
    
    def _merge_data_label(self, df, label):
        df_cols = df.columns.tolist()

        df = df.merge(label, on=['std','group'], how='inner')
        df = df[df_cols+['PR_LNG']]
        df = df.rename(columns={'PR_LNG' : 'target'})

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='preprocessing')

    parser.add_argument('--strDataPath', default='c:\\Users\\고기호\\Desktop\\vscode\\mlops\\examples\\data')
    parser.add_argument('--strDataName', default='pnr_agg_data_20230815.csv')
    parser.add_argument('--strLabelName', default='lng_agg_data.csv')
    parser.add_argument('--listYears', type=list, default=[2021,2022,2023])

    args = parser.parse_args()

    prep = pr_training_preprocess(args)
    a = prep.execution()
    print(a)