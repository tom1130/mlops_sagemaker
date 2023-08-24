import os
import argparse
import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

class monitoring:
    
    def __init__(self, args):
        self.args = args

        self.today_date = ''
        
    def logic(self):
        pass

    def execution(self):
        pass

    def _read_predictions(self):
        pass

    def _read_label(self):
        pass

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='monitoring')
    # fr
    parser.add_argument('--strFrLoungePath', default='')
    # mr
    parser.add_argument('--strMrLoungePath', default='')
    # pr
    parser.add_argument('--strPrLoungePath', default='')