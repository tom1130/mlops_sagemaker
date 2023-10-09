import os
import json
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error


class training_evaluator:

    def __init__(self):
        self.metrics = dict(
            Metrics=dict()
        )
        pass

    def load_model(self):
        pass

    def evaluation_logic(self):
        
        prefix = '/opt/ml/processing/input'

        predictions_path = os.path.join(prefix, 'predictions', 'pnr.csv.out')
        predictions = pd.read_csv(predictions_path, header=None)
        labels_path = os.path.join(prefix, 'test_data', 'lounge.csv')
        labels = pd.read_csv(labels_path, header=None)

        metric_r2 = r2_score(labels, predictions)
        metric_mae = mean_absolute_error(labels, predictions)

        self.metrics["Metrics"].update(
            {
                'TEST_R2':metric_r2,
                'TEST_MAE':metric_mae
            }
        )

        eval_path = "/opt/ml/processing/evaluation"
        file_path = os.path.join(eval_path, "evaluation_metrics.json")

        with open(file_path, 'w') as f:
            json.dump(self.metrics, f)



    def execution(self):
        
        self.evaluation_logic()


if __name__=='__main__':

    Evaluator = training_evaluator()
    Evaluator.execution()