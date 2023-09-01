import os
import boto3
import botocore
import time, datetime
import json
import sagemaker
import pandas as pd
import argparse
from config.config import config_handler
from pprint import pprint
from pytz import timezone

from sagemaker import get_execution_role
from sagemaker import ModelMetrics, MetricsSource
from sagemaker import AutoML
from sagemaker.automl.automl import AutoMLInput
from sagemaker.sklearn import SKLearn
from sagemaker.transformer import Transformer
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor


class pipeline_fr():
    '''
    pipeline_fr과 달라진 점.
    1. Preprocessing 단계를 건너뜀(따라서 Train Step에서의 input은 Preprocessing.Properties를 사용 X)
    2. tmp/lounge_2 데이터를 활용할 거라서 path도 그에 따라서 바꿈. 아래는 바뀐 path목록
        - 
        - 
        
    '''
    def __init__(self, args):
        
        self.args = args
        self.strRegionName = self.args.config.get_value("COMMON", "region")
        
        self.exp = True
        self._env_setting()
        
        
    def _get_datetime(self):
        now = datetime.datetime.now(timezone('Asia/Seoul'))
        date = now.strftime("%Y%m%d")
        date_time = now.strftime("%Y%m%d-%H%M%S")
        return date, date_time       
        
        
    def _env_setting(self, ):
        
        self.pipeline_session = PipelineSession()
        self.strExecutionRole = self.args.config.get_value("COMMON", "role")
        self.strLoungeName = self.args.config.get_value("COMMON", "lounge_name")
        self.strModelPackageGroupName = self.args.config.get_value("COMMON", "model_package_group_name")
        self.strDataBucketName = self.args.config.get_value("COMMON", "data_bucket")
        self.strCodeBucketName = self.args.config.get_value("COMMON", "code_bucket")
        self.strPipelineName = self.args.config.get_value("COMMON", "pipeline_name") #"-".join([self.strPrefix, self.strModelName])
        
        _datetime = self._get_datetime()
        self.today = _datetime[0]
        # self.today_time = _datetime[1]
        
        
    def _step_preprocessing(self, ):
        
        strPrefixPrep = '/opt/ml/processing'
        strDataPath = self.args.config.get_value('PREPROCESSING','data_path')
        strTargetPath = self.args.config.get_value('PREPROCESSING','target_path')
        
        prep_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("PREPROCESSING", "framework_version"),
            role=self.strExecutionRole,
            instance_type=self.args.config.get_value("PREPROCESSING", "instance_type"),
            instance_count=self.args.config.get_value("PREPROCESSING", "instance_count"),
            sagemaker_session=self.pipeline_session,
        )
        
        step_preprocessing_args = prep_processor.run(
            code = './fr_training_preprocess.py',
            source_dir = '../source/preprocess',
            inputs = [
                ProcessingInput(
                    input_name='input',
                    source=strDataPath,
                    destination=os.path.join(strPrefixPrep, 'input')
                ),
                ProcessingInput(
                    input_name='etc',
                    source=self.args.config.get_value('PREPROCESSING','etc_path'),
                    destination=os.path.join(strPrefixPrep, 'etc')
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train-data",
                    source=os.path.join(strPrefixPrep, "output", "train"),
                    destination=os.path.join(strTargetPath,'fr','train-data'),
                ),
                ProcessingOutput(
                    output_name="validation-data",
                    source=os.path.join(strPrefixPrep, "output", "validation"),
                    destination=os.path.join(strTargetPath,'fr','validation-data'),
                ),
                ProcessingOutput(
                    output_name="test-data",
                    source=os.path.join(strPrefixPrep, "output", "test"),
                    destination=os.path.join(strTargetPath,'fr','test-data'),
                )
            ]
        )
        
        self.preprocessing_process = ProcessingStep(
            name = "FrTrainPreprocessingProcess",
            step_args = step_preprocessing_args,
            # cache_config = self.cache_config
        )

        ## logging ##########
        print("  \n== Preprocessing Step ==")
        print("   \nArgs: ")
        for key, value in self.preprocessing_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.preprocessing_process.properties))
        
        
    '''
    boto3로 모델을 생성할 경우, Sagemaker AutoML class를 사용하는 training step 및 model creation step이 수정 필요
    '''
    def _step_training(self, ):
        
        target_attribute_name = self.args.config.get_value("TRAINING", "target_attribute_name")
        
        paramAutoMLTrain = AutoMLInput(
            channel_type = "training",
            content_type = "text/csv;header=present",
            compression = None,
            inputs = os.path.join("s3://", self.strDataBucketName, "train/preprocess/fr/train-data/pnr.csv"), # self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["train-data"].S3Output.S3Uri,
            target_attribute_name = target_attribute_name,
            s3_data_type = "S3Prefix",
        )

        paramAutoMLValid = AutoMLInput(
            channel_type = "validation",
            content_type = "text/csv;header=present",
            compression = None,
            inputs = os.path.join("s3://", self.strDataBucketName, "train/preprocess/fr/validation-data/pnr.csv"), # self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["validation-data"].S3Output.S3Uri,
            target_attribute_name = target_attribute_name,
            s3_data_type = "S3Prefix",
        )
        
        tags = [
            {
                'Key': 'Billing',
                'Value': 'KEKAL'
            },
            {
                'Key': 'Department',
                'Value': 'KEXWD'
            },
        ]
        
        
        self.auto_ml_estimator = AutoML(
            role = self.strExecutionRole,
            target_attribute_name = self.args.config.get_value("TRAINING", "target_attribute_name"),
            output_path = os.path.join("s3://", self.strCodeBucketName, "model"),
            problem_type = self.args.config.get_value("TRAINING", "problem_type"),
            max_candidates = eval(self.args.config.get_value("TRAINING", "max_candidates")), 
            max_runtime_per_training_job_in_seconds = eval(self.args.config.get_value("TRAINING", "max_runtime_per_training_job_in_seconds")),
            total_job_runtime_in_seconds = eval(self.args.config.get_value("TRAINING", "max_auto_ml_job_runtime_in_seconds")),
            job_objective = {
                'MetricName': self.args.config.get_value("TRAINING", "job_objective")
            },
            generate_candidate_definitions_only = False,
            tags = tags,
            feature_specification_s3_uri = self.args.config.get_value("TRAINING", "feature_specification_s3_uri"),
            mode = self.args.config.get_value("TRAINING", "mode"),
            sagemaker_session = self.pipeline_session,
        )
        
        step_training_args = self.auto_ml_estimator.fit(
            inputs = [
                paramAutoMLTrain,
                paramAutoMLValid,
            ],
            wait = True
        )
        
        self.training_process = AutoMLStep(
            name = "AutoMLTrainingProcess",
            step_args = step_training_args,
        )
        
        print('## Training Step Created')
        
        
    
    def _step_model_creation(self, ):
        
        self.best_auto_ml_model = self.training_process.get_best_auto_ml_model(
            role=self.strExecutionRole,
            sagemaker_session=self.pipeline_session,
        )
        
        self.best_auto_ml_model.name = self.args.config.get_value("MODEL_CREATION", "model_name")
        
        step_model_creation_args = self.best_auto_ml_model.create(
            instance_type = self.args.config.get_value("MODEL_CREATION", "instance_type"),
        )
        
        self.model_creation_process = ModelStep(
            name = "ModelCreationProcess", 
            step_args = step_model_creation_args,
        )
        
        print('## Model Creation Step Created')

    
    
    def _step_batch_transform(self, ):
        # 1. Test Data 생성 시 Features/Target 나눠서 적재
        # 2. Test Data 생성 시 index, header False
        # 3. Batch Transform시 뭐가 무슨 컬럼인지 어떻게 인식을..?

        self.transformer = Transformer(
            model_name = self.model_creation_process.properties.ModelName,
            instance_type = self.args.config.get_value("BATCH_TRANSFORM", "instance_type"),
            instance_count = self.args.config.get_value("BATCH_TRANSFORM", "instance_count", dtype="int"),
            output_path = os.path.join("s3://", self.strDataBucketName, f"train/evaluation/{self.strLoungeName}/{self.today}"),
            assemble_with="Line",
            accept="text/csv",
            sagemaker_session = self.pipeline_session,
        )
        
        step_batch_transform_args = self.transformer.transform(
            data = "s3://awsdc-s3-dlk-dev-ml/tmp/lounge_2/train/preprocess/fr/test-data/pnr_drop_target.csv",
            content_type = "text/csv",
            split_type="Line",
        )
        
        self.batch_transform_process = TransformStep(
            name = "BatchTransformProcess",
            step_args = step_batch_transform_args,
        )

        print('## Batch Transform Step Created')
        
        
    def _step_evaluation(self, ):

            
        self.eval_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("EVALUATION", "framework_version"),
            role=self.strExecutionRole,
            instance_type=self.args.config.get_value("EVALUATION", "instance_type"),
            instance_count=self.args.config.get_value("EVALUATION", "instance_count", dtype="int"),
            sagemaker_session=self.pipeline_session,
        )
        
        step_evaluation_args = self.eval_processor.run(
            code="training_evaluation.py",
            inputs=[
                ProcessingInput(
                    input_name="predictions",
                    source=self.batch_transform_process.properties.TransformOutput.S3OutputPath,
                    destination="/opt/ml/processing/input/predictions",
                ),
                ProcessingInput(
                    input_name="test-data-evaluation",
                    source="s3://awsdc-s3-dlk-dev-ml/tmp/lounge_2/train/preprocess/fr/test-data/pnr_target.csv", 
                    destination="/opt/ml/processing/input/test_data",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation-metrics",
                    source="/opt/ml/processing/evaluation",
                    destination=os.path.join("s3://", self.strDataBucketName, f"train/evaluation/{self.strLoungeName}/{self.today}"),
                ),
            ],
        )
        
        self.evaluation_process = ProcessingStep(
            name="ModelEvaluationProcess",
            step_args=step_evaluation_args,
        )
        
        print('## Evaluation Step Created')
        
    
    def _step_model_registration(self, ):
        
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=self.evaluation_process.properties.ProcessingOutputConfig.Outputs["evaluation-metrics"].S3Output.S3Uri,
                # os.path.join(
                #     self.evaluation_process.properties.ProcessingOutputConfig.Outputs["evaluation-metrics"].S3Output.S3Uri,
                #     "evaluation_metrics.json",
                # ),
                content_type="application/json", 
            )
        )

        model_approval_status = "PendingManualApproval"

        step_model_registration_args = self.best_auto_ml_model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=self.strModelPackageGroupName,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
        )
        
        self.model_registration_process = ModelStep(
            name="ModelRegistrationProcess", 
            step_args=step_model_registration_args,
        )
        

        
    def _get_pipeline(self, ):
                
        pipeline = Pipeline(
            name=self.strPipelineName,
            steps=[
                # self.preprocessing_process, 
                self.training_process, 
                self.model_creation_process,
                self.batch_transform_process,
                self.evaluation_process, 
                self.model_registration_process,
            ],
            sagemaker_session=self.pipeline_session
        )
    
        return pipeline
    
        
    def execution(self, ):
        
        # self._step_preprocessing()
        self._step_training()
        self._step_model_creation()
        self._step_batch_transform()
        self._step_evaluation()
        self._step_model_registration()
        
        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.strExecutionRole)
        execution = pipeline.start()
        desc = execution.describe()
        
        print(desc)
    
    
if __name__=="__main__":
    
    strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    args.config = config_handler('config_exp.ini')
    
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.config.get_value("COMMON", "region")
    
    pipe = pipeline_fr(args)
    pipe.execution()