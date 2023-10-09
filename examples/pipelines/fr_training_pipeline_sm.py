import os
import ast
import argparse
from datetime import datetime
from config.config import config_handler
from pprint import pprint
from pytz import timezone

from sagemaker import ModelMetrics, MetricsSource
from sagemaker import AutoML
from sagemaker.automl.automl import AutoMLInput
from sagemaker.sklearn import SKLearn
from sagemaker.transformer import Transformer
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.steps import ProcessingStep, TransformStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.functions import Join
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.workflow.parameters import ParameterString

from utils.secret_manager import get_secret


class FrTrainingPipelineSM():
    '''
    SageMaker SDK 활용하여, Step 정의 및 pipeline 생성
    '''
    def __init__(self, args):
        
        self.args = args
        self.secret = get_secret()
        
        self._env_setting()
        
        

    def _env_setting(self, ):
        
        self.pipeline_session = PipelineSession()
        self.execution_role = self.args.config.get_value("COMMON", "role")
        self.pipeline_name = self.args.config.get_value("COMMON", "pipeline_name") 

        self.today = ParameterString(
            name="today",
            default_value=datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d"),
        )

        self.now = ParameterString(
            name="now",
            default_value=datetime.now(timezone("Asia/Seoul")).strftime("%H%M%S")
        )

        self.tags = ast.literal_eval(self.args.config.get_value("COMMON", "tags"))
        self.lounge_name = self.args.config.get_value("COMMON", "lounge_name")
        self.data_bucket = self.args.config.get_value('COMMON','data_bucket')
        self.code_bucket = self.args.config.get_value('COMMON','code_bucket')
        self.prefix = self.args.config.get_value('COMMON','prefix')

        self.data_base_path = os.path.join(
            's3://',
            self.data_bucket,
            self.prefix
        )

        self.code_base_path = os.path.join(
            's3://',
            self.code_bucket,
            self.prefix
        )
        
        self.git_config = {
            'repo' : self.args.config.get_value('GIT','git_repo'),
            'branch' : self.args.config.get_value('GIT','git_branch'),
            'username' : self.secret['USER'].split('@')[0],
            'password' : self.secret['PASSWORD'],
        }
        
        
        
    def _step_preprocessing(self, ):
        
        prefix_prep = '/opt/ml/processing'
        data_path = os.path.join(
            self.data_base_path, 
            self.args.config.get_value('PREPROCESSING','data_path')
        )
        target_path = os.path.join(
            self.data_base_path, 
            self.args.config.get_value('PREPROCESSING','target_path')
        )
        etc_path = os.path.join(
            self.data_base_path, 
            self.args.config.get_value('PREPROCESSING','etc_path')
        )
        
        prep_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("PREPROCESSING", "framework_version"),
            role=self.execution_role,
            instance_type=self.args.config.get_value("PREPROCESSING", "instance_type"),
            instance_count=self.args.config.get_value("PREPROCESSING", "instance_count", dtype='int'),
            sagemaker_session=self.pipeline_session,
            tags=self.tags
        )
        
        step_preprocessing_args = prep_processor.run(
            code = './fr_training_preprocess.py',
            source_dir = './source/preprocess',
            git_config=self.git_config,
            inputs = [
                ProcessingInput(
                    input_name='input',
                    source=data_path,
                    destination=os.path.join(prefix_prep, 'input')
                ),
                ProcessingInput(
                    input_name='etc',
                    source=etc_path,
                    destination=os.path.join(prefix_prep, 'etc')
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train-data",
                    source=os.path.join(prefix_prep, "output", "train"),
                    destination=os.path.join(target_path,'fr','train-data'),
                ),
                ProcessingOutput(
                    output_name="validation-data",
                    source=os.path.join(prefix_prep, "output", "validation"),
                    destination=os.path.join(target_path,'fr','validation-data'),
                ),
                ProcessingOutput(
                    output_name="test-data",
                    source=os.path.join(prefix_prep, "output", "test"),
                    destination=os.path.join(target_path,'fr','test-data'),
                )
            ],
            arguments=[
                '--region', self.args.config.get_value("COMMON", "region"),
                '--sns_arn', self.args.config.get_value("SNS", "arn"),
                '--project_name', self.args.config.get_value("COMMON", "project_name"),
                '--pipeline_name', self.args.config.get_value("COMMON", "pipeline_name"),
            ]
        )
        
        self.preprocessing_process = ProcessingStep(
            name = "FrTrainPreprocessingProcess",
            step_args = step_preprocessing_args,
        )

        ## logging ##########
        print("  \n== Preprocessing Step ==")
        print("   \nArgs: ")
        for key, value in self.preprocessing_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.preprocessing_process.properties))



    def _step_training(self, ):
        
        target_attribute = self.args.config.get_value("TRAINING", "target_attribute")
        model_output_path = Join(
            on='/',
            values=[
                self.code_base_path,
                self.today,
                "model"
            ]
        )
        
        paramAutoMLTrain = AutoMLInput(
            channel_type = "training",
            content_type = "text/csv;header=present",
            compression = None,
            inputs = self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["train-data"].S3Output.S3Uri,
            target_attribute_name = target_attribute,
            s3_data_type = "S3Prefix",
        )

        paramAutoMLValid = AutoMLInput(
            channel_type = "validation",
            content_type = "text/csv;header=present",
            compression = None,
            inputs = self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["validation-data"].S3Output.S3Uri,
            target_attribute_name = target_attribute,
            s3_data_type = "S3Prefix",
        )
        
        self.auto_ml_estimator = AutoML(
            role = self.execution_role,
            target_attribute_name = self.args.config.get_value("TRAINING", "target_attribute"),
            output_path = model_output_path,
            problem_type = "Regression",
            max_candidates = eval(self.args.config.get_value("TRAINING", "max_candidates")), 
            max_runtime_per_training_job_in_seconds = eval(self.args.config.get_value("TRAINING", "max_runtime_per_training_job")),
            total_job_runtime_in_seconds = eval(self.args.config.get_value("TRAINING", "max_runtime_for_auto_ml_job")),
            job_objective = {
                'MetricName': self.args.config.get_value("TRAINING", "job_objective")
            },
            generate_candidate_definitions_only = False,
            tags = self.tags,
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
        
        ## logging ##########
        print("  \n== AutoML Step ==")
        print("   \nArgs: ")
        for key, value in self.training_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.training_process.properties))
        
        
    
    def _step_model_creation(self, ):
        
        self.best_auto_ml_model = self.training_process.get_best_auto_ml_model(
            role=self.execution_role,
            sagemaker_session=self.pipeline_session,
        )
        
        model_name = "sag-dev-ml-lng-fr-model-smsdk"
        self.best_auto_ml_model.name = model_name
        
        step_model_creation_args = self.best_auto_ml_model.create(
            instance_type = self.args.config.get_value("MODEL_CREATION", "instance_type"),
            tags=self.tags
        )
        
        self.model_creation_process = ModelStep(
            name = "ModelCreationProcess", 
            step_args = step_model_creation_args,
        )
        
        ## logging ##########
        print("  \n== ModelCreate Step Completed==")

    

    def _step_batch_transform(self, ):

        eval_output_path = Join(
            on='/',
            values=[
                self.code_base_path,
                "evaluation",
                self.lounge_name,
                self.today
            ]
        )

        self.transformer = Transformer(
            model_name = self.model_creation_process.properties.ModelName,
            instance_type = self.args.config.get_value("BATCH_TRANSFORM", "instance_type"),
            instance_count = self.args.config.get_value("BATCH_TRANSFORM", "instance_count", dtype="int"),
            output_path = eval_output_path,
            assemble_with="Line",
            accept="text/csv",
            sagemaker_session = self.pipeline_session,
        )
        
        step_batch_transform_args = self.transformer.transform(
            data = Join(
                on='/',
                values=[
                    self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test-data"].S3Output.S3Uri,
                    'pnr.csv'
                ]
            ),     
            content_type = "text/csv",
            split_type="Line",
        )
         
        self.batch_transform_process = TransformStep(
            name = "BatchTransformProcess",
            step_args = step_batch_transform_args,
        )

        ## logging ##########
        print("  \n== Batch Transform Step ==")
        print("   \nArgs: ")
        for key, value in self.batch_transform_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.batch_transform_process.properties))
        
        
    def _step_evaluation(self, ):

        eval_output_path = Join(
            on='/',
            values=[
                self.data_base_path,
                "train/evaluation",
                self.lounge_name,
                self.today
            ]
        )

        self.eval_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value("EVALUATION", "framework_version"),
            role=self.execution_role,
            instance_type=self.args.config.get_value("EVALUATION", "instance_type"),
            instance_count=self.args.config.get_value("EVALUATION", "instance_count", dtype="int"),
            sagemaker_session=self.pipeline_session,
        )
        
        step_evaluation_args = self.eval_processor.run(
            code="./training_evaluation.py",
            source_dir="./source/evaluation",
            git_config=self.git_config,
            inputs=[
                ProcessingInput(
                    input_name="predictions",
                    source=self.batch_transform_process.properties.TransformOutput.S3OutputPath,
                    destination="/opt/ml/processing/input/predictions",
                ),
                ProcessingInput(
                    input_name="test-data-evaluation",
                    source=Join(
                        on='/',
                        values=[
                            self.preprocessing_process.properties.ProcessingOutputConfig.Outputs["test-data"].S3Output.S3Uri,
                            'lounge.csv'
                        ]
                    ), 
                    destination="/opt/ml/processing/input/test_data",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation-metrics",
                    source="/opt/ml/processing/evaluation",
                    destination=eval_output_path,
                ),
            ],
        )
        
        self.evaluation_process = ProcessingStep(
            name="ModelEvaluationProcess",
            step_args=step_evaluation_args,
        )
        
        ## logging ##########
        print("  \n== Evaluation Step ==")
        print("   \nArgs: ")
        for key, value in self.evaluation_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.evaluation_process.properties))
        
    
    def _step_model_registration(self, ):
        
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=self.evaluation_process.properties.ProcessingOutputConfig.Outputs["evaluation-metrics"].S3Output.S3Uri,
                content_type="application/json", 
            )
        )

        model_approval_status = "PendingManualApproval"

        step_model_registration_args = self.best_auto_ml_model.register(
            content_types=[self.args.config.get_value("REGISTER", "registry_content_type")],
            response_types=[self.args.config.get_value("REGISTER", "registry_response_type")],
            inference_instances=[self.args.config.get_value("REGISTER", "registry_realtime_instance_type")],
            transform_instances=[self.args.config.get_value("REGISTER", "registry_transform_instance_type")],
            model_package_group_name=self.args.config.get_value("COMMON", "model_package_group_name"),
            approval_status=model_approval_status,
            model_metrics=model_metrics,
            description=self.best_auto_ml_model.name,
        )
        
        self.model_registration_process = ModelStep(
            name="ModelRegistrationProcess", 
            step_args=step_model_registration_args,
        )
        
        ## logging ##########
        print("  \n== ModelRegistration Step Completed==")


        
    def _get_pipeline(self, ):
                
        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[
                self.preprocessing_process, 
                self.training_process, 
                self.model_creation_process,
                self.batch_transform_process,
                self.evaluation_process, 
                self.model_registration_process,
            ],
            sagemaker_session=self.pipeline_session,
            parameters=[self.today, self.now],
            pipeline_definition_config=PipelineDefinitionConfig(use_custom_job_prefix=True), # 자동으로 AutoMLJobName, ModelName Override 되는 것 방지
        )
    
        return pipeline
    
        
    def execution(self, ):
        
        self._step_preprocessing()
        self._step_training()
        self._step_model_creation()
        self._step_batch_transform()
        self._step_evaluation()
        self._step_model_registration()
        
        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.execution_role)
        execution = pipeline.start()
        desc = execution.describe()
        
        print(desc)
    
    
if __name__=="__main__":
    
    strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    
    today = datetime.strftime(datetime.now(timezone('Asia/Seoul')), '%Y%m%d')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--today', default=today)
    args, _ = parser.parse_known_args()

    config_file = "dev_fr_train_sm_config.ini"
    # config_file = "prd_fr_train_sm_config.ini"
    args.config = config_handler(config_file)
    
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.config.get_value("COMMON", "region")
    
    pipe = FrTrainingPipelineSM(args)
    pipe.execution()