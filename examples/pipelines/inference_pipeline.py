import os
import argparse
from pprint import pprint
from config.config import config_handler

import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TransformStep
from sagemaker.processing import ProcessingInput, ProcessingOutput, FrameworkProcessor
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model import ModelPackage

class pipeline_inference:

    def __init__(self, args):
        self.args = args

        self._env_setting()
        self.sagemaker_client = boto3.client('sagemaker')

        # get mr model arn
        mr_model_grp_nm = self.args.config.get_value('MR-INFERENCING','model_package_group_name')
        self.mr_model_arn = self._get_approved_latest_model_package_arn(mr_model_grp_nm)
        # get pr model arn
        pr_model_grp_nm = self.args.config.get_value('PR-INFERENCING','model_package_group_name')
        self.pr_model_arn = self._get_approved_latest_model_package_arn(pr_model_grp_nm)

    def _env_setting(self):

        self.strExcutionRole = self.args.config.get_value("COMMON", "role")
        self.strBucketName = self.args.config.get_value("COMMON", "bucket")
        self.strPipelineName = self.args.config.get_value("COMMON","pipeline_name")

        self.cache_config = CacheConfig(
            enable_caching=self.args.config.get_value("PIPELINE", "enable_caching", dtype="boolean"),
            expire_after=self.args.config.get_value("PIPELINE", "expire_after")
        )

        self.pipeline_session = PipelineSession()

    def _get_approved_latest_model_package_arn(self, model_package_group_name):
        '''
        get latest model package arn within approved status models
        '''
        list_model_packages = self.sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name
        )['ModelPackageSummaryList']
        list_model_packages.sort(
            key=lambda model_package:model_package['ModelPackageVersion'], 
            reverse=True
        )
        list_model_packages = [model for model in list_model_packages if model['ModelApprovalStatus']=='Approved']
        
        latest_model_arn = list_model_packages[0]['ModelPackageArn']

        return latest_model_arn

    def _step_preprocess(self):

        pipeline_session = PipelineSession()
        
        strPrefixPrep = '/opt/ml/processing'
        strDataPath = self.args.config.get_value('PREPROCESSING', 'data_path')
        strTargetPath = self.args.config.get_value('PREPROCESSING', 'target_path')

        prep_processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=self.args.config.get_value('PREPROCESSING','framework_version'),
            role=self.strExcutionRole,
            instance_type=self.args.config.get_value('PREPROCESSING','instance_type'),
            instance_count=self.args.config.get_value('PREPROCESSING','instance_count', dtype='int'),
            sagemaker_session=pipeline_session
        )

        step_args = prep_processor.run(
            code='./inference_preprocess.py',
            source_dir='../source/preprocess/',
            # git_config=self.git_config,
            inputs=[
                ProcessingInput(
                    input_name='today-input',
                    source=os.path.join(strDataPath, self.args.today),
                    destination=os.path.join(strPrefixPrep, self.args.today, 'input')
                ),
                ProcessingInput(
                    input_name='yesterday-input',
                    source=os.path.join(strDataPath, self.args.yesterday),
                    destination=os.path.join(strPrefixPrep, self.args.yesterday, 'input')
                ),
                ProcessingInput(
                    input_name='holiday-input',
                    source=self.args.config.get_value('PREPROCESSING','etc_path'),
                    destination=os.path.join(strPrefixPrep, 'etc')
                ),
                ProcessingInput(
                    input_name='integrate-input',
                    source=self.args.config.get_value('PREPROCESSING','integrate_data_path'),
                    destination=os.path.join(strPrefixPrep, 'integrate')
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name='mr-inference-data',
                    source=os.path.join(strPrefixPrep, 'output', 'mr'),
                    destination=os.path.join(strTargetPath, self.args.today, 'mr')
                ),
                ProcessingOutput(
                    output_name='fr-inference-data',
                    source=os.path.join(strPrefixPrep, 'output', 'fr'),
                    destination=os.path.join(strTargetPath, self.args.today, 'fr')
                ),
                ProcessingOutput(
                    output_name='pr-inference-data',
                    source=os.path.join(strPrefixPrep, 'output', 'pr'),
                    destination=os.path.join(strTargetPath, self.args.today, 'pr')
                ),
                ProcessingOutput(
                    output_name='integrated-data',
                    source=os.path.join(strPrefixPrep, 'output', 'integrate'),
                    destination=self.args.config.get_value('PREPROCESSING','integrate_data_path')
                )
            ],
            # arguments=[],
            job_name='inference_preprocessing'
        )

        self.preprocessing_process = ProcessingStep(
            name='InferencePreprocessingProcess',
            step_args=step_args,
            cache_config=self.cache_config
        )

        ## logging ##########
        print("  \n== Preprocessing Step ==")
        print("   \nArgs: ")
        for key, value in self.preprocessing_process.arguments.items():
            print("===========================")
            print(f'key: {key}')
            pprint(value)
            
        print (type(self.preprocessing_process.properties))

    def _step_fr_inference(self):

        fr_model_grp_nm = self.args.config.get_value('FR-INFERENCING','model_package_group_name')
        fr_model_arn = self._get_approved_latest_model_package_arn(fr_model_grp_nm)
        
        model = ModelPackage(
            model_package_arn=fr_model_arn,
            role=self.strExcutionRole
        )

        transformer = model.transformer(
            instance_type=self.args.config.get_value('FR-INFERENCING', 'instance_type'),
            instance_count=self.args.config.get_value('FR-INFERENCING', 'instance_count'),
            output_path=self.args.config.get_value('FR-INFERENCING', 'target_path'),
            assemble_with='Line',
            accept='text/csv'
        )

        step_args = transformer.transform(
            data='',
            content_type='text/csv',
            split_type='Line'
        )

        self.fr_inference_process = TransformStep(
            name='FrInferenceProcess',
            step_args=step_args
        )


        
    def _step_mr_inference(self):
        pass

    def _step_pr_inference(self):
        pass

    def _step_postprocess(self):
        pass

    def _get_pipeline(self):
        
        pipeline = Pipeline(
            name=self.strPipelineName,
            steps=[self.preprocessing_process],
            sagemaker_session=self.pipeline_session
        )
        return pipeline

    def execution(self):
        
        self._step_preprocess()

        pipeline = self._get_pipeline()
        pipeline.upsert(role_arn=self.strExcutionRole)
        execution = pipeline.start()

        print(execution.describe)

if __name__=='__main__':
    strBasePath, strCurrentDir = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    os.chdir(strBasePath)
    # get config and argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--today', default='20230725')
    parser.add_argument('--yesterday', default='20230724')
    args, _ = parser.parse_known_args()
    args.config = config_handler('inference_config.ini')

    # execute monitoring pipeline
    pipe_monitor = pipeline_inference(args)
    pipe_monitor.execution()