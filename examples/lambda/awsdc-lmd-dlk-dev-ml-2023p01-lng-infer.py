import boto3
import os

from datetime import datetime, timezone, timedelta

def _get_approved_latest_model_package_arn(model_package_group_name, sagemaker_client):
    '''
    get latest model package arn within approved status models
    '''
    list_model_packages = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name
    )['ModelPackageSummaryList']
    list_model_packages.sort(
        key = lambda model_package : model_package['ModelPackageVersion'], 
        reverse=True
    )
    for package in list_model_packages:
        if package['ModelApprovalStatus']=='Approved':
            return package['ModelPackageArn']
        continue

def _get_approved_latest_model_name(model_package_group_name, sagemaker_client):
    model_arn = _get_approved_latest_model_package_arn(model_package_group_name, sagemaker_client)
    model_name = sagemaker_client.describe_model_package(
        ModelPackageName=model_arn
    )["ModelPackageDescription"]
    
    return model_name


def lambda_handler(event, context):
	print('triggering event, below')
	print(event)
	print('=' * 50)

	client = boto3.client('sagemaker')
	
	fr_model_package_group_name = os.environ['FR_MODEL_PACKAGE_GROUP_NAME'] # sag-dev-ml-lng-fr-model-group
	mr_model_package_group_name = os.environ['MR_MODEL_PACKAGE_GROUP_NAME'] # sag-dev-ml-lng-mr-model-group
	pr_model_package_group_name = os.environ['PR_MODEL_PACKAGE_GROUP_NAME'] # sag-dev-ml-lng-pr-model-group
	
	# Pipeline Parameters
	today = datetime.strftime(
        datetime.now(tz=timezone(timedelta(hours=9))), '%Y%m%d'
    )    # 20230920
	fr_model_name = _get_approved_latest_model_name(fr_model_package_group_name, client)
	mr_model_name = _get_approved_latest_model_name(mr_model_package_group_name, client)
	pr_model_name = _get_approved_latest_model_name(pr_model_package_group_name, client)
	
	res = client.start_pipeline_execution(
		PipelineName = os.environ['PIPELINE_NAME'],
		PipelineParameters = [
			{
				'Name' : 'today',
				'Value' : today,
			},
			{
				'Name' : 'fr_model_name',
				'Value' : fr_model_name,
			},
			{
				'Name' : 'mr_model_name',
				'Value' : mr_model_name,
			},
			{
				'Name' : 'pr_model_name',
				'Value' : pr_model_name,
			}
		]
	)
	print("pipeline execution details")
	print(res)
	print("=" * 50)

	return 

