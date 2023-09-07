import boto3
import argparse

from sagemaker.model import ModelPackage

class Transformer:

    def __init__(self, args):
        self.args = args

    def logic(self):
        latest_model_arn = self._get_approved_latest_model_package_arn()

        model = ModelPackage(
            model_package_arn=latest_model_arn,
            role=self.args.strExecutionRole
        )

        transformer = model.transformer(
            instance_type=self.args.instance_type,
            instance_count=self.args.instance_count,
            output_path=self.args.strOutputPath,
            assemble_with='Line',
            accept='text/csv'
        )

        transformer.transform(
            data=self.args.strInputPath,
            content_type='text/csv',
            split_type='Line',
            join_source='input'
        )

    def execution(self):
        self.logic()

    def _get_approved_latest_model_package_arn(self,):
        '''
        get latest model package arn within approved status models
        '''
        list_model_packages = self.sagemaker_client.list_model_packages(
            ModelPackageGroupName=self.model_package_group_name
        )['ModelPackageSummaryList']
        list_model_packages.sort(
            key=lambda model_package:model_package['ModelPackageVersion'], 
            reverse=True
        )
        for package in list_model_packages:
            try:
                package['ModelApprovalStatus']=='Approved'
                return package['ModelPackageArn']
            except:
                continue


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--strExecutionRole', default='')
    parser.add_argument('--instance_type', default='')
    parser.add_argument('--instance_count', default='')
    parser.add_argument('--strOutputPath', default='')
    parser.add_argument('--strInputPath', default='')

    args, _ = parser.parse_known_args()

    batch_transform = Transformer(args)
    batch_transform.execution()