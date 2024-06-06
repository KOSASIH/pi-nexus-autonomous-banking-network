import boto3

sagemaker = boto3.client('sagemaker')
comprehend = boto3.client('comprehend')

def create_sagemaker_notebook(notebook_name):
    # Create a new SageMaker notebook instance
    response = sagemaker.create_notebook_instance(
        NotebookInstanceName=notebook_name,
        RoleArn='arn:aws:iam::123456789012:role/sagemaker-execution-role',
        InstanceType='ml.t2.medium'
    )
    return response['NotebookInstance']['NotebookInstanceArn']

def create_comprehend_entity_recognizer(recognizer_name):
    # Create a new Comprehend entity recognizer
    response = comprehend.create_entity_recognizer(
        RecognizerName=recognizer_name,
        DataAccessRoleArn='arn:aws:iam::123456789012:role/comprehend-execution-role'
    )
    return response['EntityRecognizerArn']

if __name__ == '__main__':
    notebook_name = 'banking-notebook'
    recognizer_name = 'banking-entity-recognizer'

    notebook_arn = create_sagemaker_notebook(notebook_name)
    recognizer_arn = create_comprehend_entity_recognizer(recognizer_name)
    print(f"SageMaker notebook created with ARN: {notebook_arn}")
    print(f"Comprehend entity recognizer created with ARN: {recognizer_arn}")
