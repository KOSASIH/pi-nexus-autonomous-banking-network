import tensorflow as tf
from sagemaker.tensorflow import TensorFlow

def train_ai_model(bucket_name, model_path):
    # Train AI model using TensorFlow and SageMaker
    tf_estimator = TensorFlow(
        entry_point='ai_model.py',
        role='arn:aws:iam::123456789012:role/sagemaker-execution-role',
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=f's3://{bucket_name}/ai-models'
    )
    tf_estimator.fit(inputs='s3://{bucket_name}/training-data')

if __name__ == '__main__':
    bucket_name = 'your-s3-bucket-name'
    model_path = '3://{}/ai-models/model.tar.gz'.format(bucket_name)

    train_ai_model(bucket_name, model_path)
    print("AI model trained successfully!")
