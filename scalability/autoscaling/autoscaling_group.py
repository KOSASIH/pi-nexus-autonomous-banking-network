import boto3

autoscaling_client = boto3.client('autoscaling')

def create_autoscaling_group():
    # Create Auto Scaling group
    response = autoscaling_client.create_auto_scaling_group(
        AutoScalingGroupName='pi-nexus-asg',
        LaunchConfigurationName='pi-nexus-lc',
        MinSize=1,
        MaxSize=10,
        DesiredCapacity=3
    )
    asg_name = response['AutoScalingGroupName']

    return asg_name

def update_autoscaling_group(asg_name):
    # Update Auto Scaling group
    response = autoscaling_client.update_auto_scaling_group(
        AutoScalingGroupName=asg_name,
        MinSize=1,
        MaxSize=15,
        DesiredCapacity=5
    )
