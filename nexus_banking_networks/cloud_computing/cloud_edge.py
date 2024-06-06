import boto3

iot = boto3.client('iot')
greengrass = boto3.client('greengrass')

def create_iot_thing(thing_name):
    # Create a new IoT thing
    response = iot.create_thing(
        thingName=thing_name
    )
    return response['thingArn']

def create_greengrass_group(group_name):
    # Create a new Greengrass group
    response = greengrass.create_group(
        groupName=group_name
    )
    return response['groupId']

def deploy_greengrass_component(component_name, group_id):
    # Deploy a new Greengrass component
    response = greengrass.create_component(
        componentName=component_name,
        groupId=group_id
    )
    return response['componentId']

if __name__ == '__main__':
    thing_name = 'banking-iot-thing'
    group_name = 'banking-greengrass-group'
    component_name = 'banking-greengrass-component'

    thing_arn = create_iot_thing(thing_name)
    group_id = create_greengrass_group(group_name)
    component_id = deploy_greengrass_component(component_name, group_id)
    print(f"IoT thing created with ARN: {thing_arn}")
    print(f"Greengrass group created with ID: {group_id}")
    print(f"Greengrass component deployed with ID: {component_id}")
