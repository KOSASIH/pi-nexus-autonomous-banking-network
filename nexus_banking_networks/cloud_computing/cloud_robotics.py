import rospy
from gazebo_msgs.msg import ModelState

def create_robot_model(model_name):
    # Create a new robot model
    rospy.init_node('robot_model')
    pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
    model_state = ModelState()
    model_state.model_name = model_name
    pub.publish(model_state)
    return model_state

def create_robot_simulation(simulation_name):
    # Create a new robot simulation
    rospy.init_node('robot_simulation')
    pub = rospy.Publisher('/gazebo/set_simulation_state', ModelState, queue_size=10)
    simulation_state = ModelState()
    simulation_state.simulation_name = simulation_name
    pub.publish(simulation_state)
    return simulation_state

if __name__ == '__main__':
    model_name = 'banking-robot'
    simulation_name = 'banking-simulation'

    model_state = create_robot_model(model_name)
    simulation_state = create_robot_simulation(simulation_name)
    print(f"Robot model created with name: {model_name}")
    print(f"Robot simulation created with name: {simulation_name}")
