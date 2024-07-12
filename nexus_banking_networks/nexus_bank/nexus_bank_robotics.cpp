#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

class Robotics {
public:
    Robotics() {}

    void processLaserScan(const sensor_msgs::LaserScan::ConstPtr& scan) {
        // Process the laser scan data
        for (int i = 0; i < scan->ranges.size(); i++) {
            std::cout << "Range " << i << ": " << scan->ranges[i] << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "nexus_bank_robotics");
    ros::NodeHandle nh;

    Robotics robotics;
    ros::Subscriber sub = nh.subscribe("scan", 1000, &Robotics::processLaserScan, &robotics);

    ros::spin();

    return 0;
}
