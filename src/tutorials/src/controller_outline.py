#!/usr/bin/env python3

import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import TwistStamped
from motion_model import MotionModel


class Controller(object):
    # constructor
    def __init__(self):
        self.last_time = rospy.get_time()
        self.model = MotionModel()
        self.x = np.zeros([4, 1])

        self.state_pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)
        rospy.Subscriber('/robot/control/transfered', TwistStamped, self.callback_twist)

    # receive the control signal and generate the modelstate, modify the modelstate
    # @param    TwistStamped: control sognal with timestamp
    # @return   NULL
    def callback_twist(self, twistStamped: TwistStamped):
        self.dt = rospy.get_time() - self.last_time
        self.last_time = rospy.get_time()

        state = ModelState()
        state.model_name = "cylinder_laser"

        ax = twistStamped.twist.angular.z
        ay = twistStamped.twist.linear.x
        u = np.array([[ax, ay]]).T

        self.x, _ = self.model.step(self.x, u, self.dt, True)

        state.pose.position.x = self.x[0]
        state.pose.position.y = self.x[2]

        rospy.loginfo(state)
        self.state_pub.publish(state)


if __name__ == '__main__':
    try:
        rospy.init_node('Kalman_controller', anonymous=True)
        controller = Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
