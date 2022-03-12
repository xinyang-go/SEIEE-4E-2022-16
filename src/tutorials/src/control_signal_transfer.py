#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped


class Transformation(object):
    def __init__(self):
        self.twiststamped_pub = rospy.Publisher('robot/control/transfered', TwistStamped, queue_size=10)
        rospy.Subscriber('/robot/control', Twist, self.Control_signal_transmation)

    def Control_signal_transmation(self, Twist):
        rostime = rospy.get_rostime()
        twiststamped = TwistStamped()
        twiststamped.twist = Twist
        twiststamped.header.stamp = rostime
        self.twiststamped_pub.publish(twiststamped)
        rospy.loginfo(twiststamped)


if __name__ == '__main__':
    try:
        rospy.init_node('control_signal_trans', anonymous=True)
        control_signal_trans = Transformation()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
