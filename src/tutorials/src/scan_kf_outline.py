#!/usr/bin/env python3

from motion_model import MotionModel
import rospy
import matplotlib.pyplot as plt
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from gazebo_msgs.msg import ModelStates, ModelState
from std_msgs.msg import Float64MultiArray
from controller_outline import Controller


class KalmanFilter():
    def __init__(self, x_init):
        self.controller = Controller()

        self.model = MotionModel()
        self.A = None
        self.B = None
        self.C = self.model.C
        self.Q = None
        self.R = self.model.R
        self.P = np.eye(4)
        self.x = x_init

        self.last_time = None
        self.last_u = None

        self.x_real = []
        self.x_estimate = []
        self.x_sensor = []
        self.v_real = []
        self.v_estimate = []
        self.u_real = []

        rospy.Subscriber('/robot/control/transfered', TwistStamped, self.callback_twiststamped)
        rospy.Subscriber('/cylinder_laser/scan', LaserScan, self.callback_laserscan)

    def prediction_step(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update_step(self, y):
        innovation = y - self.C @ self.x
        lambda_t = self.C @ self.P @ self.C.T + self.R
        kalman_gain = self.P @ self.C.T @ np.linalg.inv(lambda_t)
        self.x = self.x + kalman_gain @ innovation
        self.P = self.P - kalman_gain @ self.C @ self.P

    # callback function of laser
    def callback_laserscan(self, laserScan: LaserScan):
        current_time = laserScan.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = current_time
            return 
        if current_time < self.last_time:
            return

        if self.last_u is None:
            return

        dt = current_time - self.last_time
        self.model.discretization(dt)
        self.A = self.model.A
        self.B = self.model.B
        self.Q = self.model.Q

        y = 5 - np.array(laserScan.ranges).reshape(2, 1)

        self.prediction_step(self.last_u)
        self.update_step(y)

        self.x_estimate.append((current_time, self.x[0], self.x[2]))
        self.x_sensor.append((current_time, y[0], y[1]))
        self.v_estimate.append((current_time, self.x[1], self.x[3]))

        self.last_time = current_time

        if len(self.x_estimate) > 50:
            self.visualize()

    # callback function of control signal
    def callback_twiststamped(self, twistStamped: TwistStamped):
        self.controller.callback_twist(twistStamped)

        current_time = twistStamped.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = current_time
            return
        if current_time < self.last_time:
            return

        dt = current_time - self.last_time
        self.model.discretization(dt)
        self.A = self.model.A
        self.B = self.model.B
        self.Q = self.model.Q

        ax = twistStamped.twist.angular.z
        ay = twistStamped.twist.linear.x
        u = np.array([[ax, ay]]).T
        self.prediction_step(u)
        self.u_real.append((current_time, ax, ay))

        # self.x_estimate.append((current_time, self.x[0], self.x[2]))
        # self.v_estimate.append((current_time, self.x[1], self.x[3]))
        self.x_real.append((current_time, self.controller.x[0], self.controller.x[2]))
        self.v_real.append((current_time, self.controller.x[1], self.controller.x[3]))
        
        # if len(self.x_estimate) > 50:
        #     self.visualize()


        self.last_u = u

        self.last_time = current_time

    def visualize(self):
        x_estimate_v = np.array(self.x_estimate)[:, 1:]
        x_estimate_t = np.array(self.x_estimate)[:, 0]

        x_real_v = np.array(self.x_real)[:, 1:]
        x_real_t = np.array(self.x_real)[:, 0]

        v_real_v = np.array(self.v_real)[:, 1:]
        v_real_t = np.array(self.v_real)[:, 0]

        v_estimate_v = np.array(self.v_estimate)[:, 1:]
        v_estimate_t = np.array(self.v_estimate)[:, 0]

        u_real_v = np.array(self.u_real)[:, 1:]

        plt.figure(0)
        plt.plot(x_estimate_v[:, 0], x_estimate_v[:, 1], label="estimated_trajectory")
        plt.plot(x_real_v[:, 0], x_real_v[:, 1], label="real_trajectory")
        plt.legend()
        plt.savefig("position.png")

        plt.figure(1)
        plt.plot(v_estimate_v[:, 0], v_estimate_v[:, 1], label="estimated_speed_trajectory")
        plt.plot(v_real_v[:, 0], v_real_v[:, 1], label="real_speed_trajectory")
        plt.legend()
        plt.savefig("speed.png")

        input(".....")

        plt.figure(2)
        plt.plot(u_real_v[:, 0], u_real_v[:, 1], label="real_force_trajectory")
        plt.legend()
        
        plt.show()


if __name__ == '__main__':
    x_init = np.zeros([4, 1])

    try:
        rospy.init_node('Kalman_filter')
        kf = KalmanFilter(x_init)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
