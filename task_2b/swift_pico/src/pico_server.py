#!/usr/bin/env python3

'''
This python file runs a ROS 2 Action Server of name pico_server which navigates the Swift Pico Drone 
through waypoints received via action goals using your Task 2A PID controller architecture.
'''

from swift_msgs.msg import SwiftMsgs
from whycode_interfaces.msg import MarkerArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from waypoint_navigation.action import NavToWaypoint
from rclpy.action import ActionServer
import time
import rclpy
from rclpy.node import Node


class PicoServer(Node):
    def __init__(self):
        super().__init__('pico_server')  # initializing ros node with name pico_server

        # Action Server setup
        self._action_server = ActionServer(
            self,
            NavToWaypoint,
            'waypoint_navigation',
            self.execute_callback
        )

        # Current position of drone. Updated in whycode callback [x, y, z]
        self.current_state = [0.0, 0.0, 0.0]

        # Setpoint desired state [x_desired_state, y_desired_state, z_desired_state]
        self.desired_state = [0.0, 0.0, 13.0]  # Overridden dynamically by action goal

        # Declaring a cmd of message type SwiftMsgs and initializing values
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # Initial setting of Kp, Kd and Ki from your Task 2A code
        self.Kp = [81.0, 81.0, 42.0]
        self.Ki = [0.0, 0.0, 0.5]
        self.Kd = [6060.6, 6060.6, 1900.0]

        # PID variables from your Task 2A code
        self.error = [0.0, 0.0, 0.0]        # error in [x, y, z] = [pitch, roll, throttle]
        self.error_sum = [0.0, 0.0, 0.0]
        self.out = [0.0, 0.0, 0.0]          # pid output in [x, y, z] = [pitch, roll, throttle]
        self.gain_index = [1, 0, 2]          # position of [x, y, z] in Kp, Ki, Kd lists ([roll, pitch, throttle])
        self.direction = [1, 1, -1]          # sign of the output for [pitch, roll, throttle]
        self.integral_zone = 0.5             # error_sum added only when error is smaller than this
        self.integral_limit = [60.0, 60.0, 150.0]  # limit on Ki * error_sum

        self.prev_error = [0.0, 0.0, 0.0]
        self.max_values = [1800, 1800, 2000]
        self.min_values = [1200, 1200, 1000]

        self.sample_time = 0.033  # in seconds (~30 Hz)
        self.marker_timeout = 0.3
        self.last_marker_time = None
        self.first_run = True

        # Publishing /drone_command, /pos_error, /controller_heartbeat
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)

        self.pos_error = Error()

        # Subscribing to whycode markers and PID tuners
        self.create_subscription(MarkerArray, '/whycode_node/markers', self.whycode_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, "/pitch_pid", self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, "/roll_pid", self.roll_set_pid, 1)

        self.publish_heartbeat()
        self.create_timer(0.1, self.publish_heartbeat)

        time.sleep(0.5)  # let heartbeat reach rosbag before arming
        self.arm()  # ARMING THE DRONE

        self.get_logger().info('Pico Action Server initialized with Task 2A PID engine.')

    def publish_heartbeat(self):
        hb = Header()
        hb.stamp = self.get_clock().now().to_msg()
        hb.frame_id = 'pico_controller_active'
        self.heartbeat_pub.publish(hb)

    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)

    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    def whycode_callback(self, msg):
        if not msg.markers:
            return
        self.last_marker_time = self.get_clock().now()
        self.current_state[0] = msg.markers[0].position.position.x
        self.current_state[1] = msg.markers[0].position.position.y
        self.current_state[2] = msg.markers[0].position.position.z

    def altitude_set_pid(self, alt):
        self.Kp[2] = alt.kp * 0.03
        self.Ki[2] = alt.ki * 0.008
        self.Kd[2] = alt.kd * 18.18

    def pitch_set_pid(self, pitch):
        self.Kp[1] = pitch.kp * 0.03
        self.Ki[1] = pitch.ki * 0.008
        self.Kd[1] = pitch.kd * 18.18

    def roll_set_pid(self, roll):
        self.Kp[0] = roll.kp * 0.03
        self.Ki[0] = roll.ki * 0.008
        self.Kd[0] = roll.kd * 18.18

    def execute_callback(self, goal_handle):
        self.get_logger().info('Received waypoint goal from action client. Navigating...')

        # Extract target goal pose from action request
        goal_pose = goal_handle.request.waypoint
        self.desired_state[0] = goal_pose.position.x
        self.desired_state[1] = goal_pose.position.y
        self.desired_state[2] = goal_pose.position.z

        # Reset integral and error states for the new leg
        self.error_sum = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]
        self.first_run = True

        feedback_msg = NavToWaypoint.Feedback()
        start_time = time.time()
        stable_start_time = None
        required_stable_duration = 3.0  # Must hold for 3 continuous seconds[cite: 2]
        tolerance = 0.4                 # ±0.4 whycode units tolerance[cite: 2]

        rate = self.create_rate(int(1.0 / self.sample_time))

        while rclpy.ok():
            # 1. Execute your exact Task 2A PID algorithm calculation & publishing
            self.compute_pid_step()

            # 2. Publish current waypoint position feedback
            curr_pose = PoseStamped()
            curr_pose.header.stamp = self.get_clock().now().to_msg()
            curr_pose.header.frame_id = 'map'
            curr_pose.pose.position.x = self.current_state[0]
            curr_pose.pose.position.y = self.current_state[1]
            curr_pose.pose.position.z = self.current_state[2]
            feedback_msg.current_waypoint = curr_pose
            goal_handle.publish_feedback(feedback_msg)

            # 3. Check stability condition (±0.4 band on X, Y, Z continuously for 3 seconds)[cite: 2]
            err_x = abs(self.current_state[0] - self.desired_state[0])
            err_y = abs(self.current_state[1] - self.desired_state[1])
            err_z = abs(self.current_state[2] - self.desired_state[2])

            if err_x <= tolerance and err_y <= tolerance and err_z <= tolerance:
                if stable_start_time is None:
                    stable_start_time = time.time()
                elif (time.time() - stable_start_time) >= required_stable_duration:
                    hov_time = int(time.time() - start_time)
                    goal_handle.succeed()
                    result = NavToWaypoint.Result()
                    result.hov_time = hov_time
                    self.get_logger().info(f'Waypoint reached and held successfully in {hov_time}s!')
                    return result
            else:
                stable_start_time = None

            rate.sleep()

    def compute_pid_step(self):
        # Marker timeout safety check
        if self.last_marker_time is None or (self.get_clock().now() - self.last_marker_time).nanoseconds * 1e-9 > self.marker_timeout:
            self.cmd.rc_pitch = 1500
            self.cmd.rc_roll = 1500
            self.cmd.rc_throttle = 1500
            self.command_pub.publish(self.cmd)
            self.error_sum = [0.0, 0.0, 0.0]
            self.first_run = True
            return

        for i in range(3):  # i: 0 = x (pitch), 1 = y (roll), 2 = z (throttle)
            g = self.gain_index[i]

            # 1. error
            self.error[i] = self.desired_state[i] - self.current_state[i]

            # 2. sum of errors & change in error
            if abs(self.error[i]) < self.integral_zone:
                self.error_sum[i] += self.error[i]  # 8. Add error_sum
            else:
                self.error_sum[i] = 0.0
            if self.Ki[g] > 0:
                limit = self.integral_limit[i] / self.Ki[g]
                self.error_sum[i] = max(-limit, min(limit, self.error_sum[i]))
            d_error = self.error[i] - self.prev_error[i]

            # 3. pid output of this axis (with its sign)
            self.out[i] = self.direction[i] * (self.Kp[g] * self.error[i] + self.Ki[g] * self.error_sum[i] + self.Kd[g] * d_error)

            # 7. update previous error
            self.prev_error[i] = self.error[i]

        self.first_run = False

        # 4. + 6. add to the average value 1500 and limit between min and max
        self.cmd.rc_roll = int(max(self.min_values[0], min(self.max_values[0], 1500 + self.out[1])))
        self.cmd.rc_pitch = int(max(self.min_values[1], min(self.max_values[1], 1500 + self.out[0])))
        self.cmd.rc_throttle = int(max(self.min_values[2], min(self.max_values[2], 1500 + self.out[2])))

        # errors to publish
        self.pos_error.pitch_error = float(self.error[0])
        self.pos_error.roll_error = float(self.error[1])
        self.pos_error.throttle_error = float(self.error[2])

        self.command_pub.publish(self.cmd)
        self.pos_error_pub.publish(self.pos_error)


def main(args=None):
    rclpy.init(args=args)
    pico_server = PicoServer()

    try:
        rclpy.spin(pico_server)
    except KeyboardInterrupt:
        pico_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        pico_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
