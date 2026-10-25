#!/usr/bin/env python3

'''
ROS 2 Action Server "pico_server": navigates the Swift Pico drone to waypoints
received as NavToWaypoint goals, using the Task 2A PID controller (+ simple yaw PD).
The PID runs continuously on a timer; the action sets the target, streams
feedback and returns once the drone has held the waypoint for 3 s.
'''

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from swift_msgs.msg import SwiftMsgs
from whycode_interfaces.msg import MarkerArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from waypoint_navigation.action import NavToWaypoint


class PicoServer(Node):
    def __init__(self):
        super().__init__('pico_server')

        # Reentrant group so the PID timer, marker callback and the action's
        # execute loop can all run at the same time (needs MultiThreadedExecutor)
        self.cb_group = ReentrantCallbackGroup()

        # [x, y, z] current / desired position
        self.current_state = [0.0, 0.0, 0.0]
        self.desired_state = [0.0, 0.0, 13.0]   # overwritten by each action goal

        # yaw
        self.current_yaw = None
        self.desired_yaw = None                 # captured from the first marker reading

        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # Gains from Task 2A, index = [roll, pitch, throttle]
        self.Kp = [81.0, 81.0, 42.0]
        self.Ki = [0.0, 0.0, 0.5]
        self.Kd = [6060.6, 6060.6, 1900.0]

        self.error = [0.0, 0.0, 0.0]            # [x, y, z] = [pitch, roll, throttle]
        self.error_sum = [0.0, 0.0, 0.0]
        self.out = [0.0, 0.0, 0.0]
        self.gain_index = [1, 0, 2]
        self.direction = [1, 1, -1]
        self.integral_zone = 0.5
        self.integral_limit = [60.0, 60.0, 150.0]
        self.prev_error = [0.0, 0.0, 0.0]
        self.max_values = [1800, 1800, 2000]    # [roll, pitch, throttle]
        self.min_values = [1200, 1200, 1000]

        # ---- yaw PD (tune / flip yaw_dir if needed) ----
        self.use_yaw = True
        self.yaw_dir = 1.0          # flip to -1.0 if the drone spins instead of correcting
        self.Kp_yaw = 150.0         # rc units per radian
        self.Kd_yaw = 400.0
        self.yaw_limit = 150.0      # max deviation from 1500
        self.prev_yaw_error = 0.0

        self.sample_time = 0.033
        self.marker_timeout = 0.3
        self.last_marker_time = None
        self.first_run = True

        # Publishers
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)
        self.pos_error = Error()

        # Subscribers
        self.create_subscription(MarkerArray, '/whycode_node/markers',
                                 self.whycode_callback, 1, callback_group=self.cb_group)
        self.create_subscription(PIDTune, '/throttle_pid', self.altitude_set_pid, 1,
                                 callback_group=self.cb_group)
        self.create_subscription(PIDTune, '/pitch_pid', self.pitch_set_pid, 1,
                                 callback_group=self.cb_group)
        self.create_subscription(PIDTune, '/roll_pid', self.roll_set_pid, 1,
                                 callback_group=self.cb_group)

        # Action server
        self._action_server = ActionServer(
            self, NavToWaypoint, 'waypoint_navigation',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group)

        self.publish_heartbeat()
        self.create_timer(0.1, self.publish_heartbeat, callback_group=self.cb_group)

        time.sleep(0.5)     # let the heartbeat reach the rosbag before arming
        self.arm()

        # PID runs continuously, holding whatever desired_state currently is
        self.create_timer(self.sample_time, self.pid, callback_group=self.cb_group)

        self.get_logger().info('Pico Action Server ready (action: waypoint_navigation).')

    # ------------------------------------------------------------------ basics
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

    # --------------------------------------------------------------- callbacks
    def whycode_callback(self, msg):
        if not msg.markers:
            return
        self.last_marker_time = self.get_clock().now()
        pose = msg.markers[0].position
        self.current_state[0] = pose.position.x
        self.current_state[1] = pose.position.y
        self.current_state[2] = pose.position.z

        # yaw from the marker's quaternion
        q = pose.orientation
        self.current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.desired_yaw is None:
            self.desired_yaw = self.current_yaw     # hold the yaw we start with

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

    # ------------------------------------------------------------------ action
    def goal_callback(self, goal_request):
        self.get_logger().info('Waypoint goal received.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        wp = goal_handle.request.waypoint
        self.error_sum = [0.0, 0.0, 0.0]     # fresh integral for the new leg
        self.first_run = True                # avoid a derivative kick on the new setpoint
        self.desired_state = [wp.position.x, wp.position.y, wp.position.z]
        self.get_logger().info(
            f'Navigating to x={wp.position.x:.2f}, y={wp.position.y:.2f}, z={wp.position.z:.2f}')

        feedback_msg = NavToWaypoint.Feedback()
        start_time = time.time()
        stable_since = None
        hold_time = 3.0         # must stay inside the band for 3 s
        stable_band = 0.35      # slightly tighter than the 0.4 limit, for safety margin

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return NavToWaypoint.Result()

            # feedback: current whycode position
            fb = PoseStamped()
            fb.header.stamp = self.get_clock().now().to_msg()
            fb.header.frame_id = 'map'
            fb.pose.position.x = self.current_state[0]
            fb.pose.position.y = self.current_state[1]
            fb.pose.position.z = self.current_state[2]
            feedback_msg.current_waypoint = fb
            goal_handle.publish_feedback(feedback_msg)

            in_band = all(abs(self.desired_state[i] - self.current_state[i]) <= stable_band
                          for i in range(3))

            if in_band:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= hold_time:
                    result = NavToWaypoint.Result()
                    result.hov_time = int(round(time.time() - start_time))
                    goal_handle.succeed()
                    self.get_logger().info(f'Waypoint reached and held. Time taken: {result.hov_time}s')
                    return result
            else:
                stable_since = None

            self.get_logger().info(
                f'err x={self.error[0]:+.2f} y={self.error[1]:+.2f} z={self.error[2]:+.2f}',
                throttle_duration_sec=1.0)
            time.sleep(0.05)

        goal_handle.abort()
        return NavToWaypoint.Result()

    # --------------------------------------------------------------------- PID
    def pid(self):
        # No fresh marker -> neutral command
        if (self.last_marker_time is None or
                (self.get_clock().now() - self.last_marker_time).nanoseconds * 1e-9 > self.marker_timeout):
            self.cmd.rc_pitch = 1500
            self.cmd.rc_roll = 1500
            self.cmd.rc_throttle = 1500
            self.cmd.rc_yaw = 1500
            self.command_pub.publish(self.cmd)
            self.error_sum = [0.0, 0.0, 0.0]
            self.first_run = True
            return

        desired = list(self.desired_state)

        # 1. errors
        for i in range(3):
            self.error[i] = desired[i] - self.current_state[i]

        # on a new setpoint, start the derivative from the current error (no kick)
        if self.first_run:
            for i in range(3):
                self.prev_error[i] = self.error[i]

        # 2./3. position PID per axis
        for i in range(3):          # 0 = x (pitch), 1 = y (roll), 2 = z (throttle)
            g = self.gain_index[i]

            if abs(self.error[i]) < self.integral_zone:
                self.error_sum[i] += self.error[i]
            else:
                self.error_sum[i] = 0.0
            if self.Ki[g] > 0:
                limit = self.integral_limit[i] / self.Ki[g]
                self.error_sum[i] = max(-limit, min(limit, self.error_sum[i]))

            d_error = self.error[i] - self.prev_error[i]
            self.out[i] = self.direction[i] * (self.Kp[g] * self.error[i]
                                               + self.Ki[g] * self.error_sum[i]
                                               + self.Kd[g] * d_error)
            self.prev_error[i] = self.error[i]

        # 4. + 6. add to 1500 and clamp  (max/min order: [roll, pitch, throttle])
        self.cmd.rc_roll = int(max(self.min_values[0], min(self.max_values[0], 1500 + self.out[1])))
        self.cmd.rc_pitch = int(max(self.min_values[1], min(self.max_values[1], 1500 + self.out[0])))
        self.cmd.rc_throttle = int(max(self.min_values[2], min(self.max_values[2], 1500 + self.out[2])))

        # yaw PD
        if self.use_yaw and self.current_yaw is not None and self.desired_yaw is not None:
            e_yaw = self.desired_yaw - self.current_yaw
            e_yaw = math.atan2(math.sin(e_yaw), math.cos(e_yaw))     # wrap to [-pi, pi]
            if self.first_run:
                self.prev_yaw_error = e_yaw
            d_yaw = e_yaw - self.prev_yaw_error
            out_yaw = self.yaw_dir * (self.Kp_yaw * e_yaw + self.Kd_yaw * d_yaw)
            out_yaw = max(-self.yaw_limit, min(self.yaw_limit, out_yaw))
            self.cmd.rc_yaw = int(1500 + out_yaw)
            self.prev_yaw_error = e_yaw
        else:
            self.cmd.rc_yaw = 1500

        self.first_run = False

        # publish errors + command
        self.pos_error.pitch_error = float(self.error[0])
        self.pos_error.roll_error = float(self.error[1])
        self.pos_error.throttle_error = float(self.error[2])

        self.command_pub.publish(self.cmd)
        self.pos_error_pub.publish(self.pos_error)


def main(args=None):
    rclpy.init(args=args)
    node = PicoServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
