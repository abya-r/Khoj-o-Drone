#!/usr/bin/env python3

'''
This python file runs a ROS 2-node of name pico_control which holds the position of Swift Pico Drone on the given drone.
This node publishes and subsribes the following topics:

        PUBLICATIONS            SUBSCRIPTIONS
        /drone_command            /whycode_node/markers
        /pos_error                /throttle_pid
                                  /pitch_pid
                                  /roll_pid

Rather than using different variables, use list. eg : self.desired_state = [1,2,3], where index corresponds to x,y,z ...rather than defining self.x_desired_state = 1, self.y_desired_state = 2
CODE MODULARITY AND TECHNIQUES MENTIONED LIKE THIS WILL HELP YOU GAINING MORE MARKS WHILE CODE EVALUATION.
'''

# Importing the required libraries

from swift_msgs.msg import SwiftMsgs
from whycode_interfaces.msg import MarkerArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error
from std_msgs.msg import Header
import rclpy
from rclpy.node import Node


class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')  # initializing ros node with name pico_controller

        # This corresponds to your current position of drone. This value must be updated in your whycode callback
        # [x,y,z]
        self.current_state = [0.0, 0.0, 0.0]

        # This corresponds to the setpoint you want the drone to reach or hold
        # [x_desired_state, y_desired_state, z_desired_state]
        self.desired_state = [1.5, -1.5, 13.0]  # whycode marker at the position of the drone given in the scene. Make the whycode marker associated with position_to_hold drone renderable and make changes accordingly

        # Declaring a cmd of message type swift_msgs and initializing values
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # initial setting of Kp, Kd and ki for [roll, pitch, throttle]. eg: self.Kp[2] corresponds to Kp value in throttle axis
        # after tuning and computing corresponding PID parameters, change the parameters
        self.Kp = [9.0, 9.0, 42.0]
        self.Ki = [0.12, 0.12, 0.25]
        self.Kd = [8.0, 8.0, 32.0]

        #-----------------------Add other required variables for pid here ----------------------------------------------
        # ==========================================
        # ABYA'S TASK: Core State Variables
        # Add variables for tracking previous errors, error sums (for integral), and error values:
        self.error = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]
        self.out = [0.0, 0.0, 0.0]
        #   self.max_values = [2000, 2000, 2000]
        #   self.min_values = [1000, 1000, 1000]
        # ==========================================
        self.direction = [1, 1, -1]
        self.integral_zone = 1.0
        self.integral_limit = [60.0, 60.0, 150.0]

        self.marker_timeout = 0.3
        self.last_marker_time = None
        self.first_run = True

        self.pos_error = Error()

        self.max_values = [1800, 1800, 2000]
        self.min_values = [1200, 1200, 1000]
        
        #----------------------------------------------------------------------------------------------------------

        # # This is the sample time in which you need to run pid. Choose any time which you seem fit.
        self.sample_time = 0.033  # in seconds

        # Publishing /drone_command, /pos_error
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)

        #------------------------Add other ROS 2 Publishers here-----------------------------------------------------

        # Subscribing to /whycode_node/markers, /throttle_pid, /pitch_pid, roll_pid
        self.create_subscription(MarkerArray, '/whycode_node/markers', self.whycode_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)

        #------------------------Add other ROS Subscribers here-----------------------------------------------------
        # ==========================================
        # ANAMIKA'S TASK: Pitch and Roll Subscriptions
        # Add subscriptions for pitch and roll tuning topics:
        #   self.create_subscription(PIDTune, "/pitch_pid", self.pitch_set_pid, 1)
        #   self.create_subscription(PIDTune, "/roll_pid", self.roll_set_pid, 1)
        # ==========================================

        self.arm()  # ARMING THE DRONE

        self.publish_heartbeat()
        self.create_timer(1.0, self.publish_heartbeat)

        # Creating a timer to run the pid function periodically, refer ROS 2 tutorials on how to create a publisher subscriber(Python)
        # ==========================================
        # KRITIKA'S TASK: Control Loop Timer Setup
        # Initialize the timer to execute self.pid periodically:
        #   self.create_timer(self.sample_time, self.pid)
        # ==========================================

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
        self.command_pub.publish(self.cmd)  # Publishing /drone_command

    # Whycode callback function
    # The function gets executed each time when /whycode_node publishes /whycode_node/markers
    def whycode_callback(self, msg):
        self.current_state[0] = msg.markers[0].position.position.x
        #--------------------Set the remaining co-ordinates of the drone from msg----------------------------------------------
        # ==========================================
        # ADITRI'S TASK: Position Extraction
        # Extract y and z position markers:
        #   self.current_state[1] = msg.markers[0].position.position.y
        #   self.current_state[2] = msg.markers[0].position.position.z
        # ==========================================

        #---------------------------------------------------------------------------------------------------------------

    # Callback function for /throttle_pid
    # This function gets executed each time when /drone_pid_tuner publishes /throttle_pid
    def altitude_set_pid(self, alt):
        self.Kp[2] = alt.kp * 0.03  # This is just for an example. You can change the ratio/fraction value accordingly
        self.Ki[2] = alt.ki * 0.008
        self.Kd[2] = alt.kd * 0.6

    #----------------------------Define callback function like altitide_set_pid to tune pitch, roll--------------
    # ==========================================
    # ANAMIKA'S TASK: Pitch and Roll Callbacks
    # Define pitch_set_pid(self, pitch) and roll_set_pid(self, roll):
    #   def pitch_set_pid(self, pitch):
    #       self.Kp[1] = pitch.kp * 0.03
    #       self.Ki[1] = pitch.ki * 0.008
    #       self.Kd[1] = pitch.kd * 0.6
    #
    #   def roll_set_pid(self, roll):
    #       self.Kp[0] = roll.kp * 0.03
    #       self.Ki[0] = roll.ki * 0.008
    #       self.Kd[0] = roll.kd * 0.6
    # ==========================================

    #----------------------------------------------------------------------------------------------------------------------

    def pid(self):
    #-----------------------------Write the PID algorithm here--------------------------------------------------------------
    # ==========================================
    # ABYA'S TASK: Core PID Algorithm & RC Mapping
    # 1. Compute error[i] = self.current_state[i] - self.desired_state[i] for axes 0, 1, 2
    # 2. Accumulate self.error_sum[i] += self.error[i] with anti-windup clamping
    # 3. Calculate p_term, i_term, and d_term
    # 4. Compute output command and update RC values:
    #      self.cmd.rc_roll = int(1500 + out[0])
    #      self.cmd.rc_pitch = int(1500 + out[1])
    #      self.cmd.rc_throttle = int(1500 + out[2])
    # 5. Clamp RC outputs between self.min_values and self.max_values
    # 6. Update self.prev_error[i] = self.error[i]
    # ==========================================
        if self.last_marker_time is None or (self.get_clock().now() - self.last_marker_time).nanoseconds * 1e-9 > self.marker_timeout:
            self.cmd.rc_pitch = 1500
            self.cmd.rc_roll = 1500
            self.cmd.rc_throttle = 1500
            self.command_pub.publish(self.cmd)
            self.error_sum = [0.0, 0.0, 0.0]
            self.first_run = True
            return

        for i in range(3):
			# 1. error = desired - current
            self.error[i] = self.desired_state[i] - self.current_state[i]
            if self.first_run:
                self.prev_error[i] = self.error[i]
 
			# 2. error_sum (only near the setpoint, clamped -> no windup) and change in error
            if abs(self.error[i]) < self.integral_zone:
                self.error_sum[i] += self.error[i]
            else:
                self.error_sum[i] = 0.0
            if self.Ki[i] > 0:
                limit = self.integral_limit[i] / self.Ki[i]
                self.error_sum[i] = max(-limit, min(limit, self.error_sum[i]))
            d_error = self.error[i] - self.prev_error[i]
 
			# 3. pid output, with the sign of this axis
            self.out[i] = self.direction[i] * (self.Kp[i] * self.error[i] + self.Ki[i] * self.error_sum[i] + self.Kd[i] * d_error)
 
			# 7. update previous error
            self.prev_error[i] = self.error[i]
        self.first_run = False
 
		# 4. + 6. add to the neutral value 1500 and limit to the valid range
        self.cmd.rc_pitch = int(max(self.min_values[0], min(self.max_values[0], 1500 + self.out[0])))
        self.cmd.rc_roll = int(max(self.min_values[1], min(self.max_values[1], 1500 + self.out[1])))
        self.cmd.rc_throttle = int(max(self.min_values[2], min(self.max_values[2], 1500 + self.out[2])))
 
		# errors to publish
        self.pos_error.pitch_error = float(self.error[0])
        self.pos_error.roll_error = float(self.error[1])
        self.pos_error.throttle_error = float(self.error[2])

    #------------------------------------------------------------------------------------------------------------------------
        self.command_pub.publish(self.cmd)
        # calculate throttle error, pitch error and roll error, then publish it accordingly
        # ==========================================
        # ADITRI'S TASK: Positional Error Publishing
        # Construct and publish pos_error message:
        #   pos_error_msg = Error()
        #   pos_error_msg.x_error = float(self.error[0])
        #   pos_error_msg.y_error = float(self.error[1])
        #   pos_error_msg.z_error = float(self.error[2])
        #   self.pos_error_pub.publish(pos_error_msg)
        # ==========================================
        self.pos_error_pub.publish(self.pos_error)


def main(args=None):
    rclpy.init(args=args)
    swift_pico = Swift_Pico()
     # Creating a timer to run the pid function periodically, refer ROS 2 tutorials on how to create a publisher subscriber(Python)
        self.create_timer(self.sample_time, self.pid)
    # ==========================================
    # KRITIKA'S TASK: Execution & Shutdown
    # Ensure clean spin loop and safe node shutdown sequence
    # ==========================================
    try:
        rclpy.spin(swift_pico)
    except KeyboardInterrupt:
        swift_pico.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        swift_pico.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
