#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from waypoint_navigation.srv import GetWaypoints
from waypoint_navigation.action import NavToWaypoint


class PicoClient(Node):
    def __init__(self):
        super().__init__('pico_client')

        # Service client to fetch the 4 predefined waypoints
        self.cli = self.create_client(GetWaypoints, 'waypoint_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for waypoint service to become available...')

        # Action client for waypoint navigation
        self._action_client = ActionClient(self, NavToWaypoint, 'waypoint_navigation')

        self.req = GetWaypoints.Request()
        self.req.get_waypoints = True
        self.waypoints_list = []
        self.marker_ids_list = []
        self.current_waypoint_index = 0
        self.start_time = None
        self.done = False

    def send_request(self):
        self.future = self.cli.call_async(self.req)

    def send_goal(self, pose, marker_name):
        goal_msg = NavToWaypoint.Goal()
        goal_msg.waypoint = pose
        goal_msg.hover = True

        self.get_logger().info(
            f'Sending Goal [{marker_name}] -> X: {pose.position.x:.2f}, '
            f'Y: {pose.position.y:.2f}, Z: {pose.position.z:.2f}')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by the action server. Stopping.')
            self.done = True
            return

        self.get_logger().info('Goal accepted. Waiting for the drone to reach and stabilise...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        wrapped = future.result()
        name = self.marker_ids_list[self.current_waypoint_index]

        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(f'Waypoint [{name}] failed (status {wrapped.status}). Stopping.')
            self.done = True
            return

        self.get_logger().info(
            f'Waypoint [{name}] reached and stabilised. Hover time: {wrapped.result.hov_time}s '
            f'(total elapsed {time.time() - self.start_time:.1f}s)')

        # Only now do we move on to the next waypoint
        self.current_waypoint_index += 1
        self.execute_next_waypoint()

    def feedback_callback(self, feedback_msg):
        p = feedback_msg.feedback.current_waypoint.pose.position
        self.get_logger().debug(f'Feedback: x={p.x:.2f} y={p.y:.2f} z={p.z:.2f}')

    def execute_next_waypoint(self):
        if self.current_waypoint_index < len(self.waypoints_list):
            pose = self.waypoints_list[self.current_waypoint_index]
            name = self.marker_ids_list[self.current_waypoint_index]
            self.send_goal(pose, name)
        else:
            self.get_logger().info(
                f'Navigation complete! All {len(self.waypoints_list)} waypoints done in '
                f'{time.time() - self.start_time:.1f}s.')
            self.done = True


def main(args=None):
    rclpy.init(args=args)
    client = PicoClient()

    # 1. Ask the service for the waypoints (order is kept exactly as returned)
    client.send_request()
    rclpy.spin_until_future_complete(client, client.future)

    if client.future.result() is None:
        client.get_logger().error('Failed to call waypoint_service.')
    else:
        response = client.future.result()
        client.waypoints_list = list(response.waypoints.poses)
        client.marker_ids_list = list(response.marker_ids)
        client.get_logger().info(
            f'Retrieved {len(client.waypoints_list)} waypoints: {client.marker_ids_list}')

        # 2. Wait for the action server (starting the server last kicks off the flight)
        client.get_logger().info('Waiting for action server...')
        client._action_client.wait_for_server()

        # 3. Send waypoints one by one
        client.start_time = time.time()
        client.execute_next_waypoint()
        while rclpy.ok() and not client.done:
            rclpy.spin_once(client, timeout_sec=0.1)

    client.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
