#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from waypoint_navigation.srv import GetWaypoints
from waypoint_navigation.action import NavToWaypoint

class PicoClient(Node):
    def __init__(self):
        super().__init__('pico_client')
        
        # 1. Create service client to fetch the 4 predefined waypoints
        self.cli = self.create_client(GetWaypoints, 'waypoint_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for waypoint service to become available...')
        
        # 2. Create action client for waypoint navigation
        self._action_client = ActionClient(self, NavToWaypoint, 'waypoint_navigation')
        
        self.req = GetWaypoints.Request()
        self.req.get_waypoints = True
        self.waypoints_list = []
        self.marker_ids_list = []
        self.current_waypoint_index = 0

    def send_request(self):
        self.future = self.cli.call_async(self.req)

    def send_goal(self, pose, marker_name):
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        
        goal_msg = NavToWaypoint.Goal()
        goal_msg.waypoint = pose
        goal_msg.hover = True
        
        self.get_logger().info(f'Sending Goal [{marker_name}] -> X: {pose.position.x:.2f}, Y: {pose.position.y:.2f}, Z: {pose.position.z:.2f}')
        
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by the action server.')
            return

        self.get_logger().info('Goal accepted by server. Waiting for completion and stabilization...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        current_id = self.marker_ids_list[self.current_waypoint_index] if self.current_waypoint_index < len(self.marker_ids_list) else f"wp{self.current_waypoint_index+1}"
        
        self.get_logger().info(f'Waypoint [{current_id}] successfully reached and stabilized! Hover time: {result.hov_time}s.')
        
        # Move to the next waypoint in sequence
        self.current_waypoint_index += 1
        self.execute_next_waypoint()

    def feedback_callback(self, feedback_msg):
        # Optional: You can monitor feedback stream here if needed
        pass

    def execute_next_waypoint(self):
        if self.current_waypoint_index < len(self.waypoints_list):
            next_pose = self.waypoints_list[self.current_waypoint_index]
            marker_name = self.marker_ids_list[self.current_waypoint_index]
            self.send_goal(next_pose, marker_name)
        else:
            self.get_logger().info('All 4 waypoints successfully completed! Mission finished.')

def main(args=None):
    rclpy.init(args=args)
    client = PicoClient()
    
    # Request waypoints from the service
    client.send_request()
    rclpy.spin_until_future_complete(client, client.future)
    
    if client.future.result() is not None:
        response = client.future.result()
        client.waypoints_list = response.waypoints.poses
        client.marker_ids_list = response.marker_ids
        client.get_logger().info(f'Successfully retrieved {len(client.waypoints_list)} waypoints from service: {client.marker_ids_list}')
        
        # Start the sequence with the first waypoint
        client.execute_next_waypoint()
        
        rclpy.spin(client)
    else:
        client.get_logger().error('Failed to call waypoint_service.')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
