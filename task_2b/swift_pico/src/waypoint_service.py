#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from waypoint_navigation.srv import GetWaypoints

class WaypointService(Node):
    def __init__(self):
        super().__init__('waypoint_service')
        self.srv = self.create_service(GetWaypoints, 'waypoint_service', self.handle_get_waypoints)
        self.get_logger().info('Waypoint Service is ready and waiting for requests.')

    def handle_get_waypoints(self, request, response):
        if request.get_waypoints:
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = 'map'

            # Waypoints mapping:
            # wp1: x=-2.41, y=-2.41, z=13.00
            # wp2: x=2.41,  y=1.20,  z=13.00
            # wp3: x=1.20,  y=-2.41, z=13.00
            # wp4: x=-1.20, y=1.20,  z=13.00
            coords = [
                (-2.41, -2.41, 13.00),
                (2.41, 1.20, 13.00),
                (1.20, -2.41, 13.00),
                (-1.20, 1.20, 13.00)
            ]

            marker_list = []
            for i, (x, y, z) in enumerate(coords, start=1):
                p = Pose()
                p.position.x = x
                p.position.y = y
                p.position.z = z
                # Orientation default (no rotation)
                p.orientation.w = 1.0
                pose_array.poses.append(p)
                marker_list.append(f"wp{i}")

            response.waypoints = pose_array
            response.marker_ids = marker_list
            self.get_logger().info('Returning 4 waypoints successfully.')
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = WaypointService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
