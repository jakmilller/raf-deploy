import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool as BoolMsg
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped, Pose, Point, Quaternion

import tf_transformations
import numpy as np
import math
import threading
import time

from raf_interfaces.srv import SetPose as SetPoseService
from raf_interfaces.srv import SetJointAngles as SetJointAnglesService
from raf_interfaces.srv import SetGripper as SetGripperService

# Kortex API imports (ensure these are correct for your setup)
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

from .kortex_api_utils import KortexAPIManager, wait_for_action_end_or_abort, create_action_notification_callback, KORTEX_ACTION_TIMEOUT_DURATION

# Custom services (define these in a separate .srv file if needed, e.g., in a 'raf_interfaces' package)
# For now, we'll use standard messages and simplify service definitions.
# from raf_interfaces.srv import SetPose, SetJointAngles, SetGripper
# For simplicity, we'll create services with standard types
from std_srvs.srv import Trigger # Example, replace with actual services

# If using ROS2 Actions for longer tasks
# from raf_interfaces.action import MoveToPose 

class KinovaControllerNode(Node):
    def __init__(self):
        super().__init__('kinova_controller_node')
        self.get_logger().info("Initializing Kinova Controller Node...")

        # Declare parameters
        self.declare_parameter('ip_address', '192.168.1.10')
        self.declare_parameter('username', 'admin')
        self.declare_parameter('password', 'admin')
        self.declare_parameter('gripper_dof', 1) # For Robotiq 2F-140 (1 finger object in Kortex API)
        self.declare_parameter('robot_dof', 6) # Degrees of freedom for the arm
        self.declare_parameter('state_publish_rate', 50.0) # Hz

        # Get parameters
        self.ip_address = self.get_parameter('ip_address').get_parameter_value().string_value
        self.username = self.get_parameter('username').get_parameter_value().string_value
        self.password = self.get_parameter('password').get_parameter_value().string_value
        self.gripper_dof = self.get_parameter('gripper_dof').get_parameter_value().integer_value
        self.robot_dof = self.get_parameter('robot_dof').get_parameter_value().integer_value
        state_publish_period = 1.0 / self.get_parameter('state_publish_rate').get_parameter_value().double_value
        
        # For handling Kortex actions that require waiting
        self.action_lock = threading.Lock()
        self.callback_group = ReentrantCallbackGroup() # For services that call Kortex actions

        # Kortex API Setup
        self.kortex_manager = KortexAPIManager(self.get_logger(), self.ip_address, self.username, self.password)
        if not self.kortex_manager.initialize_api():
            self.get_logger().fatal("Failed to initialize Kortex API. Shutting down.")
            # rclpy.shutdown() # This might cause issues if called directly in __init__
            # A better way is to let the node creation fail or handle it externally
            raise RuntimeError("Failed to initialize Kortex API")
        
        self.base = self.kortex_manager.base
        self.base_cyclic = self.kortex_manager.base_cyclic

        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, '~/joint_states', 10) # Use ~/ for relative topic
        self.cartesian_pose_pub = self.create_publisher(PoseStamped, '~/cartesian_pose', 10)
        # Add TwistStamped and WrenchStamped if needed

        # Subscribers
        self.e_stop_sub = self.create_subscription(
            BoolMsg, '~/emergency_stop', self.e_stop_callback, 10)
        
        # Subscriber for target pose from perception node
        self.target_pose_sub = self.create_subscription(
            PoseStamped, # Assuming perception publishes a PoseStamped
            '/perception/target_pose', # Example topic
            self.target_pose_callback,
            10,
            callback_group=self.callback_group)


        # Services (using standard messages for simplicity, define custom .srv for better typing)
        # Note: ROS2 services should ideally be non-blocking or use actions for long tasks.
        # For Kortex movements, which can take time, actions are preferred.
        # Here, services will block until Kortex action is done.
        self.set_pose_srv = self.create_service(
            SetPoseService, # Placeholder, replace with an actual service type e.g. from a custom interface
                            # For now, let's imagine a service that takes geometry_msgs/Pose
                            # Create a dummy service type for compilation for now:
                            # Dummy srv: geometry_msgs/Pose target_pose -> bool success
            '~/set_cartesian_pose', 
            self.handle_set_cartesian_pose,
            callback_group=self.callback_group)
        
        self.set_joint_angles_srv = self.create_service(
            SetJointAnglesService, # Placeholder
                                   # Dummy srv: float64[] joint_angles -> bool success
            '~/set_joint_angles', 
            self.handle_set_joint_angles,
            callback_group=self.callback_group)

        self.set_gripper_srv = self.create_service(
            SetGripperService,   # Placeholder
                                 # Dummy srv: float64 position (0-1) -> bool success
            '~/set_gripper_position', 
            self.handle_set_gripper_position,
            callback_group=self.callback_group)

        # Timer for publishing state
        self.state_publish_timer = self.create_timer(state_publish_period, self.publish_robot_state)

        self.get_logger().info("Kinova Controller Node initialized successfully.")

    def destroy_node(self):
        self.get_logger().info("Shutting down Kinova Controller Node...")
        if self.state_publish_timer:
            self.state_publish_timer.cancel()
        self.kortex_manager.close_api()
        super().destroy_node()
        self.get_logger().info("Kinova Controller Node shutdown complete.")

    def e_stop_callback(self, msg):
        if msg.data:
            self.get_logger().warn("Emergency Stop Requested!")
            try:
                if self.base:
                    self.base.Stop(Base_pb2.Empty())
                    self.get_logger().info("Kortex Stop command sent.")
            except Exception as e:
                self.get_logger().error(f"Error sending Kortex Stop command: {e}")
            # Potentially trigger a shutdown or safe state
            # rclpy.shutdown() # Be careful with this

    def publish_robot_state(self):
        try:
            feedback = self.base_cyclic.Refresh(BaseCyclic_pb2.Feedback())
            
            # Publish Joint States
            js_msg = JointState()
            js_msg.header.stamp = self.get_clock().now().to_msg()
            
            num_arm_actuators = min(self.robot_dof, len(feedback.actuators))
            for i in range(num_arm_actuators):
                js_msg.name.append(f'joint_{i+1}')
                js_msg.position.append(math.radians(feedback.actuators[i].position))
                js_msg.velocity.append(math.radians(feedback.actuators[i].velocity))
                js_msg.effort.append(feedback.actuators[i].torque)
            
            # Gripper state (Robotiq 2F-140 typically reported as one finger)
            if len(feedback.interconnect.oneof_tool_feedback.gripper_feedback) > 0:
                gripper_feedback = feedback.interconnect.oneof_tool_feedback.gripper_feedback[0]
                if len(gripper_feedback.motor) > 0: # Robotiq 2F-140
                    js_msg.name.append('gripper_finger_joint') # Or a more descriptive name
                    # Position is 0-100 from Kortex, convert to 0-1 or radians if needed
                    # Assuming 0-100 from Kortex corresponds to 0 (open) to 1 (closed) for our use
                    gripper_pos_percent = gripper_feedback.motor[0].position 
                    js_msg.position.append(gripper_pos_percent / 100.0) 
                    js_msg.velocity.append(gripper_feedback.motor[0].velocity / 100.0) # Assuming similar scale
                    js_msg.effort.append(gripper_feedback.motor[0].current_motor)

            self.joint_state_pub.publish(js_msg)

            # Publish Cartesian Pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = js_msg.header.stamp
            pose_msg.header.frame_id = "base_link" # Or your robot's base frame
            pose_msg.pose.position.x = feedback.base.tool_pose_x
            pose_msg.pose.position.y = feedback.base.tool_pose_y
            pose_msg.pose.position.z = feedback.base.tool_pose_z
            
            # Kortex provides ThetaX, ThetaY, ThetaZ in degrees ( Tait-Bryan ZYX, or RPY relative to fixed frame)
            # Convert to quaternion
            q = tf_transformations.quaternion_from_euler(
                math.radians(feedback.base.tool_pose_theta_x),
                math.radians(feedback.base.tool_pose_theta_y),
                math.radians(feedback.base.tool_pose_theta_z),
                axes='sxyz' # Check Kortex documentation for exact Euler angle convention
            )
            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]
            self.cartesian_pose_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing robot state: {e}")


    def _execute_kortex_action(self, action, description="Kortex action"):
        """Helper to execute a Kortex action and wait for completion."""
        with self.action_lock: # Ensure only one Kortex action is commanded at a time
            action_event = threading.Event()
            notification_cb = create_action_notification_callback(action_event, self.get_logger())
            notification_handle = self.base.OnNotificationActionTopic(
                notification_cb, 
                Base_pb2.NotificationOptions()
            )
            
            self.get_logger().info(f"Executing {description}...")
            try:
                self.base.ExecuteAction(action)
            except Exception as e:
                self.get_logger().error(f"Error executing {description}: {e}")
                self.base.Unsubscribe(notification_handle)
                return False

            if wait_for_action_end_or_abort(self.base, action_event):
                self.get_logger().info(f"{description} completed.")
                success = True
            else:
                self.get_logger().error(f"{description} timed out or failed.")
                success = False
            
            self.base.Unsubscribe(notification_handle)
            return success

    # --- Service Handlers ---
    # IMPORTANT: Define actual .srv files for these services in a separate interface package.
    # For now, these are conceptual placeholders using standard types.

    def handle_set_cartesian_pose(self, request, response):
        # Assuming request is geometry_msgs/Pose and response is a bool success
        self.get_logger().info(f"Set Cartesian Pose request received: {request.target_pose}")
        
        action = Base_pb2.Action()
        action.name = "Move to Cartesian Pose"
        action.application_data = ""

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = request.target_pose.position.x
        cartesian_pose.y = request.target_pose.position.y
        cartesian_pose.z = request.target_pose.position.z

        orientation_q = request.target_pose.orientation
        euler_angles_rad = tf_transformations.euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w],
            axes='sxyz' # Ensure this matches Kortex expectation
        )
        cartesian_pose.theta_x = math.degrees(euler_angles_rad[0])
        cartesian_pose.theta_y = math.degrees(euler_angles_rad[1])
        cartesian_pose.theta_z = math.degrees(euler_angles_rad[2])
        
        # Add constraint if needed (e.g., cartesian speed)
        # constraint = action.reach_pose.constraint
        # constraint.oneof_type.speed.translation = 0.1 # m/s
        # constraint.oneof_type.speed.orientation = 15.0 # deg/s

        response.success = self._execute_kortex_action(action, "Cartesian Pose Movement")
        return response

    def handle_set_joint_angles(self, request, response):
        # Assuming request is float64[] joint_angles (in radians) and response is bool success
        self.get_logger().info(f"Set Joint Angles request received: {request.joint_angles}")

        if len(request.joint_angles) != self.robot_dof:
            self.get_logger().error(f"Incorrect number of joint angles. Expected {self.robot_dof}, got {len(request.joint_angles)}")
            response.success = False
            return response

        action = Base_pb2.Action()
        action.name = "Move to Joint Angles"
        action.application_data = ""

        reach_joint_angles = action.reach_joint_angles.joint_angles
        for i, angle_rad in enumerate(request.joint_angles):
            joint_angle = reach_joint_angles.joint_angles.add()
            joint_angle.joint_identifier = i
            joint_angle.value = math.degrees(angle_rad)
        
        # Add constraint if needed (e.g., joint speed)
        # constraint = action.reach_joint_angles.constraint
        # constraint.type = Base_pb2.JOINT_CONSTRAINT_SPEED
        # constraint.value = 20.0 # deg/s for all joints

        response.success = self._execute_kortex_action(action, "Joint Angles Movement")
        return response

    def handle_set_gripper_position(self, request, response):
        # Assuming request is float64 position (0.0 open to 1.0 closed) and response is bool success
        self.get_logger().info(f"Set Gripper Position request received: {request.position}")

        if not (0.0 <= request.position <= 1.0):
            self.get_logger().error(f"Gripper position out of range (0.0-1.0): {request.position}")
            response.success = False
            return response

        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION # Position control
        
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1 # For Robotiq 2F-140, usually identifier 1
        finger.value = request.position * 100.0 # Kortex API uses 0-100 for position

        self.get_logger().info(f"Sending gripper command: value {finger.value}")
        try:
            with self.action_lock: # Ensure no other arm motion while commanding gripper
                self.base.SendGripperCommand(gripper_command)
                # Gripper commands are typically quick, but a short delay might be good
                # or if there's a way to confirm gripper action completion.
                # For now, assume it's sent. The state publisher will reflect change.
                time.sleep(0.5) # Small delay to allow command to be processed
            response.success = True
            self.get_logger().info("Gripper command sent.")
        except Exception as e:
            self.get_logger().error(f"Error sending gripper command: {e}")
            response.success = False
        return response

    def target_pose_callback(self, msg: PoseStamped):
        self.get_logger().info(f"Received target pose from perception: {msg.pose}")
        
        # This is where you'd call your internal service or directly execute the motion.
        # For simplicity, directly form the Kortex action here.
        # This logic should ideally be in an action server for better preemptibility.

        action = Base_pb2.Action()
        action.name = "Move to Perceived Target"
        action.application_data = ""

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = msg.pose.position.x
        cartesian_pose.y = msg.pose.position.y
        cartesian_pose.z = msg.pose.position.z # Add Z-offset for grasping here if needed

        orientation_q = msg.pose.orientation
        euler_angles_rad = tf_transformations.euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w],
            axes='sxyz' # Or the convention your perception node outputs
        )
        cartesian_pose.theta_x = math.degrees(euler_angles_rad[0])
        cartesian_pose.theta_y = math.degrees(euler_angles_rad[1])
        cartesian_pose.theta_z = math.degrees(euler_angles_rad[2])
        
        # TODO: Add logic from inference_class_final.py for approach, grasp, lift sequence
        # This might involve multiple calls to _execute_kortex_action or more complex action definitions.
        # Example:
        # 1. Move to a pre-grasp pose (offset above the target)
        # 2. Move down to grasp pose
        # 3. Close gripper
        # 4. Lift
        # For now, just move to the target pose
        
        if self._execute_kortex_action(action, "Movement to Perceived Target"):
            self.get_logger().info("Successfully moved to perceived target.")
            # Potentially trigger next step in automation (e.g., gripper close)
        else:
            self.get_logger().error("Failed to move to perceived target.")


# --- Dummy Service Definitions ---
# Replace these with actual .srv files in an interface package
class SetPoseService: # Placeholder
    class Request:
        target_pose = Pose()
    class Response:
        success = False

class SetJointAnglesService: # Placeholder
    class Request:
        joint_angles = [] # list of floats
    class Response:
        success = False

class SetGripperService: # Placeholder
    class Request:
        position = 0.0 # float
    class Response:
        success = False
# --- End Dummy Service Definitions ---


def main(args=None):
    rclpy.init(args=args)
    kinova_controller_node = None
    try:
        kinova_controller_node = KinovaControllerNode()
        # Use MultiThreadedExecutor to handle callbacks on different threads,
        # especially important if services block or for reentrant callback groups.
        executor = MultiThreadedExecutor()
        executor.add_node(kinova_controller_node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            kinova_controller_node.destroy_node() # Ensure cleanup
    except RuntimeError as e:
        if kinova_controller_node:
            kinova_controller_node.get_logger().fatal(f"Node initialization failed: {e}")
        else:
            rclpy.logging.get_logger("kinova_controller_main").fatal(f"Node initialization failed: {e}")
    except KeyboardInterrupt:
        if kinova_controller_node:
            kinova_controller_node.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()