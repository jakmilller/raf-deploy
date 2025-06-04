import rclpy
import threading

# Kortex API imports
# These will depend on how you've installed the Kortex Python API.
# Typically, they are like:
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.SessionManagerClientRpc import SessionManagerClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager

# Timeout duration for Kortex actions (in seconds)
KORTEX_ACTION_TIMEOUT_DURATION = 20.0


class KortexAPIManager:
    def __init__(self, logger, ip_address, username, password,
                 api_session_inactivity_timeout_ms=60000,
                 api_connection_inactivity_timeout_ms=2000):
        self.logger = logger
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.api_session_inactivity_timeout_ms = api_session_inactivity_timeout_ms
        self.api_connection_inactivity_timeout_ms = api_connection_inactivity_timeout_ms

        self.transport = None
        self.router = None
        self.session_manager = None
        self.base = None
        self.base_cyclic = None

    def initialize_api(self):
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

        self.logger.info(f"Connecting TCP transport to {self.ip_address}...")
        try:
            self.transport.connect(self.ip_address, Common_pb2.ROUTER_PORT) # Default Kortex API port
        except Exception as e:
            self.logger.error(f"Failed to connect TCP transport: {e}")
            return False

        self.logger.info("Creating Kortex API session...")
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self.username
        session_info.password = self.password
        session_info.session_inactivity_timeout = self.api_session_inactivity_timeout_ms
        session_info.connection_inactivity_timeout = self.api_connection_inactivity_timeout_ms

        self.session_manager = SessionManagerClient(self.router)
        try:
            self.session_manager.CreateSession(session_info)
            self.logger.info("Kortex API Session created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to create Kortex API session: {e}")
            self.transport.disconnect()
            return False

        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        
        # Clear faults
        try:
            self.logger.info("Clearing robot faults...")
            self.base.ClearFaults(Base_pb2.Empty())
            self.logger.info("Robot faults cleared.")
        except Exception as e:
            self.logger.warn(f"Could not clear robot faults: {e}")
            # Continue if clearing faults fails, but log a warning

        return True

    def close_api(self):
        if self.session_manager:
            try:
                self.logger.info("Closing Kortex API session...")
                self.session_manager.CloseSession(Session_pb2.Empty())
                self.logger.info("Kortex API session closed.")
            except Exception as e:
                self.logger.error(f"Error closing Kortex API session: {e}")
        
        if self.router:
            try:
                # This method might not exist or be named differently in newer API versions
                # self.router.SetActivationStatus(False) 
                pass
            except Exception as e:
                self.logger.error(f"Error deactivating router: {e}")

        if self.transport and self.transport.is_connected:
            try:
                self.logger.info("Disconnecting TCP transport...")
                self.transport.disconnect()
                self.logger.info("TCP transport disconnected.")
            except Exception as e:
                self.logger.error(f"Error disconnecting TCP transport: {e}")
                
        self.logger.info("Kortex API resources cleaned up.")


def wait_for_action_end_or_abort(base, action_future, timeout_seconds=KORTEX_ACTION_TIMEOUT_DURATION):
    """
    Waits for a Kortex action to complete (end or abort).
    This version assumes the action execution itself is asynchronous and we're
    monitoring notifications.
    Args:
        base: The BaseClient instance.
        action_future: A threading.Event object that will be set by the notification callback.
        timeout_seconds: How long to wait for the action.
    Returns:
        True if action completed (ended or aborted), False on timeout or error.
    """
    if not action_future.wait(timeout_seconds):
        rclpy.logging.get_logger("kortex_api_utils").error("Timeout waiting for Kortex action to complete.")
        # Consider sending a StopAction here if appropriate
        try:
            base.Stop(Base_pb2.Empty()) # Stop any ongoing action
        except Exception as e:
            rclpy.logging.get_logger("kortex_api_utils").error(f"Error sending stop action on timeout: {e}")
        return False
    return True

def create_action_notification_callback(action_event_to_set, logger):
    """
    Creates a callback function for Kortex action notifications.
    Sets the provided threading.Event when an action ends or aborts.
    """
    def on_action_notification(notification):
        action_event = notification.action_event
        if action_event == Base_pb2.ACTION_END:
            logger.info("Kortex action finished (ACTION_END).")
            action_event_to_set.set()
        elif action_event == Base_pb2.ACTION_ABORT:
            logger.warn("Kortex action aborted (ACTION_ABORT).")
            action_event_to_set.set()
        elif action_event == Base_pb2.ACTION_PAUSE:
            logger.info("Kortex action paused (ACTION_PAUSE).")
        # Add other events as needed
    return on_action_notification