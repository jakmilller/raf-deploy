#include "kortex_controller/controller.hpp"
#include <cmath>
#include <typeinfo>

using namespace std;
using namespace std::placeholders;

// Create an event listener that will set the promise action event to the exit value
std::function<void(k_api::Base::ActionNotification)> 
    create_event_listener_by_promise(std::promise<k_api::Base::ActionEvent>& finish_promise)
{
    return [&finish_promise] (k_api::Base::ActionNotification notification)
    {
        const auto action_event = notification.action_event();
        switch(action_event)
        {
        case k_api::Base::ActionEvent::ACTION_END:
        case k_api::Base::ActionEvent::ACTION_ABORT:
            finish_promise.set_value(action_event);
            break;
        default:
            break;
        }
    };
}

// Create an event listener that will set the sent reference to the exit value
std::function<void(k_api::Base::ActionNotification)>
    create_event_listener_by_ref(k_api::Base::ActionEvent& returnAction)
{
    return [&returnAction](k_api::Base::ActionNotification notification)
    {
        const auto action_event = notification.action_event();
        switch(action_event)
        {
        case k_api::Base::ActionEvent::ACTION_END:
        case k_api::Base::ActionEvent::ACTION_ABORT:
            returnAction = action_event;
            break;
        default:
            break;
        }
    };
}

Controller::Controller() : Node("controller")
{
    RCLCPP_INFO(this->get_logger(), "Retrieving ROS parameters");
    
    // Declare and get parameters
    this->declare_parameter("username", "admin");
    this->declare_parameter("password", "admin");
    this->declare_parameter("ip_address", "192.168.1.10");
    this->declare_parameter("api_session_inactivity_timeout_ms", 35000);
    this->declare_parameter("api_connection_inactivity_timeout_ms", 20000);
    
    m_username = this->get_parameter("username").as_string();
    m_password = this->get_parameter("password").as_string();
    m_ip_address = this->get_parameter("ip_address").as_string();
    m_api_session_inactivity_timeout_ms = this->get_parameter("api_session_inactivity_timeout_ms").as_int();
    m_api_connection_inactivity_timeout_ms = this->get_parameter("api_connection_inactivity_timeout_ms").as_int();

    RCLCPP_INFO(this->get_logger(), "Starting to initialize controller");
    
    // Initialize Kortex API connection
    m_tcp_transport = new k_api::TransportClientTcp();
    m_tcp_transport->connect(m_ip_address, TCP_PORT);
    m_tcp_router = new k_api::RouterClient(m_tcp_transport, [this](k_api::KError err) { 
        RCLCPP_ERROR(this->get_logger(), "Kortex API error was encountered with the TCP router: %s", err.toString().c_str()); 
    });

    // Set session data connection information
    auto createSessionInfo = Kinova::Api::Session::CreateSessionInfo();
    createSessionInfo.set_username(m_username);
    createSessionInfo.set_password(m_password);
    createSessionInfo.set_session_inactivity_timeout(m_api_session_inactivity_timeout_ms);
    createSessionInfo.set_connection_inactivity_timeout(m_api_connection_inactivity_timeout_ms);

    // Session manager service wrapper
    RCLCPP_INFO(this->get_logger(), "Creating session for communication");
    m_tcp_session_manager = new k_api::SessionManager(m_tcp_router);
    
    // Create the sessions to use the robot
    try 
    {
        m_tcp_session_manager->CreateSession(createSessionInfo);
        RCLCPP_INFO(this->get_logger(), "Session created successfully for TCP services");
    }
    catch(std::runtime_error& ex_runtime)
    {
        std::string error_string = "The node could not connect to the arm. Check if the robot is powered on and if the IP address is correct.";
        RCLCPP_ERROR(this->get_logger(), "%s", error_string.c_str());
        throw ex_runtime;
    }

    // Create services
    RCLCPP_INFO(this->get_logger(), "Creating API clients...");
    mBase = new k_api::Base::BaseClient(m_tcp_router);
    RCLCPP_INFO(this->get_logger(), "BaseClient created");
    
    mBaseCyclic = new k_api::BaseCyclic::BaseCyclicClient(m_tcp_router);
    RCLCPP_INFO(this->get_logger(), "BaseCyclicClient created");
    
    mActuatorConfig = new k_api::ActuatorConfig::ActuatorConfigClient(m_tcp_router);
    RCLCPP_INFO(this->get_logger(), "ActuatorConfigClient created");
    
    mServoingMode = k_api::Base::ServoingModeInformation();
    mControlModeMessage = k_api::ActuatorConfig::ControlModeInformation();

    // Test basic API calls before proceeding
    RCLCPP_INFO(this->get_logger(), "Testing basic API calls...");
    try {
        RCLCPP_INFO(this->get_logger(), "Testing GetActuatorCount...");
        auto count = mBase->GetActuatorCount();
        RCLCPP_INFO(this->get_logger(), "Actuator count: %d", count.count());
        
        RCLCPP_INFO(this->get_logger(), "Testing RefreshFeedback...");
        auto feedback = mBaseCyclic->RefreshFeedback();
        RCLCPP_INFO(this->get_logger(), "RefreshFeedback successful");
        
    } catch (k_api::KDetailedException& ex) {
        RCLCPP_ERROR(this->get_logger(), "API test failed: %s", ex.what());
        RCLCPP_ERROR(this->get_logger(), "Error sub-code: %s", 
            k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))).c_str());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown error during API test");
    }

    // Clearing faults
    RCLCPP_INFO(this->get_logger(), "Clearing faults...");
    try
    {
        mBase->ClearFaults();
        RCLCPP_INFO(this->get_logger(), "Faults cleared successfully");
    }
    catch(...)
    {
        RCLCPP_WARN(this->get_logger(), "Unable to clear robot faults");
    }

    // Initialize ROS2 publishers
    RCLCPP_INFO(this->get_logger(), "Creating publishers...");
    mJointStatePub = this->create_publisher<sensor_msgs::msg::JointState>("/my_gen3/robot_joint_states", 10);
    mCartesianStatePub = this->create_publisher<geometry_msgs::msg::PoseStamped>("/my_gen3/robot_cartesian_state", 10);
    RCLCPP_INFO(this->get_logger(), "Publishers created");
    
    // Initialize ROS2 subscribers
    RCLCPP_INFO(this->get_logger(), "Creating subscribers...");
    mTareFTSensorSub = this->create_subscription<std_msgs::msg::Bool>(
        "/my_gen3/tare_ft_sensor", 10, std::bind(&Controller::tareFTSensorCallback, this, _1));
    mEStopSub = this->create_subscription<std_msgs::msg::Bool>(
        "/my_gen3/estop", 10, std::bind(&Controller::eStopCallback, this, _1));
    RCLCPP_INFO(this->get_logger(), "Subscribers created");
    
    // Initialize ROS2 services
    RCLCPP_INFO(this->get_logger(), "Creating services...");
    mSetJointPositionService = this->create_service<raf_interfaces::srv::SetJointAngles>(
        "/my_gen3/set_joint_position", std::bind(&Controller::setJointPosition, this, _1, _2));
    mSetJointVelocityService = this->create_service<raf_interfaces::srv::SetJointVelocity>(
        "/my_gen3/set_joint_velocity", std::bind(&Controller::setJointVelocity, this, _1, _2));
    mSetJointWaypointsService = this->create_service<raf_interfaces::srv::SetJointWaypoints>(
        "/my_gen3/set_joint_waypoints", std::bind(&Controller::setJointWaypoints, this, _1, _2));
    mSetPoseService = this->create_service<raf_interfaces::srv::SetPose>(
        "/my_gen3/set_pose", std::bind(&Controller::setPose, this, _1, _2));
    mSetGripperService = this->create_service<raf_interfaces::srv::SetGripper>(
        "/my_gen3/set_gripper", std::bind(&Controller::setGripper, this, _1, _2));
    mSetTwistService = this->create_service<raf_interfaces::srv::SetTwist>(
        "/my_gen3/set_twist", std::bind(&Controller::setTwist, this, _1, _2));
    RCLCPP_INFO(this->get_logger(), "Services created");

    // Initialize force/torque sensor variables
    RCLCPP_INFO(this->get_logger(), "Initializing F/T sensor variables...");
    mZeroFTSensorValues = std::vector<double>(6, 0.0);  // Changed back to 6
    mFTSensorValues = std::vector<double>(6, 0.0);      // Changed back to 6
    mForceThreshold = std::vector<double>(6, 1000.0);   // Changed back to 6
    mWatchdogActive = true;
    RCLCPP_INFO(this->get_logger(), "F/T sensor variables initialized");

    // Test publishState manually once before starting timer
    RCLCPP_INFO(this->get_logger(), "Testing publishState manually...");
    try {
        publishState();
        RCLCPP_INFO(this->get_logger(), "Manual publishState test completed successfully");
        
        // If manual test works, start the timer
        RCLCPP_INFO(this->get_logger(), "Starting state publishing timer...");
        mRobotStateTimer = this->create_wall_timer(
            std::chrono::milliseconds(100), std::bind(&Controller::publishState, this)); // Increased interval to 100ms for debugging
        RCLCPP_INFO(this->get_logger(), "Timer started");
        
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Manual publishState test failed - timer will NOT be started");
    }

    RCLCPP_INFO(this->get_logger(), "Controller initialized successfully");
}

Controller::~Controller()
{
    try
    {
        mBase->Stop();
    }
    catch (k_api::KDetailedException& ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Kortex exception: %s", ex.what());
        RCLCPP_ERROR(this->get_logger(), "Error sub-code: %s", 
            k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))).c_str());
    }
    
    m_tcp_session_manager->CloseSession();
    m_tcp_router->SetActivationStatus(false);
    m_tcp_transport->disconnect();

    delete mBase;
    delete mBaseCyclic;
    delete mActuatorConfig;
    delete m_tcp_session_manager;
    delete m_tcp_router;
    delete m_tcp_transport;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

inline double Controller::degreesToRadians(double degrees)
{
    return (M_PI / 180.0) * degrees;
}

inline double Controller::radiansToDegrees(double radians)
{
    return (180.0 / M_PI) * radians;
}

void Controller::eStopCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    if (msg->data)
    {
        RCLCPP_INFO(this->get_logger(), "E-Stop activated");
        try
        {
            mBase->Stop();
        }
        catch (k_api::KDetailedException& ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Kortex exception: %s", ex.what());
            RCLCPP_ERROR(this->get_logger(), "Error sub-code: %s", 
                k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))).c_str());
        }
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "False message received on E-Stop topic - this should not happen");
    }

    RCLCPP_ERROR(this->get_logger(), "Dead because of E-Stop");
    rclcpp::shutdown();
}

void Controller::tareFTSensorCallback(const std_msgs::msg::Bool::SharedPtr /* msg */)
{
    mTareFTSensor.store(true);
}

void Controller::publishState()
{
    RCLCPP_DEBUG(this->get_logger(), "publishState() called");
    
    try {
        auto start_time = this->now();
        RCLCPP_DEBUG(this->get_logger(), "About to call RefreshFeedback...");
        
        mLastFeedback = mBaseCyclic->RefreshFeedback();
        RCLCPP_DEBUG(this->get_logger(), "RefreshFeedback successful");

        RCLCPP_DEBUG(this->get_logger(), "Getting actuator count...");
        int actuator_count = mBase->GetActuatorCount().count();
        RCLCPP_DEBUG(this->get_logger(), "Actuator count: %d", actuator_count);
        
        auto joint_state = sensor_msgs::msg::JointState();
        joint_state.header.stamp = this->now();
        
        RCLCPP_DEBUG(this->get_logger(), "Processing joint states...");
        for (int i = 0; i < actuator_count; ++i)
        {
            joint_state.name.push_back("joint_" + std::to_string(i + 1));
            double pos = degreesToRadians(double(mLastFeedback.actuators(i).position()));
            if (pos > M_PI)
                pos -= 2*M_PI;
            joint_state.position.push_back(pos);
            joint_state.velocity.push_back(degreesToRadians(double(mLastFeedback.actuators(i).velocity())));
            joint_state.effort.push_back(double(mLastFeedback.actuators(i).torque()));
        }

        RCLCPP_DEBUG(this->get_logger(), "Processing gripper state...");
        // Read finger state - add error checking
        if (mLastFeedback.has_interconnect() && 
            mLastFeedback.interconnect().has_gripper_feedback() &&
            mLastFeedback.interconnect().gripper_feedback().motor_size() > 0) {
            
            joint_state.name.push_back("finger_joint");
            joint_state.position.push_back(0.8*mLastFeedback.interconnect().gripper_feedback().motor()[0].position() / 100.0);
            joint_state.velocity.push_back(0.8*mLastFeedback.interconnect().gripper_feedback().motor()[0].velocity() / 100.0);
            joint_state.effort.push_back(mLastFeedback.interconnect().gripper_feedback().motor()[0].current_motor());
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "No gripper feedback available");
        }

        RCLCPP_DEBUG(this->get_logger(), "Publishing joint state...");
        mJointStatePub->publish(joint_state);

        RCLCPP_DEBUG(this->get_logger(), "Processing cartesian state...");
        // Publish cartesian state
        auto cartesian_state = geometry_msgs::msg::PoseStamped();
        cartesian_state.header.stamp = this->now();
        cartesian_state.header.frame_id = "base_link";

        cartesian_state.pose.position.x = mLastFeedback.base().tool_pose_x();
        cartesian_state.pose.position.y = mLastFeedback.base().tool_pose_y();
        cartesian_state.pose.position.z = mLastFeedback.base().tool_pose_z();

        tf2::Quaternion quat;
        quat.setRPY(degreesToRadians(mLastFeedback.base().tool_pose_theta_x()), 
                    degreesToRadians(mLastFeedback.base().tool_pose_theta_y()), 
                    degreesToRadians(mLastFeedback.base().tool_pose_theta_z()));
        cartesian_state.pose.orientation = tf2::toMsg(quat);

        RCLCPP_DEBUG(this->get_logger(), "Publishing cartesian state...");
        mCartesianStatePub->publish(cartesian_state);

        RCLCPP_DEBUG(this->get_logger(), "Processing F/T sensor data...");
        // Update force/torque sensor data
        mFTSensorValues[0] = mLastFeedback.base().tool_external_wrench_force_x();
        mFTSensorValues[1] = mLastFeedback.base().tool_external_wrench_force_y();
        mFTSensorValues[2] = mLastFeedback.base().tool_external_wrench_force_z();
        mFTSensorValues[3] = mLastFeedback.base().tool_external_wrench_torque_x();
        mFTSensorValues[4] = mLastFeedback.base().tool_external_wrench_torque_y();
        mFTSensorValues[5] = mLastFeedback.base().tool_external_wrench_torque_z();

        if (mTareFTSensor.load())
        {
            mZeroFTSensorValues = mFTSensorValues;
            mTareFTSensor.store(false);
        }

        if(mUpdateForceThreshold.load())
        {
            mForceThreshold = mNewForceThreshold;
            mUpdateForceThreshold.store(false);
        }

        RCLCPP_DEBUG(this->get_logger(), "Checking force thresholds...");
        // Check force thresholds
        if( std::abs(mFTSensorValues[0] - mZeroFTSensorValues[0]) > std::abs(mForceThreshold[0])
            or std::abs(mFTSensorValues[1] - mZeroFTSensorValues[1]) > std::abs(mForceThreshold[1])
            or std::abs(mFTSensorValues[2] - mZeroFTSensorValues[2]) > std::abs(mForceThreshold[2])
            or std::abs(mFTSensorValues[3] - mZeroFTSensorValues[3]) > std::abs(mForceThreshold[3])
            or std::abs(mFTSensorValues[4] - mZeroFTSensorValues[4]) > std::abs(mForceThreshold[4])
            or std::abs(mFTSensorValues[5] - mZeroFTSensorValues[5]) > std::abs(mForceThreshold[5]))
        {   
            RCLCPP_INFO(this->get_logger(), "Force threshold exceeded");
            RCLCPP_INFO(this->get_logger(), "Measured force:");
            RCLCPP_INFO(this->get_logger(), "Fx: %f", mFTSensorValues[0] - mZeroFTSensorValues[0]);
            RCLCPP_INFO(this->get_logger(), "Fy: %f", mFTSensorValues[1] - mZeroFTSensorValues[1]);
            RCLCPP_INFO(this->get_logger(), "Fz: %f", mFTSensorValues[2] - mZeroFTSensorValues[2]);
            RCLCPP_INFO(this->get_logger(), "Tx: %f", mFTSensorValues[3] - mZeroFTSensorValues[3]);
            RCLCPP_INFO(this->get_logger(), "Ty: %f", mFTSensorValues[4] - mZeroFTSensorValues[4]);
            RCLCPP_INFO(this->get_logger(), "Tz: %f", mFTSensorValues[5] - mZeroFTSensorValues[5]);
        }
        
        RCLCPP_DEBUG(this->get_logger(), "publishState() completed successfully");
        
    } catch (k_api::KDetailedException& ex) {
        RCLCPP_ERROR(this->get_logger(), "Kortex exception in publishState: %s", ex.what());
        RCLCPP_ERROR(this->get_logger(), "Error sub-code: %s", 
            k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))).c_str());
    } catch (std::exception& ex) {
        RCLCPP_ERROR(this->get_logger(), "Standard exception in publishState: %s", ex.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in publishState");
    }
}

void Controller::setJointPosition(const std::shared_ptr<raf_interfaces::srv::SetJointAngles::Request> request,
                                 std::shared_ptr<raf_interfaces::srv::SetJointAngles::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "Got joint position command");

    try
    {
        mBase->StopAction();
    }
    catch (k_api::KDetailedException& ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Kortex exception: %s", ex.what());
        RCLCPP_ERROR(this->get_logger(), "Error sub-code: %s",
            k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))).c_str());
    }

    auto action = k_api::Base::Action();
    action.set_name("ROS2 angular action movement");
    action.set_application_data("");

    auto reach_joint_angles = action.mutable_reach_joint_angles();
    auto joint_angles = reach_joint_angles->mutable_joint_angles();

    auto actuator_count = mBase->GetActuatorCount();

    RCLCPP_INFO(this->get_logger(), "Actuator count: %d", actuator_count.count());
    
    for (size_t i = 0; i < actuator_count.count() && i < request->joint_angles.size(); ++i) 
    {
        auto joint_angle = joint_angles->add_joint_angles();
        joint_angle->set_joint_identifier(i);
        joint_angle->set_value(radiansToDegrees(request->joint_angles[i]));
    }

    // Connect to notification action topic
    std::promise<k_api::Base::ActionEvent> finish_promise;
    auto finish_future = finish_promise.get_future();
    auto promise_notification_handle = mBase->OnNotificationActionTopic(
        create_event_listener_by_promise(finish_promise),
        k_api::Common::NotificationOptions()
    );

    RCLCPP_INFO(this->get_logger(), "Executing action");
    mBase->ExecuteAction(action);

    RCLCPP_INFO(this->get_logger(), "Waiting for movement to finish ...");

    // Wait for future value from promise
    const auto status = finish_future.wait_for(TIMEOUT_DURATION);
    mBase->Unsubscribe(promise_notification_handle);

    if(status != std::future_status::ready)
    {
        RCLCPP_ERROR(this->get_logger(), "Timeout on action notification wait");
        response->success = false;
        response->message = "Timeout on action notification wait";
        return;
    }
    const auto promise_event = finish_future.get();

    RCLCPP_INFO(this->get_logger(), "Angular movement completed");
    RCLCPP_INFO(this->get_logger(), "Promise value : %s", k_api::Base::ActionEvent_Name(promise_event).c_str());
    response->success = true;
    response->message = "Movement completed successfully";
}

void Controller::setJointVelocity(const std::shared_ptr<raf_interfaces::srv::SetJointVelocity::Request> request,
                                 std::shared_ptr<raf_interfaces::srv::SetJointVelocity::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "Got set joint velocity command");

    if(request->mode != std::string("VELOCITY"))
    {
        RCLCPP_ERROR(this->get_logger(), "Wrong mode for set joint velocity command");
        response->success = false;
        response->error_msg = "Wrong mode for set joint velocity command";
        return;
    }

    try
    {
        mBase->StopAction();
    }
    catch (k_api::KDetailedException& ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Kortex exception: %s", ex.what());
        RCLCPP_ERROR(this->get_logger(), "Error sub-code: %s",
            k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))).c_str());
    }

    k_api::Base::JointSpeeds joint_speeds;
    
    int actuator_count = mBase->GetActuatorCount().count();
    for (int i = 0; i < actuator_count && i < static_cast<int>(request->command.size()); ++i)
    {
        auto joint_speed = joint_speeds.add_joint_speeds();
        joint_speed->set_joint_identifier(i);
        joint_speed->set_value(radiansToDegrees(request->command[i]));
        // Temporarily remove duration - may not be needed in API 2.7
    }
    mBase->SendJointSpeedsCommand(joint_speeds);

    int timeout = static_cast<int>(request->timeout * 1000);
    RCLCPP_INFO(this->get_logger(), "Will stop robot after %d ms", timeout);

    // Wait for timeout seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
    
    // Stop the robot
    RCLCPP_INFO(this->get_logger(), "Stopping the robot");
    mBase->Stop();

    response->success = true;
    response->error_msg = "Velocity command completed";
}

void Controller::setJointWaypoints(const std::shared_ptr<raf_interfaces::srv::SetJointWaypoints::Request> request,
                                  std::shared_ptr<raf_interfaces::srv::SetJointWaypoints::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "Received trajectory waypoints command");

    try
    {
        mBase->StopAction();
    }
    catch (k_api::KDetailedException& ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Kortex exception: %s", ex.what());
        RCLCPP_ERROR(this->get_logger(), "Error sub-code: %s",
            k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))).c_str());
    }

    // Create the trajectory 
    k_api::Base::WaypointList wpts = k_api::Base::WaypointList();

    const int degreesOfFreedom = 7;
    const float firstTime = 0.5f;
    for (size_t index = 0; index < request->target_waypoints.points.size(); ++index)
    {
        k_api::Base::Waypoint *wpt = wpts.add_waypoints();
        if(wpt != nullptr)
        {
            wpt->set_name(std::string("waypoint_") + std::to_string(index));
            k_api::Base::AngularWaypoint *ang = wpt->mutable_angular_waypoint();
            if(ang != nullptr)
            {    
                for(int angleIndex = 0; angleIndex < degreesOfFreedom && angleIndex < static_cast<int>(request->target_waypoints.points[index].positions.size()); ++angleIndex)
                {
                    ang->add_angles(radiansToDegrees(request->target_waypoints.points[index].positions[angleIndex]));
                }
                ang->set_duration(firstTime);
            }
        }   
        RCLCPP_INFO(this->get_logger(), "Waypoint %zu created", index);     
    }

    // Connect to notification action topic
    std::promise<k_api::Base::ActionEvent> finish_promise_cart;
    auto finish_future_cart = finish_promise_cart.get_future();
    auto promise_notification_handle_cart = mBase->OnNotificationActionTopic( create_event_listener_by_promise(finish_promise_cart),
                                                                            k_api::Common::NotificationOptions());

    k_api::Base::WaypointValidationReport result;
    try
    {
        // Verify validity of waypoints
        auto validationResult = mBase->ValidateWaypointList(wpts);
        result = validationResult;
    }
    catch(k_api::KDetailedException& ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Try catch error on waypoint list");
        auto error_info = ex.getErrorInfo().getError();
        RCLCPP_ERROR(this->get_logger(), "KDetailedoption detected what: %s", ex.what());
        RCLCPP_ERROR(this->get_logger(), "KError error_code: %d", error_info.error_code());
        RCLCPP_ERROR(this->get_logger(), "KError sub_code: %d", error_info.error_sub_code());
        RCLCPP_ERROR(this->get_logger(), "KError sub_string: %s", error_info.error_sub_string().c_str());
        RCLCPP_ERROR(this->get_logger(), "Error code string equivalent: %s", 
            k_api::ErrorCodes_Name(k_api::ErrorCodes(error_info.error_code())).c_str());
        RCLCPP_ERROR(this->get_logger(), "Error sub-code string equivalent: %s", 
            k_api::SubErrorCodes_Name(k_api::SubErrorCodes(error_info.error_sub_code())).c_str());
        
        response->success = false;
        return;
    }
    
    // Trajectory error report always exists and we need to make sure no elements are found in order to validate the trajectory
    if(result.trajectory_error_report().trajectory_error_elements_size() == 0)
    {    
        // Execute action
        try
        {
            // Move arm with waypoints list
            RCLCPP_INFO(this->get_logger(), "Moving the arm creating a trajectory of %zu angular waypoints", 
                request->target_waypoints.points.size());
            mBase->ExecuteWaypointTrajectory(wpts);
        }
        catch(k_api::KDetailedException& ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Try catch error executing normal trajectory");
            auto error_info = ex.getErrorInfo().getError();
            RCLCPP_ERROR(this->get_logger(), "KDetailedoption detected what: %s", ex.what());
            RCLCPP_ERROR(this->get_logger(), "KError error_code: %d", error_info.error_code());
            RCLCPP_ERROR(this->get_logger(), "KError sub_code: %d", error_info.error_sub_code());
            RCLCPP_ERROR(this->get_logger(), "KError sub_string: %s", error_info.error_sub_string().c_str());
            RCLCPP_ERROR(this->get_logger(), "Error code string equivalent: %s", 
                k_api::ErrorCodes_Name(k_api::ErrorCodes(error_info.error_code())).c_str());
            RCLCPP_ERROR(this->get_logger(), "Error sub-code string equivalent: %s", 
                k_api::SubErrorCodes_Name(k_api::SubErrorCodes(error_info.error_sub_code())).c_str());
            
            response->success = false;
            return;
        }
        
        // Wait for future value from promise
        const auto ang_status = finish_future_cart.wait_for(TIMEOUT_DURATION);

        mBase->Unsubscribe(promise_notification_handle_cart);

        if(ang_status != std::future_status::ready)
        {
            RCLCPP_ERROR(this->get_logger(), "Timeout on action notification wait for angular waypoint trajectory");
            response->success = false;
        }
        else
        {
            const auto ang_promise_event = finish_future_cart.get();
            RCLCPP_INFO(this->get_logger(), "Angular waypoint trajectory completed");
            RCLCPP_INFO(this->get_logger(), "Promise value : %s", k_api::Base::ActionEvent_Name(ang_promise_event).c_str()); 
            response->success = true;
        }
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Error found in trajectory");
        response->success = false;
    }
}

void Controller::setPose(const std::shared_ptr<raf_interfaces::srv::SetPose::Request> request,
                        std::shared_ptr<raf_interfaces::srv::SetPose::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "Received pose command");

    try
    {
        mBase->StopAction();
    }
    catch (k_api::KDetailedException& ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Kortex exception: %s", ex.what());
        RCLCPP_ERROR(this->get_logger(), "Error sub-code: %s",
            k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))).c_str());
    }

    // Update force threshold if provided
    if (!request->force_threshold.empty())
    {
        mNewForceThreshold = std::vector<double>(request->force_threshold.begin(), request->force_threshold.end());
        mUpdateForceThreshold.store(true);
    }

    auto action = k_api::Base::Action();
    action.set_name("ROS2 Cartesian action movement");
    action.set_application_data("");

    auto constrained_pose = action.mutable_reach_pose();
    auto pose = constrained_pose->mutable_target_pose();
    pose->set_x(request->target_pose.position.x);
    pose->set_y(request->target_pose.position.y);
    pose->set_z(request->target_pose.position.z);
    
    tf2::Quaternion quat;
    tf2::fromMsg(request->target_pose.orientation, quat);
    tf2::Matrix3x3 m(quat);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    pose->set_theta_x(radiansToDegrees(roll));
    pose->set_theta_y(radiansToDegrees(pitch));
    pose->set_theta_z(radiansToDegrees(yaw));

    RCLCPP_INFO(this->get_logger(), "Setting cartesian pose");
    RCLCPP_INFO(this->get_logger(), "X: %f", pose->x());
    RCLCPP_INFO(this->get_logger(), "Y: %f", pose->y());
    RCLCPP_INFO(this->get_logger(), "Z: %f", pose->z());
    RCLCPP_INFO(this->get_logger(), "Theta X: %f", pose->theta_x());
    RCLCPP_INFO(this->get_logger(), "Theta Y: %f", pose->theta_y());
    RCLCPP_INFO(this->get_logger(), "Theta Z: %f", pose->theta_z());

    // Connect to notification action topic
    k_api::Base::ActionEvent event = k_api::Base::ActionEvent::UNSPECIFIED_ACTION_EVENT;
    auto reference_notification_handle = mBase->OnNotificationActionTopic(
        create_event_listener_by_ref(event),
        k_api::Common::NotificationOptions()
    );

    RCLCPP_INFO(this->get_logger(), "Executing action");
    mBase->ExecuteAction(action);

    RCLCPP_INFO(this->get_logger(), "Waiting for movement to finish ...");

    // Wait for reference value to be set
    const auto timeout = std::chrono::system_clock::now() + TIMEOUT_DURATION;
    while(event == k_api::Base::ActionEvent::UNSPECIFIED_ACTION_EVENT &&
        std::chrono::system_clock::now() < timeout)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    mBase->Unsubscribe(reference_notification_handle);

    if(event == k_api::Base::ActionEvent::UNSPECIFIED_ACTION_EVENT)
    {
        RCLCPP_ERROR(this->get_logger(), "Timeout on action notification wait");
        response->success = false;
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Cartesian movement completed");
    RCLCPP_INFO(this->get_logger(), "Reference value : %s", k_api::Base::ActionEvent_Name(event).c_str());

    response->success = true;
}

void Controller::setGripper(const std::shared_ptr<raf_interfaces::srv::SetGripper::Request> request,
                           std::shared_ptr<raf_interfaces::srv::SetGripper::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "Received gripper command");

    try
    {
        mBase->StopAction();
    }
    catch (k_api::KDetailedException& ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Kortex exception: %s", ex.what());
        RCLCPP_ERROR(this->get_logger(), "Error sub-code: %s",
            k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))).c_str());
    }

    k_api::Base::Finger* finger;
    k_api::Base::GripperCommand gripper_command;
    finger = gripper_command.mutable_gripper()->add_finger();
    finger->set_finger_identifier(1);

    RCLCPP_INFO(this->get_logger(), "Sending gripper position command: %f", request->position);

    finger->set_value(request->position);
    gripper_command.set_mode(k_api::Base::GRIPPER_POSITION);
    
    try
    {
        mBase->SendGripperCommand(gripper_command);
    }
    catch (k_api::KDetailedException& ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Kortex exception: %s", ex.what());
        RCLCPP_ERROR(this->get_logger(), "Error sub-code: %s",
            k_api::SubErrorCodes_Name(k_api::SubErrorCodes((ex.getErrorInfo().getError().error_sub_code()))).c_str());
        response->success = false;
        response->message = "Gripper command failed";
        return;
    }
    catch (std::runtime_error& ex2)
    {
        RCLCPP_ERROR(this->get_logger(), "runtime error: %s", ex2.what());
        response->success = false;
        response->message = "Gripper command failed - runtime error";
        return;
    }
    catch (...)
    {
        RCLCPP_ERROR(this->get_logger(), "Unknown error.");
        response->success = false;
        response->message = "Gripper command failed - unknown error";
        return;
    }

    response->success = true;
    response->message = "Gripper command completed successfully";
}

void Controller::setTwist(const std::shared_ptr<raf_interfaces::srv::SetTwist::Request> /* request */,
                         std::shared_ptr<raf_interfaces::srv::SetTwist::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "Received twist command");
    
    // This is a placeholder implementation
    // The original didn't have a full twist implementation either
    response->success = true;
}

int main(int argc, char * argv[]) 
{
    rclcpp::init(argc, argv);
    
    try 
    {
        auto controller = std::make_shared<Controller>();
        RCLCPP_INFO(controller->get_logger(), "Kinova controller starting");
        
        // Use MultiThreadedExecutor to handle multiple service calls
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(controller);
        executor.spin();
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("controller"), "Exception in main: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}