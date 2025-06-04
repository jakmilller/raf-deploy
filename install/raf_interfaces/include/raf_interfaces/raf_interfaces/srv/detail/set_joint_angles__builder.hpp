// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from raf_interfaces:srv/SetJointAngles.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "raf_interfaces/srv/set_joint_angles.hpp"


#ifndef RAF_INTERFACES__SRV__DETAIL__SET_JOINT_ANGLES__BUILDER_HPP_
#define RAF_INTERFACES__SRV__DETAIL__SET_JOINT_ANGLES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "raf_interfaces/srv/detail/set_joint_angles__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace raf_interfaces
{

namespace srv
{

namespace builder
{

class Init_SetJointAngles_Request_joint_angles
{
public:
  Init_SetJointAngles_Request_joint_angles()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::raf_interfaces::srv::SetJointAngles_Request joint_angles(::raf_interfaces::srv::SetJointAngles_Request::_joint_angles_type arg)
  {
    msg_.joint_angles = std::move(arg);
    return std::move(msg_);
  }

private:
  ::raf_interfaces::srv::SetJointAngles_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::raf_interfaces::srv::SetJointAngles_Request>()
{
  return raf_interfaces::srv::builder::Init_SetJointAngles_Request_joint_angles();
}

}  // namespace raf_interfaces


namespace raf_interfaces
{

namespace srv
{

namespace builder
{

class Init_SetJointAngles_Response_message
{
public:
  explicit Init_SetJointAngles_Response_message(::raf_interfaces::srv::SetJointAngles_Response & msg)
  : msg_(msg)
  {}
  ::raf_interfaces::srv::SetJointAngles_Response message(::raf_interfaces::srv::SetJointAngles_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::raf_interfaces::srv::SetJointAngles_Response msg_;
};

class Init_SetJointAngles_Response_success
{
public:
  Init_SetJointAngles_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetJointAngles_Response_message success(::raf_interfaces::srv::SetJointAngles_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_SetJointAngles_Response_message(msg_);
  }

private:
  ::raf_interfaces::srv::SetJointAngles_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::raf_interfaces::srv::SetJointAngles_Response>()
{
  return raf_interfaces::srv::builder::Init_SetJointAngles_Response_success();
}

}  // namespace raf_interfaces


namespace raf_interfaces
{

namespace srv
{

namespace builder
{

class Init_SetJointAngles_Event_response
{
public:
  explicit Init_SetJointAngles_Event_response(::raf_interfaces::srv::SetJointAngles_Event & msg)
  : msg_(msg)
  {}
  ::raf_interfaces::srv::SetJointAngles_Event response(::raf_interfaces::srv::SetJointAngles_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::raf_interfaces::srv::SetJointAngles_Event msg_;
};

class Init_SetJointAngles_Event_request
{
public:
  explicit Init_SetJointAngles_Event_request(::raf_interfaces::srv::SetJointAngles_Event & msg)
  : msg_(msg)
  {}
  Init_SetJointAngles_Event_response request(::raf_interfaces::srv::SetJointAngles_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_SetJointAngles_Event_response(msg_);
  }

private:
  ::raf_interfaces::srv::SetJointAngles_Event msg_;
};

class Init_SetJointAngles_Event_info
{
public:
  Init_SetJointAngles_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetJointAngles_Event_request info(::raf_interfaces::srv::SetJointAngles_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_SetJointAngles_Event_request(msg_);
  }

private:
  ::raf_interfaces::srv::SetJointAngles_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::raf_interfaces::srv::SetJointAngles_Event>()
{
  return raf_interfaces::srv::builder::Init_SetJointAngles_Event_info();
}

}  // namespace raf_interfaces

#endif  // RAF_INTERFACES__SRV__DETAIL__SET_JOINT_ANGLES__BUILDER_HPP_
