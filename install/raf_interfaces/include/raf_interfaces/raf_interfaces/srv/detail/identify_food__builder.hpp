// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from raf_interfaces:srv/IdentifyFood.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "raf_interfaces/srv/identify_food.hpp"


#ifndef RAF_INTERFACES__SRV__DETAIL__IDENTIFY_FOOD__BUILDER_HPP_
#define RAF_INTERFACES__SRV__DETAIL__IDENTIFY_FOOD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "raf_interfaces/srv/detail/identify_food__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace raf_interfaces
{

namespace srv
{

namespace builder
{

class Init_IdentifyFood_Request_frame
{
public:
  Init_IdentifyFood_Request_frame()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::raf_interfaces::srv::IdentifyFood_Request frame(::raf_interfaces::srv::IdentifyFood_Request::_frame_type arg)
  {
    msg_.frame = std::move(arg);
    return std::move(msg_);
  }

private:
  ::raf_interfaces::srv::IdentifyFood_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::raf_interfaces::srv::IdentifyFood_Request>()
{
  return raf_interfaces::srv::builder::Init_IdentifyFood_Request_frame();
}

}  // namespace raf_interfaces


namespace raf_interfaces
{

namespace srv
{

namespace builder
{

class Init_IdentifyFood_Response_success
{
public:
  explicit Init_IdentifyFood_Response_success(::raf_interfaces::srv::IdentifyFood_Response & msg)
  : msg_(msg)
  {}
  ::raf_interfaces::srv::IdentifyFood_Response success(::raf_interfaces::srv::IdentifyFood_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::raf_interfaces::srv::IdentifyFood_Response msg_;
};

class Init_IdentifyFood_Response_food_items
{
public:
  Init_IdentifyFood_Response_food_items()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_IdentifyFood_Response_success food_items(::raf_interfaces::srv::IdentifyFood_Response::_food_items_type arg)
  {
    msg_.food_items = std::move(arg);
    return Init_IdentifyFood_Response_success(msg_);
  }

private:
  ::raf_interfaces::srv::IdentifyFood_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::raf_interfaces::srv::IdentifyFood_Response>()
{
  return raf_interfaces::srv::builder::Init_IdentifyFood_Response_food_items();
}

}  // namespace raf_interfaces


namespace raf_interfaces
{

namespace srv
{

namespace builder
{

class Init_IdentifyFood_Event_response
{
public:
  explicit Init_IdentifyFood_Event_response(::raf_interfaces::srv::IdentifyFood_Event & msg)
  : msg_(msg)
  {}
  ::raf_interfaces::srv::IdentifyFood_Event response(::raf_interfaces::srv::IdentifyFood_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::raf_interfaces::srv::IdentifyFood_Event msg_;
};

class Init_IdentifyFood_Event_request
{
public:
  explicit Init_IdentifyFood_Event_request(::raf_interfaces::srv::IdentifyFood_Event & msg)
  : msg_(msg)
  {}
  Init_IdentifyFood_Event_response request(::raf_interfaces::srv::IdentifyFood_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_IdentifyFood_Event_response(msg_);
  }

private:
  ::raf_interfaces::srv::IdentifyFood_Event msg_;
};

class Init_IdentifyFood_Event_info
{
public:
  Init_IdentifyFood_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_IdentifyFood_Event_request info(::raf_interfaces::srv::IdentifyFood_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_IdentifyFood_Event_request(msg_);
  }

private:
  ::raf_interfaces::srv::IdentifyFood_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::raf_interfaces::srv::IdentifyFood_Event>()
{
  return raf_interfaces::srv::builder::Init_IdentifyFood_Event_info();
}

}  // namespace raf_interfaces

#endif  // RAF_INTERFACES__SRV__DETAIL__IDENTIFY_FOOD__BUILDER_HPP_
