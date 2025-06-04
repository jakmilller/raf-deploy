// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from raf_interfaces:srv/SetPose.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "raf_interfaces/srv/set_pose.hpp"


#ifndef RAF_INTERFACES__SRV__DETAIL__SET_POSE__TRAITS_HPP_
#define RAF_INTERFACES__SRV__DETAIL__SET_POSE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "raf_interfaces/srv/detail/set_pose__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'target_pose'
#include "geometry_msgs/msg/detail/pose__traits.hpp"

namespace raf_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetPose_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: target_pose
  {
    out << "target_pose: ";
    to_flow_style_yaml(msg.target_pose, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetPose_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: target_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_pose:\n";
    to_block_style_yaml(msg.target_pose, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetPose_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace raf_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use raf_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const raf_interfaces::srv::SetPose_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  raf_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use raf_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const raf_interfaces::srv::SetPose_Request & msg)
{
  return raf_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<raf_interfaces::srv::SetPose_Request>()
{
  return "raf_interfaces::srv::SetPose_Request";
}

template<>
inline const char * name<raf_interfaces::srv::SetPose_Request>()
{
  return "raf_interfaces/srv/SetPose_Request";
}

template<>
struct has_fixed_size<raf_interfaces::srv::SetPose_Request>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Pose>::value> {};

template<>
struct has_bounded_size<raf_interfaces::srv::SetPose_Request>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Pose>::value> {};

template<>
struct is_message<raf_interfaces::srv::SetPose_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace raf_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetPose_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetPose_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetPose_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace raf_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use raf_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const raf_interfaces::srv::SetPose_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  raf_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use raf_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const raf_interfaces::srv::SetPose_Response & msg)
{
  return raf_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<raf_interfaces::srv::SetPose_Response>()
{
  return "raf_interfaces::srv::SetPose_Response";
}

template<>
inline const char * name<raf_interfaces::srv::SetPose_Response>()
{
  return "raf_interfaces/srv/SetPose_Response";
}

template<>
struct has_fixed_size<raf_interfaces::srv::SetPose_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<raf_interfaces::srv::SetPose_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<raf_interfaces::srv::SetPose_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace raf_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const SetPose_Event & msg,
  std::ostream & out)
{
  out << "{";
  // member: info
  {
    out << "info: ";
    to_flow_style_yaml(msg.info, out);
    out << ", ";
  }

  // member: request
  {
    if (msg.request.size() == 0) {
      out << "request: []";
    } else {
      out << "request: [";
      size_t pending_items = msg.request.size();
      for (auto item : msg.request) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: response
  {
    if (msg.response.size() == 0) {
      out << "response: []";
    } else {
      out << "response: [";
      size_t pending_items = msg.response.size();
      for (auto item : msg.response) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SetPose_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "info:\n";
    to_block_style_yaml(msg.info, out, indentation + 2);
  }

  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.request.size() == 0) {
      out << "request: []\n";
    } else {
      out << "request:\n";
      for (auto item : msg.request) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.response.size() == 0) {
      out << "response: []\n";
    } else {
      out << "response:\n";
      for (auto item : msg.response) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SetPose_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace raf_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use raf_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const raf_interfaces::srv::SetPose_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  raf_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use raf_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const raf_interfaces::srv::SetPose_Event & msg)
{
  return raf_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<raf_interfaces::srv::SetPose_Event>()
{
  return "raf_interfaces::srv::SetPose_Event";
}

template<>
inline const char * name<raf_interfaces::srv::SetPose_Event>()
{
  return "raf_interfaces/srv/SetPose_Event";
}

template<>
struct has_fixed_size<raf_interfaces::srv::SetPose_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<raf_interfaces::srv::SetPose_Event>
  : std::integral_constant<bool, has_bounded_size<raf_interfaces::srv::SetPose_Request>::value && has_bounded_size<raf_interfaces::srv::SetPose_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<raf_interfaces::srv::SetPose_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<raf_interfaces::srv::SetPose>()
{
  return "raf_interfaces::srv::SetPose";
}

template<>
inline const char * name<raf_interfaces::srv::SetPose>()
{
  return "raf_interfaces/srv/SetPose";
}

template<>
struct has_fixed_size<raf_interfaces::srv::SetPose>
  : std::integral_constant<
    bool,
    has_fixed_size<raf_interfaces::srv::SetPose_Request>::value &&
    has_fixed_size<raf_interfaces::srv::SetPose_Response>::value
  >
{
};

template<>
struct has_bounded_size<raf_interfaces::srv::SetPose>
  : std::integral_constant<
    bool,
    has_bounded_size<raf_interfaces::srv::SetPose_Request>::value &&
    has_bounded_size<raf_interfaces::srv::SetPose_Response>::value
  >
{
};

template<>
struct is_service<raf_interfaces::srv::SetPose>
  : std::true_type
{
};

template<>
struct is_service_request<raf_interfaces::srv::SetPose_Request>
  : std::true_type
{
};

template<>
struct is_service_response<raf_interfaces::srv::SetPose_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // RAF_INTERFACES__SRV__DETAIL__SET_POSE__TRAITS_HPP_
