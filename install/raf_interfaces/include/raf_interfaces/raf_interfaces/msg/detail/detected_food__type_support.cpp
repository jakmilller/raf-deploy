// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "raf_interfaces/msg/detail/detected_food__functions.h"
#include "raf_interfaces/msg/detail/detected_food__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace raf_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void DetectedFood_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) raf_interfaces::msg::DetectedFood(_init);
}

void DetectedFood_fini_function(void * message_memory)
{
  auto typed_message = static_cast<raf_interfaces::msg::DetectedFood *>(message_memory);
  typed_message->~DetectedFood();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DetectedFood_message_member_array[1] = {
  {
    "food_items",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(raf_interfaces::msg::DetectedFood, food_items),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DetectedFood_message_members = {
  "raf_interfaces::msg",  // message namespace
  "DetectedFood",  // message name
  1,  // number of fields
  sizeof(raf_interfaces::msg::DetectedFood),
  false,  // has_any_key_member_
  DetectedFood_message_member_array,  // message members
  DetectedFood_init_function,  // function to initialize message memory (memory has to be allocated)
  DetectedFood_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DetectedFood_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DetectedFood_message_members,
  get_message_typesupport_handle_function,
  &raf_interfaces__msg__DetectedFood__get_type_hash,
  &raf_interfaces__msg__DetectedFood__get_type_description,
  &raf_interfaces__msg__DetectedFood__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace raf_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<raf_interfaces::msg::DetectedFood>()
{
  return &::raf_interfaces::msg::rosidl_typesupport_introspection_cpp::DetectedFood_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, raf_interfaces, msg, DetectedFood)() {
  return &::raf_interfaces::msg::rosidl_typesupport_introspection_cpp::DetectedFood_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
