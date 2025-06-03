// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "raf_interfaces/msg/detail/detected_food__rosidl_typesupport_introspection_c.h"
#include "raf_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "raf_interfaces/msg/detail/detected_food__functions.h"
#include "raf_interfaces/msg/detail/detected_food__struct.h"


// Include directives for member types
// Member `food_items`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  raf_interfaces__msg__DetectedFood__init(message_memory);
}

void raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_fini_function(void * message_memory)
{
  raf_interfaces__msg__DetectedFood__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_message_member_array[1] = {
  {
    "food_items",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(raf_interfaces__msg__DetectedFood, food_items),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_message_members = {
  "raf_interfaces__msg",  // message namespace
  "DetectedFood",  // message name
  1,  // number of fields
  sizeof(raf_interfaces__msg__DetectedFood),
  false,  // has_any_key_member_
  raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_message_member_array,  // message members
  raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_init_function,  // function to initialize message memory (memory has to be allocated)
  raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_message_type_support_handle = {
  0,
  &raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_message_members,
  get_message_typesupport_handle_function,
  &raf_interfaces__msg__DetectedFood__get_type_hash,
  &raf_interfaces__msg__DetectedFood__get_type_description,
  &raf_interfaces__msg__DetectedFood__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_raf_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, msg, DetectedFood)() {
  if (!raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_message_type_support_handle.typesupport_identifier) {
    raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &raf_interfaces__msg__DetectedFood__rosidl_typesupport_introspection_c__DetectedFood_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
