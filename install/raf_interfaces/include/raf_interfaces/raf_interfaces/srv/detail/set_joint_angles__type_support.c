// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from raf_interfaces:srv/SetJointAngles.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "raf_interfaces/srv/detail/set_joint_angles__rosidl_typesupport_introspection_c.h"
#include "raf_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "raf_interfaces/srv/detail/set_joint_angles__functions.h"
#include "raf_interfaces/srv/detail/set_joint_angles__struct.h"


// Include directives for member types
// Member `joint_angles`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  raf_interfaces__srv__SetJointAngles_Request__init(message_memory);
}

void raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_fini_function(void * message_memory)
{
  raf_interfaces__srv__SetJointAngles_Request__fini(message_memory);
}

size_t raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__size_function__SetJointAngles_Request__joint_angles(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__get_const_function__SetJointAngles_Request__joint_angles(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__get_function__SetJointAngles_Request__joint_angles(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__fetch_function__SetJointAngles_Request__joint_angles(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__get_const_function__SetJointAngles_Request__joint_angles(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__assign_function__SetJointAngles_Request__joint_angles(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__get_function__SetJointAngles_Request__joint_angles(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__resize_function__SetJointAngles_Request__joint_angles(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_message_member_array[1] = {
  {
    "joint_angles",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(raf_interfaces__srv__SetJointAngles_Request, joint_angles),  // bytes offset in struct
    NULL,  // default value
    raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__size_function__SetJointAngles_Request__joint_angles,  // size() function pointer
    raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__get_const_function__SetJointAngles_Request__joint_angles,  // get_const(index) function pointer
    raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__get_function__SetJointAngles_Request__joint_angles,  // get(index) function pointer
    raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__fetch_function__SetJointAngles_Request__joint_angles,  // fetch(index, &value) function pointer
    raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__assign_function__SetJointAngles_Request__joint_angles,  // assign(index, value) function pointer
    raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__resize_function__SetJointAngles_Request__joint_angles  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_message_members = {
  "raf_interfaces__srv",  // message namespace
  "SetJointAngles_Request",  // message name
  1,  // number of fields
  sizeof(raf_interfaces__srv__SetJointAngles_Request),
  false,  // has_any_key_member_
  raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_message_member_array,  // message members
  raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_message_type_support_handle = {
  0,
  &raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_message_members,
  get_message_typesupport_handle_function,
  &raf_interfaces__srv__SetJointAngles_Request__get_type_hash,
  &raf_interfaces__srv__SetJointAngles_Request__get_type_description,
  &raf_interfaces__srv__SetJointAngles_Request__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_raf_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Request)() {
  if (!raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_message_type_support_handle.typesupport_identifier) {
    raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__rosidl_typesupport_introspection_c.h"
// already included above
// #include "raf_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__functions.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__struct.h"


// Include directives for member types
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  raf_interfaces__srv__SetJointAngles_Response__init(message_memory);
}

void raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_fini_function(void * message_memory)
{
  raf_interfaces__srv__SetJointAngles_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_member_array[2] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(raf_interfaces__srv__SetJointAngles_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "message",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(raf_interfaces__srv__SetJointAngles_Response, message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_members = {
  "raf_interfaces__srv",  // message namespace
  "SetJointAngles_Response",  // message name
  2,  // number of fields
  sizeof(raf_interfaces__srv__SetJointAngles_Response),
  false,  // has_any_key_member_
  raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_member_array,  // message members
  raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_type_support_handle = {
  0,
  &raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_members,
  get_message_typesupport_handle_function,
  &raf_interfaces__srv__SetJointAngles_Response__get_type_hash,
  &raf_interfaces__srv__SetJointAngles_Response__get_type_description,
  &raf_interfaces__srv__SetJointAngles_Response__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_raf_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Response)() {
  if (!raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_type_support_handle.typesupport_identifier) {
    raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__rosidl_typesupport_introspection_c.h"
// already included above
// #include "raf_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__functions.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__struct.h"


// Include directives for member types
// Member `info`
#include "service_msgs/msg/service_event_info.h"
// Member `info`
#include "service_msgs/msg/detail/service_event_info__rosidl_typesupport_introspection_c.h"
// Member `request`
// Member `response`
#include "raf_interfaces/srv/set_joint_angles.h"
// Member `request`
// Member `response`
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  raf_interfaces__srv__SetJointAngles_Event__init(message_memory);
}

void raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_fini_function(void * message_memory)
{
  raf_interfaces__srv__SetJointAngles_Event__fini(message_memory);
}

size_t raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__size_function__SetJointAngles_Event__request(
  const void * untyped_member)
{
  const raf_interfaces__srv__SetJointAngles_Request__Sequence * member =
    (const raf_interfaces__srv__SetJointAngles_Request__Sequence *)(untyped_member);
  return member->size;
}

const void * raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_const_function__SetJointAngles_Event__request(
  const void * untyped_member, size_t index)
{
  const raf_interfaces__srv__SetJointAngles_Request__Sequence * member =
    (const raf_interfaces__srv__SetJointAngles_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void * raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_function__SetJointAngles_Event__request(
  void * untyped_member, size_t index)
{
  raf_interfaces__srv__SetJointAngles_Request__Sequence * member =
    (raf_interfaces__srv__SetJointAngles_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__fetch_function__SetJointAngles_Event__request(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const raf_interfaces__srv__SetJointAngles_Request * item =
    ((const raf_interfaces__srv__SetJointAngles_Request *)
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_const_function__SetJointAngles_Event__request(untyped_member, index));
  raf_interfaces__srv__SetJointAngles_Request * value =
    (raf_interfaces__srv__SetJointAngles_Request *)(untyped_value);
  *value = *item;
}

void raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__assign_function__SetJointAngles_Event__request(
  void * untyped_member, size_t index, const void * untyped_value)
{
  raf_interfaces__srv__SetJointAngles_Request * item =
    ((raf_interfaces__srv__SetJointAngles_Request *)
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_function__SetJointAngles_Event__request(untyped_member, index));
  const raf_interfaces__srv__SetJointAngles_Request * value =
    (const raf_interfaces__srv__SetJointAngles_Request *)(untyped_value);
  *item = *value;
}

bool raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__resize_function__SetJointAngles_Event__request(
  void * untyped_member, size_t size)
{
  raf_interfaces__srv__SetJointAngles_Request__Sequence * member =
    (raf_interfaces__srv__SetJointAngles_Request__Sequence *)(untyped_member);
  raf_interfaces__srv__SetJointAngles_Request__Sequence__fini(member);
  return raf_interfaces__srv__SetJointAngles_Request__Sequence__init(member, size);
}

size_t raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__size_function__SetJointAngles_Event__response(
  const void * untyped_member)
{
  const raf_interfaces__srv__SetJointAngles_Response__Sequence * member =
    (const raf_interfaces__srv__SetJointAngles_Response__Sequence *)(untyped_member);
  return member->size;
}

const void * raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_const_function__SetJointAngles_Event__response(
  const void * untyped_member, size_t index)
{
  const raf_interfaces__srv__SetJointAngles_Response__Sequence * member =
    (const raf_interfaces__srv__SetJointAngles_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void * raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_function__SetJointAngles_Event__response(
  void * untyped_member, size_t index)
{
  raf_interfaces__srv__SetJointAngles_Response__Sequence * member =
    (raf_interfaces__srv__SetJointAngles_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__fetch_function__SetJointAngles_Event__response(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const raf_interfaces__srv__SetJointAngles_Response * item =
    ((const raf_interfaces__srv__SetJointAngles_Response *)
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_const_function__SetJointAngles_Event__response(untyped_member, index));
  raf_interfaces__srv__SetJointAngles_Response * value =
    (raf_interfaces__srv__SetJointAngles_Response *)(untyped_value);
  *value = *item;
}

void raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__assign_function__SetJointAngles_Event__response(
  void * untyped_member, size_t index, const void * untyped_value)
{
  raf_interfaces__srv__SetJointAngles_Response * item =
    ((raf_interfaces__srv__SetJointAngles_Response *)
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_function__SetJointAngles_Event__response(untyped_member, index));
  const raf_interfaces__srv__SetJointAngles_Response * value =
    (const raf_interfaces__srv__SetJointAngles_Response *)(untyped_value);
  *item = *value;
}

bool raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__resize_function__SetJointAngles_Event__response(
  void * untyped_member, size_t size)
{
  raf_interfaces__srv__SetJointAngles_Response__Sequence * member =
    (raf_interfaces__srv__SetJointAngles_Response__Sequence *)(untyped_member);
  raf_interfaces__srv__SetJointAngles_Response__Sequence__fini(member);
  return raf_interfaces__srv__SetJointAngles_Response__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_member_array[3] = {
  {
    "info",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(raf_interfaces__srv__SetJointAngles_Event, info),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "request",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(raf_interfaces__srv__SetJointAngles_Event, request),  // bytes offset in struct
    NULL,  // default value
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__size_function__SetJointAngles_Event__request,  // size() function pointer
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_const_function__SetJointAngles_Event__request,  // get_const(index) function pointer
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_function__SetJointAngles_Event__request,  // get(index) function pointer
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__fetch_function__SetJointAngles_Event__request,  // fetch(index, &value) function pointer
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__assign_function__SetJointAngles_Event__request,  // assign(index, value) function pointer
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__resize_function__SetJointAngles_Event__request  // resize(index) function pointer
  },
  {
    "response",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(raf_interfaces__srv__SetJointAngles_Event, response),  // bytes offset in struct
    NULL,  // default value
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__size_function__SetJointAngles_Event__response,  // size() function pointer
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_const_function__SetJointAngles_Event__response,  // get_const(index) function pointer
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__get_function__SetJointAngles_Event__response,  // get(index) function pointer
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__fetch_function__SetJointAngles_Event__response,  // fetch(index, &value) function pointer
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__assign_function__SetJointAngles_Event__response,  // assign(index, value) function pointer
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__resize_function__SetJointAngles_Event__response  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_members = {
  "raf_interfaces__srv",  // message namespace
  "SetJointAngles_Event",  // message name
  3,  // number of fields
  sizeof(raf_interfaces__srv__SetJointAngles_Event),
  false,  // has_any_key_member_
  raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_member_array,  // message members
  raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_init_function,  // function to initialize message memory (memory has to be allocated)
  raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_type_support_handle = {
  0,
  &raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_members,
  get_message_typesupport_handle_function,
  &raf_interfaces__srv__SetJointAngles_Event__get_type_hash,
  &raf_interfaces__srv__SetJointAngles_Event__get_type_description,
  &raf_interfaces__srv__SetJointAngles_Event__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_raf_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Event)() {
  raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, service_msgs, msg, ServiceEventInfo)();
  raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Request)();
  raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Response)();
  if (!raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_type_support_handle.typesupport_identifier) {
    raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "raf_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers raf_interfaces__srv__detail__set_joint_angles__rosidl_typesupport_introspection_c__SetJointAngles_service_members = {
  "raf_interfaces__srv",  // service namespace
  "SetJointAngles",  // service name
  // the following fields are initialized below on first access
  NULL,  // request message
  // raf_interfaces__srv__detail__set_joint_angles__rosidl_typesupport_introspection_c__SetJointAngles_Request_message_type_support_handle,
  NULL,  // response message
  // raf_interfaces__srv__detail__set_joint_angles__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_type_support_handle
  NULL  // event_message
  // raf_interfaces__srv__detail__set_joint_angles__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_type_support_handle
};


static rosidl_service_type_support_t raf_interfaces__srv__detail__set_joint_angles__rosidl_typesupport_introspection_c__SetJointAngles_service_type_support_handle = {
  0,
  &raf_interfaces__srv__detail__set_joint_angles__rosidl_typesupport_introspection_c__SetJointAngles_service_members,
  get_service_typesupport_handle_function,
  &raf_interfaces__srv__SetJointAngles_Request__rosidl_typesupport_introspection_c__SetJointAngles_Request_message_type_support_handle,
  &raf_interfaces__srv__SetJointAngles_Response__rosidl_typesupport_introspection_c__SetJointAngles_Response_message_type_support_handle,
  &raf_interfaces__srv__SetJointAngles_Event__rosidl_typesupport_introspection_c__SetJointAngles_Event_message_type_support_handle,
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    raf_interfaces,
    srv,
    SetJointAngles
  ),
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    raf_interfaces,
    srv,
    SetJointAngles
  ),
  &raf_interfaces__srv__SetJointAngles__get_type_hash,
  &raf_interfaces__srv__SetJointAngles__get_type_description,
  &raf_interfaces__srv__SetJointAngles__get_type_description_sources,
};

// Forward declaration of message type support functions for service members
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Request)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Response)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Event)(void);

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_raf_interfaces
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles)(void) {
  if (!raf_interfaces__srv__detail__set_joint_angles__rosidl_typesupport_introspection_c__SetJointAngles_service_type_support_handle.typesupport_identifier) {
    raf_interfaces__srv__detail__set_joint_angles__rosidl_typesupport_introspection_c__SetJointAngles_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)raf_interfaces__srv__detail__set_joint_angles__rosidl_typesupport_introspection_c__SetJointAngles_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Response)()->data;
  }
  if (!service_members->event_members_) {
    service_members->event_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Event)()->data;
  }

  return &raf_interfaces__srv__detail__set_joint_angles__rosidl_typesupport_introspection_c__SetJointAngles_service_type_support_handle;
}
