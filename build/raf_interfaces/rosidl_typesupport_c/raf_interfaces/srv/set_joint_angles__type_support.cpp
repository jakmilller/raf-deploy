// generated from rosidl_typesupport_c/resource/idl__type_support.cpp.em
// with input from raf_interfaces:srv/SetJointAngles.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "raf_interfaces/srv/detail/set_joint_angles__struct.h"
#include "raf_interfaces/srv/detail/set_joint_angles__type_support.h"
#include "raf_interfaces/srv/detail/set_joint_angles__functions.h"
#include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/message_type_support_dispatch.h"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_c/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace raf_interfaces
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _SetJointAngles_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SetJointAngles_Request_type_support_ids_t;

static const _SetJointAngles_Request_type_support_ids_t _SetJointAngles_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _SetJointAngles_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _SetJointAngles_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _SetJointAngles_Request_type_support_symbol_names_t _SetJointAngles_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, raf_interfaces, srv, SetJointAngles_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Request)),
  }
};

typedef struct _SetJointAngles_Request_type_support_data_t
{
  void * data[2];
} _SetJointAngles_Request_type_support_data_t;

static _SetJointAngles_Request_type_support_data_t _SetJointAngles_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _SetJointAngles_Request_message_typesupport_map = {
  2,
  "raf_interfaces",
  &_SetJointAngles_Request_message_typesupport_ids.typesupport_identifier[0],
  &_SetJointAngles_Request_message_typesupport_symbol_names.symbol_name[0],
  &_SetJointAngles_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t SetJointAngles_Request_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SetJointAngles_Request_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &raf_interfaces__srv__SetJointAngles_Request__get_type_hash,
  &raf_interfaces__srv__SetJointAngles_Request__get_type_description,
  &raf_interfaces__srv__SetJointAngles_Request__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace raf_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, raf_interfaces, srv, SetJointAngles_Request)() {
  return &::raf_interfaces::srv::rosidl_typesupport_c::SetJointAngles_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__struct.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__type_support.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace raf_interfaces
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _SetJointAngles_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SetJointAngles_Response_type_support_ids_t;

static const _SetJointAngles_Response_type_support_ids_t _SetJointAngles_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _SetJointAngles_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _SetJointAngles_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _SetJointAngles_Response_type_support_symbol_names_t _SetJointAngles_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, raf_interfaces, srv, SetJointAngles_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Response)),
  }
};

typedef struct _SetJointAngles_Response_type_support_data_t
{
  void * data[2];
} _SetJointAngles_Response_type_support_data_t;

static _SetJointAngles_Response_type_support_data_t _SetJointAngles_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _SetJointAngles_Response_message_typesupport_map = {
  2,
  "raf_interfaces",
  &_SetJointAngles_Response_message_typesupport_ids.typesupport_identifier[0],
  &_SetJointAngles_Response_message_typesupport_symbol_names.symbol_name[0],
  &_SetJointAngles_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t SetJointAngles_Response_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SetJointAngles_Response_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &raf_interfaces__srv__SetJointAngles_Response__get_type_hash,
  &raf_interfaces__srv__SetJointAngles_Response__get_type_description,
  &raf_interfaces__srv__SetJointAngles_Response__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace raf_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, raf_interfaces, srv, SetJointAngles_Response)() {
  return &::raf_interfaces::srv::rosidl_typesupport_c::SetJointAngles_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__struct.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__type_support.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace raf_interfaces
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _SetJointAngles_Event_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SetJointAngles_Event_type_support_ids_t;

static const _SetJointAngles_Event_type_support_ids_t _SetJointAngles_Event_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _SetJointAngles_Event_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _SetJointAngles_Event_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _SetJointAngles_Event_type_support_symbol_names_t _SetJointAngles_Event_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, raf_interfaces, srv, SetJointAngles_Event)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles_Event)),
  }
};

typedef struct _SetJointAngles_Event_type_support_data_t
{
  void * data[2];
} _SetJointAngles_Event_type_support_data_t;

static _SetJointAngles_Event_type_support_data_t _SetJointAngles_Event_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _SetJointAngles_Event_message_typesupport_map = {
  2,
  "raf_interfaces",
  &_SetJointAngles_Event_message_typesupport_ids.typesupport_identifier[0],
  &_SetJointAngles_Event_message_typesupport_symbol_names.symbol_name[0],
  &_SetJointAngles_Event_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t SetJointAngles_Event_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SetJointAngles_Event_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &raf_interfaces__srv__SetJointAngles_Event__get_type_hash,
  &raf_interfaces__srv__SetJointAngles_Event__get_type_description,
  &raf_interfaces__srv__SetJointAngles_Event__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace raf_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, raf_interfaces, srv, SetJointAngles_Event)() {
  return &::raf_interfaces::srv::rosidl_typesupport_c::SetJointAngles_Event_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__type_support.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/service_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
#include "service_msgs/msg/service_event_info.h"
#include "builtin_interfaces/msg/time.h"

namespace raf_interfaces
{

namespace srv
{

namespace rosidl_typesupport_c
{
typedef struct _SetJointAngles_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SetJointAngles_type_support_ids_t;

static const _SetJointAngles_type_support_ids_t _SetJointAngles_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _SetJointAngles_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _SetJointAngles_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _SetJointAngles_type_support_symbol_names_t _SetJointAngles_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, raf_interfaces, srv, SetJointAngles)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, raf_interfaces, srv, SetJointAngles)),
  }
};

typedef struct _SetJointAngles_type_support_data_t
{
  void * data[2];
} _SetJointAngles_type_support_data_t;

static _SetJointAngles_type_support_data_t _SetJointAngles_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _SetJointAngles_service_typesupport_map = {
  2,
  "raf_interfaces",
  &_SetJointAngles_service_typesupport_ids.typesupport_identifier[0],
  &_SetJointAngles_service_typesupport_symbol_names.symbol_name[0],
  &_SetJointAngles_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t SetJointAngles_service_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SetJointAngles_service_typesupport_map),
  rosidl_typesupport_c__get_service_typesupport_handle_function,
  &SetJointAngles_Request_message_type_support_handle,
  &SetJointAngles_Response_message_type_support_handle,
  &SetJointAngles_Event_message_type_support_handle,
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

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace raf_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_c, raf_interfaces, srv, SetJointAngles)() {
  return &::raf_interfaces::srv::rosidl_typesupport_c::SetJointAngles_service_type_support_handle;
}

#ifdef __cplusplus
}
#endif
