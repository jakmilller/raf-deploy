// generated from rosidl_typesupport_cpp/resource/idl__type_support.cpp.em
// with input from raf_interfaces:srv/SetJointAngles.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "raf_interfaces/srv/detail/set_joint_angles__functions.h"
#include "raf_interfaces/srv/detail/set_joint_angles__struct.hpp"
#include "rosidl_typesupport_cpp/identifier.hpp"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
#include "rosidl_typesupport_cpp/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace raf_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _SetJointAngles_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SetJointAngles_Request_type_support_ids_t;

static const _SetJointAngles_Request_type_support_ids_t _SetJointAngles_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
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
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, raf_interfaces, srv, SetJointAngles_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, raf_interfaces, srv, SetJointAngles_Request)),
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
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SetJointAngles_Request_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &raf_interfaces__srv__SetJointAngles_Request__get_type_hash,
  &raf_interfaces__srv__SetJointAngles_Request__get_type_description,
  &raf_interfaces__srv__SetJointAngles_Request__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace raf_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<raf_interfaces::srv::SetJointAngles_Request>()
{
  return &::raf_interfaces::srv::rosidl_typesupport_cpp::SetJointAngles_Request_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, raf_interfaces, srv, SetJointAngles_Request)() {
  return get_message_type_support_handle<raf_interfaces::srv::SetJointAngles_Request>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__functions.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace raf_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _SetJointAngles_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SetJointAngles_Response_type_support_ids_t;

static const _SetJointAngles_Response_type_support_ids_t _SetJointAngles_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
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
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, raf_interfaces, srv, SetJointAngles_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, raf_interfaces, srv, SetJointAngles_Response)),
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
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SetJointAngles_Response_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &raf_interfaces__srv__SetJointAngles_Response__get_type_hash,
  &raf_interfaces__srv__SetJointAngles_Response__get_type_description,
  &raf_interfaces__srv__SetJointAngles_Response__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace raf_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<raf_interfaces::srv::SetJointAngles_Response>()
{
  return &::raf_interfaces::srv::rosidl_typesupport_cpp::SetJointAngles_Response_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, raf_interfaces, srv, SetJointAngles_Response)() {
  return get_message_type_support_handle<raf_interfaces::srv::SetJointAngles_Response>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__functions.h"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace raf_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _SetJointAngles_Event_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SetJointAngles_Event_type_support_ids_t;

static const _SetJointAngles_Event_type_support_ids_t _SetJointAngles_Event_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
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
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, raf_interfaces, srv, SetJointAngles_Event)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, raf_interfaces, srv, SetJointAngles_Event)),
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
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SetJointAngles_Event_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &raf_interfaces__srv__SetJointAngles_Event__get_type_hash,
  &raf_interfaces__srv__SetJointAngles_Event__get_type_description,
  &raf_interfaces__srv__SetJointAngles_Event__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace raf_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<raf_interfaces::srv::SetJointAngles_Event>()
{
  return &::raf_interfaces::srv::rosidl_typesupport_cpp::SetJointAngles_Event_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, raf_interfaces, srv, SetJointAngles_Event)() {
  return get_message_type_support_handle<raf_interfaces::srv::SetJointAngles_Event>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/service_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace raf_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _SetJointAngles_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _SetJointAngles_type_support_ids_t;

static const _SetJointAngles_type_support_ids_t _SetJointAngles_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
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
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, raf_interfaces, srv, SetJointAngles)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, raf_interfaces, srv, SetJointAngles)),
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
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_SetJointAngles_service_typesupport_map),
  ::rosidl_typesupport_cpp::get_service_typesupport_handle_function,
  ::rosidl_typesupport_cpp::get_message_type_support_handle<raf_interfaces::srv::SetJointAngles_Request>(),
  ::rosidl_typesupport_cpp::get_message_type_support_handle<raf_interfaces::srv::SetJointAngles_Response>(),
  ::rosidl_typesupport_cpp::get_message_type_support_handle<raf_interfaces::srv::SetJointAngles_Event>(),
  &::rosidl_typesupport_cpp::service_create_event_message<raf_interfaces::srv::SetJointAngles>,
  &::rosidl_typesupport_cpp::service_destroy_event_message<raf_interfaces::srv::SetJointAngles>,
  &raf_interfaces__srv__SetJointAngles__get_type_hash,
  &raf_interfaces__srv__SetJointAngles__get_type_description,
  &raf_interfaces__srv__SetJointAngles__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace raf_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<raf_interfaces::srv::SetJointAngles>()
{
  return &::raf_interfaces::srv::rosidl_typesupport_cpp::SetJointAngles_service_type_support_handle;
}

}  // namespace rosidl_typesupport_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_cpp, raf_interfaces, srv, SetJointAngles)() {
  return ::rosidl_typesupport_cpp::get_service_type_support_handle<raf_interfaces::srv::SetJointAngles>();
}

#ifdef __cplusplus
}
#endif
