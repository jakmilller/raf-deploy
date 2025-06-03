// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from raf_interfaces:srv/IdentifyFood.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "raf_interfaces/srv/identify_food.h"


#ifndef RAF_INTERFACES__SRV__DETAIL__IDENTIFY_FOOD__STRUCT_H_
#define RAF_INTERFACES__SRV__DETAIL__IDENTIFY_FOOD__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'frame'
#include "sensor_msgs/msg/detail/image__struct.h"

/// Struct defined in srv/IdentifyFood in the package raf_interfaces.
typedef struct raf_interfaces__srv__IdentifyFood_Request
{
  sensor_msgs__msg__Image frame;
} raf_interfaces__srv__IdentifyFood_Request;

// Struct for a sequence of raf_interfaces__srv__IdentifyFood_Request.
typedef struct raf_interfaces__srv__IdentifyFood_Request__Sequence
{
  raf_interfaces__srv__IdentifyFood_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} raf_interfaces__srv__IdentifyFood_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'food_items'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/IdentifyFood in the package raf_interfaces.
typedef struct raf_interfaces__srv__IdentifyFood_Response
{
  rosidl_runtime_c__String food_items;
  bool success;
} raf_interfaces__srv__IdentifyFood_Response;

// Struct for a sequence of raf_interfaces__srv__IdentifyFood_Response.
typedef struct raf_interfaces__srv__IdentifyFood_Response__Sequence
{
  raf_interfaces__srv__IdentifyFood_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} raf_interfaces__srv__IdentifyFood_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  raf_interfaces__srv__IdentifyFood_Event__request__MAX_SIZE = 1
};
// response
enum
{
  raf_interfaces__srv__IdentifyFood_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/IdentifyFood in the package raf_interfaces.
typedef struct raf_interfaces__srv__IdentifyFood_Event
{
  service_msgs__msg__ServiceEventInfo info;
  raf_interfaces__srv__IdentifyFood_Request__Sequence request;
  raf_interfaces__srv__IdentifyFood_Response__Sequence response;
} raf_interfaces__srv__IdentifyFood_Event;

// Struct for a sequence of raf_interfaces__srv__IdentifyFood_Event.
typedef struct raf_interfaces__srv__IdentifyFood_Event__Sequence
{
  raf_interfaces__srv__IdentifyFood_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} raf_interfaces__srv__IdentifyFood_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RAF_INTERFACES__SRV__DETAIL__IDENTIFY_FOOD__STRUCT_H_
