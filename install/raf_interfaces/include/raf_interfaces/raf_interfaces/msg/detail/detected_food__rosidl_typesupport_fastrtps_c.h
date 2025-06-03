// generated from rosidl_typesupport_fastrtps_c/resource/idl__rosidl_typesupport_fastrtps_c.h.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice
#ifndef RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
#define RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_


#include <stddef.h>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "raf_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "raf_interfaces/msg/detail/detected_food__struct.h"
#include "fastcdr/Cdr.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_raf_interfaces
bool cdr_serialize_raf_interfaces__msg__DetectedFood(
  const raf_interfaces__msg__DetectedFood * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_raf_interfaces
bool cdr_deserialize_raf_interfaces__msg__DetectedFood(
  eprosima::fastcdr::Cdr &,
  raf_interfaces__msg__DetectedFood * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_raf_interfaces
size_t get_serialized_size_raf_interfaces__msg__DetectedFood(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_raf_interfaces
size_t max_serialized_size_raf_interfaces__msg__DetectedFood(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_raf_interfaces
bool cdr_serialize_key_raf_interfaces__msg__DetectedFood(
  const raf_interfaces__msg__DetectedFood * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_raf_interfaces
size_t get_serialized_size_key_raf_interfaces__msg__DetectedFood(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_raf_interfaces
size_t max_serialized_size_key_raf_interfaces__msg__DetectedFood(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_raf_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, raf_interfaces, msg, DetectedFood)();

#ifdef __cplusplus
}
#endif

#endif  // RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__ROSIDL_TYPESUPPORT_FASTRTPS_C_H_
