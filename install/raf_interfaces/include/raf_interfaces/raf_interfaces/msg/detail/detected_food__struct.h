// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "raf_interfaces/msg/detected_food.h"


#ifndef RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__STRUCT_H_
#define RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'food_items'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/DetectedFood in the package raf_interfaces.
typedef struct raf_interfaces__msg__DetectedFood
{
  rosidl_runtime_c__String food_items;
} raf_interfaces__msg__DetectedFood;

// Struct for a sequence of raf_interfaces__msg__DetectedFood.
typedef struct raf_interfaces__msg__DetectedFood__Sequence
{
  raf_interfaces__msg__DetectedFood * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} raf_interfaces__msg__DetectedFood__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__STRUCT_H_
