// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice
#include "raf_interfaces/msg/detail/detected_food__rosidl_typesupport_fastrtps_cpp.hpp"
#include "raf_interfaces/msg/detail/detected_food__functions.h"
#include "raf_interfaces/msg/detail/detected_food__struct.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/serialization_helpers.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace raf_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{


bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_raf_interfaces
cdr_serialize(
  const raf_interfaces::msg::DetectedFood & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: food_items
  cdr << ros_message.food_items;

  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_raf_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  raf_interfaces::msg::DetectedFood & ros_message)
{
  // Member: food_items
  cdr >> ros_message.food_items;

  return true;
}


size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_raf_interfaces
get_serialized_size(
  const raf_interfaces::msg::DetectedFood & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: food_items
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.food_items.size() + 1);

  return current_alignment - initial_alignment;
}


size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_raf_interfaces
max_serialized_size_DetectedFood(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // Member: food_items
  {
    size_t array_size = 1;
    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = raf_interfaces::msg::DetectedFood;
    is_plain =
      (
      offsetof(DataType, food_items) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_raf_interfaces
cdr_serialize_key(
  const raf_interfaces::msg::DetectedFood & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: food_items
  cdr << ros_message.food_items;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_raf_interfaces
get_serialized_size_key(
  const raf_interfaces::msg::DetectedFood & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: food_items
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.food_items.size() + 1);

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_raf_interfaces
max_serialized_size_key_DetectedFood(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // Member: food_items
  {
    size_t array_size = 1;
    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = raf_interfaces::msg::DetectedFood;
    is_plain =
      (
      offsetof(DataType, food_items) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}


static bool _DetectedFood__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const raf_interfaces::msg::DetectedFood *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _DetectedFood__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<raf_interfaces::msg::DetectedFood *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _DetectedFood__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const raf_interfaces::msg::DetectedFood *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _DetectedFood__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_DetectedFood(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _DetectedFood__callbacks = {
  "raf_interfaces::msg",
  "DetectedFood",
  _DetectedFood__cdr_serialize,
  _DetectedFood__cdr_deserialize,
  _DetectedFood__get_serialized_size,
  _DetectedFood__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _DetectedFood__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_DetectedFood__callbacks,
  get_message_typesupport_handle_function,
  &raf_interfaces__msg__DetectedFood__get_type_hash,
  &raf_interfaces__msg__DetectedFood__get_type_description,
  &raf_interfaces__msg__DetectedFood__get_type_description_sources,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace raf_interfaces

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_raf_interfaces
const rosidl_message_type_support_t *
get_message_type_support_handle<raf_interfaces::msg::DetectedFood>()
{
  return &raf_interfaces::msg::typesupport_fastrtps_cpp::_DetectedFood__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, raf_interfaces, msg, DetectedFood)() {
  return &raf_interfaces::msg::typesupport_fastrtps_cpp::_DetectedFood__handle;
}

#ifdef __cplusplus
}
#endif
