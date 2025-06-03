// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "raf_interfaces/msg/detected_food.hpp"


#ifndef RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__TRAITS_HPP_
#define RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "raf_interfaces/msg/detail/detected_food__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace raf_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const DetectedFood & msg,
  std::ostream & out)
{
  out << "{";
  // member: food_items
  {
    out << "food_items: ";
    rosidl_generator_traits::value_to_yaml(msg.food_items, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DetectedFood & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: food_items
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "food_items: ";
    rosidl_generator_traits::value_to_yaml(msg.food_items, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DetectedFood & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace raf_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use raf_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const raf_interfaces::msg::DetectedFood & msg,
  std::ostream & out, size_t indentation = 0)
{
  raf_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use raf_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const raf_interfaces::msg::DetectedFood & msg)
{
  return raf_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<raf_interfaces::msg::DetectedFood>()
{
  return "raf_interfaces::msg::DetectedFood";
}

template<>
inline const char * name<raf_interfaces::msg::DetectedFood>()
{
  return "raf_interfaces/msg/DetectedFood";
}

template<>
struct has_fixed_size<raf_interfaces::msg::DetectedFood>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<raf_interfaces::msg::DetectedFood>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<raf_interfaces::msg::DetectedFood>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__TRAITS_HPP_
