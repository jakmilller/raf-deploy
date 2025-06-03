// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "raf_interfaces/msg/detected_food.hpp"


#ifndef RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__BUILDER_HPP_
#define RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "raf_interfaces/msg/detail/detected_food__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace raf_interfaces
{

namespace msg
{

namespace builder
{

class Init_DetectedFood_food_items
{
public:
  Init_DetectedFood_food_items()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::raf_interfaces::msg::DetectedFood food_items(::raf_interfaces::msg::DetectedFood::_food_items_type arg)
  {
    msg_.food_items = std::move(arg);
    return std::move(msg_);
  }

private:
  ::raf_interfaces::msg::DetectedFood msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::raf_interfaces::msg::DetectedFood>()
{
  return raf_interfaces::msg::builder::Init_DetectedFood_food_items();
}

}  // namespace raf_interfaces

#endif  // RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__BUILDER_HPP_
