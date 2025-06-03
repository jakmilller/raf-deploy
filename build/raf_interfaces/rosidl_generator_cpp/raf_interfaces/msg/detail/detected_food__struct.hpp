// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "raf_interfaces/msg/detected_food.hpp"


#ifndef RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__STRUCT_HPP_
#define RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__raf_interfaces__msg__DetectedFood __attribute__((deprecated))
#else
# define DEPRECATED__raf_interfaces__msg__DetectedFood __declspec(deprecated)
#endif

namespace raf_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DetectedFood_
{
  using Type = DetectedFood_<ContainerAllocator>;

  explicit DetectedFood_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->food_items = "";
    }
  }

  explicit DetectedFood_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : food_items(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->food_items = "";
    }
  }

  // field types and members
  using _food_items_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _food_items_type food_items;

  // setters for named parameter idiom
  Type & set__food_items(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->food_items = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    raf_interfaces::msg::DetectedFood_<ContainerAllocator> *;
  using ConstRawPtr =
    const raf_interfaces::msg::DetectedFood_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<raf_interfaces::msg::DetectedFood_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<raf_interfaces::msg::DetectedFood_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      raf_interfaces::msg::DetectedFood_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<raf_interfaces::msg::DetectedFood_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      raf_interfaces::msg::DetectedFood_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<raf_interfaces::msg::DetectedFood_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<raf_interfaces::msg::DetectedFood_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<raf_interfaces::msg::DetectedFood_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__raf_interfaces__msg__DetectedFood
    std::shared_ptr<raf_interfaces::msg::DetectedFood_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__raf_interfaces__msg__DetectedFood
    std::shared_ptr<raf_interfaces::msg::DetectedFood_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DetectedFood_ & other) const
  {
    if (this->food_items != other.food_items) {
      return false;
    }
    return true;
  }
  bool operator!=(const DetectedFood_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DetectedFood_

// alias to use template instance with default allocator
using DetectedFood =
  raf_interfaces::msg::DetectedFood_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace raf_interfaces

#endif  // RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__STRUCT_HPP_
