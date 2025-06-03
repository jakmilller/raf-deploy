// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "raf_interfaces/msg/detected_food.h"


#ifndef RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__FUNCTIONS_H_
#define RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/action_type_support_struct.h"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_runtime_c/service_type_support_struct.h"
#include "rosidl_runtime_c/type_description/type_description__struct.h"
#include "rosidl_runtime_c/type_description/type_source__struct.h"
#include "rosidl_runtime_c/type_hash.h"
#include "rosidl_runtime_c/visibility_control.h"
#include "raf_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "raf_interfaces/msg/detail/detected_food__struct.h"

/// Initialize msg/DetectedFood message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * raf_interfaces__msg__DetectedFood
 * )) before or use
 * raf_interfaces__msg__DetectedFood__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__msg__DetectedFood__init(raf_interfaces__msg__DetectedFood * msg);

/// Finalize msg/DetectedFood message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__msg__DetectedFood__fini(raf_interfaces__msg__DetectedFood * msg);

/// Create msg/DetectedFood message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * raf_interfaces__msg__DetectedFood__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
raf_interfaces__msg__DetectedFood *
raf_interfaces__msg__DetectedFood__create(void);

/// Destroy msg/DetectedFood message.
/**
 * It calls
 * raf_interfaces__msg__DetectedFood__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__msg__DetectedFood__destroy(raf_interfaces__msg__DetectedFood * msg);

/// Check for msg/DetectedFood message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__msg__DetectedFood__are_equal(const raf_interfaces__msg__DetectedFood * lhs, const raf_interfaces__msg__DetectedFood * rhs);

/// Copy a msg/DetectedFood message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__msg__DetectedFood__copy(
  const raf_interfaces__msg__DetectedFood * input,
  raf_interfaces__msg__DetectedFood * output);

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_type_hash_t *
raf_interfaces__msg__DetectedFood__get_type_hash(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
raf_interfaces__msg__DetectedFood__get_type_description(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeSource *
raf_interfaces__msg__DetectedFood__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
raf_interfaces__msg__DetectedFood__get_type_description_sources(
  const rosidl_message_type_support_t * type_support);

/// Initialize array of msg/DetectedFood messages.
/**
 * It allocates the memory for the number of elements and calls
 * raf_interfaces__msg__DetectedFood__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__msg__DetectedFood__Sequence__init(raf_interfaces__msg__DetectedFood__Sequence * array, size_t size);

/// Finalize array of msg/DetectedFood messages.
/**
 * It calls
 * raf_interfaces__msg__DetectedFood__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__msg__DetectedFood__Sequence__fini(raf_interfaces__msg__DetectedFood__Sequence * array);

/// Create array of msg/DetectedFood messages.
/**
 * It allocates the memory for the array and calls
 * raf_interfaces__msg__DetectedFood__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
raf_interfaces__msg__DetectedFood__Sequence *
raf_interfaces__msg__DetectedFood__Sequence__create(size_t size);

/// Destroy array of msg/DetectedFood messages.
/**
 * It calls
 * raf_interfaces__msg__DetectedFood__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__msg__DetectedFood__Sequence__destroy(raf_interfaces__msg__DetectedFood__Sequence * array);

/// Check for msg/DetectedFood message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__msg__DetectedFood__Sequence__are_equal(const raf_interfaces__msg__DetectedFood__Sequence * lhs, const raf_interfaces__msg__DetectedFood__Sequence * rhs);

/// Copy an array of msg/DetectedFood messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__msg__DetectedFood__Sequence__copy(
  const raf_interfaces__msg__DetectedFood__Sequence * input,
  raf_interfaces__msg__DetectedFood__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // RAF_INTERFACES__MSG__DETAIL__DETECTED_FOOD__FUNCTIONS_H_
