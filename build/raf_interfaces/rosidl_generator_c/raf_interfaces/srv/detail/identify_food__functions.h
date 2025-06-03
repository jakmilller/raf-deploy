// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from raf_interfaces:srv/IdentifyFood.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "raf_interfaces/srv/identify_food.h"


#ifndef RAF_INTERFACES__SRV__DETAIL__IDENTIFY_FOOD__FUNCTIONS_H_
#define RAF_INTERFACES__SRV__DETAIL__IDENTIFY_FOOD__FUNCTIONS_H_

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

#include "raf_interfaces/srv/detail/identify_food__struct.h"

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_type_hash_t *
raf_interfaces__srv__IdentifyFood__get_type_hash(
  const rosidl_service_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
raf_interfaces__srv__IdentifyFood__get_type_description(
  const rosidl_service_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeSource *
raf_interfaces__srv__IdentifyFood__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
raf_interfaces__srv__IdentifyFood__get_type_description_sources(
  const rosidl_service_type_support_t * type_support);

/// Initialize srv/IdentifyFood message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * raf_interfaces__srv__IdentifyFood_Request
 * )) before or use
 * raf_interfaces__srv__IdentifyFood_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Request__init(raf_interfaces__srv__IdentifyFood_Request * msg);

/// Finalize srv/IdentifyFood message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Request__fini(raf_interfaces__srv__IdentifyFood_Request * msg);

/// Create srv/IdentifyFood message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * raf_interfaces__srv__IdentifyFood_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
raf_interfaces__srv__IdentifyFood_Request *
raf_interfaces__srv__IdentifyFood_Request__create(void);

/// Destroy srv/IdentifyFood message.
/**
 * It calls
 * raf_interfaces__srv__IdentifyFood_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Request__destroy(raf_interfaces__srv__IdentifyFood_Request * msg);

/// Check for srv/IdentifyFood message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Request__are_equal(const raf_interfaces__srv__IdentifyFood_Request * lhs, const raf_interfaces__srv__IdentifyFood_Request * rhs);

/// Copy a srv/IdentifyFood message.
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
raf_interfaces__srv__IdentifyFood_Request__copy(
  const raf_interfaces__srv__IdentifyFood_Request * input,
  raf_interfaces__srv__IdentifyFood_Request * output);

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_type_hash_t *
raf_interfaces__srv__IdentifyFood_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
raf_interfaces__srv__IdentifyFood_Request__get_type_description(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeSource *
raf_interfaces__srv__IdentifyFood_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
raf_interfaces__srv__IdentifyFood_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support);

/// Initialize array of srv/IdentifyFood messages.
/**
 * It allocates the memory for the number of elements and calls
 * raf_interfaces__srv__IdentifyFood_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Request__Sequence__init(raf_interfaces__srv__IdentifyFood_Request__Sequence * array, size_t size);

/// Finalize array of srv/IdentifyFood messages.
/**
 * It calls
 * raf_interfaces__srv__IdentifyFood_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Request__Sequence__fini(raf_interfaces__srv__IdentifyFood_Request__Sequence * array);

/// Create array of srv/IdentifyFood messages.
/**
 * It allocates the memory for the array and calls
 * raf_interfaces__srv__IdentifyFood_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
raf_interfaces__srv__IdentifyFood_Request__Sequence *
raf_interfaces__srv__IdentifyFood_Request__Sequence__create(size_t size);

/// Destroy array of srv/IdentifyFood messages.
/**
 * It calls
 * raf_interfaces__srv__IdentifyFood_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Request__Sequence__destroy(raf_interfaces__srv__IdentifyFood_Request__Sequence * array);

/// Check for srv/IdentifyFood message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Request__Sequence__are_equal(const raf_interfaces__srv__IdentifyFood_Request__Sequence * lhs, const raf_interfaces__srv__IdentifyFood_Request__Sequence * rhs);

/// Copy an array of srv/IdentifyFood messages.
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
raf_interfaces__srv__IdentifyFood_Request__Sequence__copy(
  const raf_interfaces__srv__IdentifyFood_Request__Sequence * input,
  raf_interfaces__srv__IdentifyFood_Request__Sequence * output);

/// Initialize srv/IdentifyFood message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * raf_interfaces__srv__IdentifyFood_Response
 * )) before or use
 * raf_interfaces__srv__IdentifyFood_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Response__init(raf_interfaces__srv__IdentifyFood_Response * msg);

/// Finalize srv/IdentifyFood message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Response__fini(raf_interfaces__srv__IdentifyFood_Response * msg);

/// Create srv/IdentifyFood message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * raf_interfaces__srv__IdentifyFood_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
raf_interfaces__srv__IdentifyFood_Response *
raf_interfaces__srv__IdentifyFood_Response__create(void);

/// Destroy srv/IdentifyFood message.
/**
 * It calls
 * raf_interfaces__srv__IdentifyFood_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Response__destroy(raf_interfaces__srv__IdentifyFood_Response * msg);

/// Check for srv/IdentifyFood message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Response__are_equal(const raf_interfaces__srv__IdentifyFood_Response * lhs, const raf_interfaces__srv__IdentifyFood_Response * rhs);

/// Copy a srv/IdentifyFood message.
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
raf_interfaces__srv__IdentifyFood_Response__copy(
  const raf_interfaces__srv__IdentifyFood_Response * input,
  raf_interfaces__srv__IdentifyFood_Response * output);

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_type_hash_t *
raf_interfaces__srv__IdentifyFood_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
raf_interfaces__srv__IdentifyFood_Response__get_type_description(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeSource *
raf_interfaces__srv__IdentifyFood_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
raf_interfaces__srv__IdentifyFood_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support);

/// Initialize array of srv/IdentifyFood messages.
/**
 * It allocates the memory for the number of elements and calls
 * raf_interfaces__srv__IdentifyFood_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Response__Sequence__init(raf_interfaces__srv__IdentifyFood_Response__Sequence * array, size_t size);

/// Finalize array of srv/IdentifyFood messages.
/**
 * It calls
 * raf_interfaces__srv__IdentifyFood_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Response__Sequence__fini(raf_interfaces__srv__IdentifyFood_Response__Sequence * array);

/// Create array of srv/IdentifyFood messages.
/**
 * It allocates the memory for the array and calls
 * raf_interfaces__srv__IdentifyFood_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
raf_interfaces__srv__IdentifyFood_Response__Sequence *
raf_interfaces__srv__IdentifyFood_Response__Sequence__create(size_t size);

/// Destroy array of srv/IdentifyFood messages.
/**
 * It calls
 * raf_interfaces__srv__IdentifyFood_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Response__Sequence__destroy(raf_interfaces__srv__IdentifyFood_Response__Sequence * array);

/// Check for srv/IdentifyFood message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Response__Sequence__are_equal(const raf_interfaces__srv__IdentifyFood_Response__Sequence * lhs, const raf_interfaces__srv__IdentifyFood_Response__Sequence * rhs);

/// Copy an array of srv/IdentifyFood messages.
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
raf_interfaces__srv__IdentifyFood_Response__Sequence__copy(
  const raf_interfaces__srv__IdentifyFood_Response__Sequence * input,
  raf_interfaces__srv__IdentifyFood_Response__Sequence * output);

/// Initialize srv/IdentifyFood message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * raf_interfaces__srv__IdentifyFood_Event
 * )) before or use
 * raf_interfaces__srv__IdentifyFood_Event__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Event__init(raf_interfaces__srv__IdentifyFood_Event * msg);

/// Finalize srv/IdentifyFood message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Event__fini(raf_interfaces__srv__IdentifyFood_Event * msg);

/// Create srv/IdentifyFood message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * raf_interfaces__srv__IdentifyFood_Event__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
raf_interfaces__srv__IdentifyFood_Event *
raf_interfaces__srv__IdentifyFood_Event__create(void);

/// Destroy srv/IdentifyFood message.
/**
 * It calls
 * raf_interfaces__srv__IdentifyFood_Event__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Event__destroy(raf_interfaces__srv__IdentifyFood_Event * msg);

/// Check for srv/IdentifyFood message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Event__are_equal(const raf_interfaces__srv__IdentifyFood_Event * lhs, const raf_interfaces__srv__IdentifyFood_Event * rhs);

/// Copy a srv/IdentifyFood message.
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
raf_interfaces__srv__IdentifyFood_Event__copy(
  const raf_interfaces__srv__IdentifyFood_Event * input,
  raf_interfaces__srv__IdentifyFood_Event * output);

/// Retrieve pointer to the hash of the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_type_hash_t *
raf_interfaces__srv__IdentifyFood_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeDescription *
raf_interfaces__srv__IdentifyFood_Event__get_type_description(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the single raw source text that defined this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeSource *
raf_interfaces__srv__IdentifyFood_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support);

/// Retrieve pointer to the recursive raw sources that defined the description of this type.
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_runtime_c__type_description__TypeSource__Sequence *
raf_interfaces__srv__IdentifyFood_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support);

/// Initialize array of srv/IdentifyFood messages.
/**
 * It allocates the memory for the number of elements and calls
 * raf_interfaces__srv__IdentifyFood_Event__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Event__Sequence__init(raf_interfaces__srv__IdentifyFood_Event__Sequence * array, size_t size);

/// Finalize array of srv/IdentifyFood messages.
/**
 * It calls
 * raf_interfaces__srv__IdentifyFood_Event__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Event__Sequence__fini(raf_interfaces__srv__IdentifyFood_Event__Sequence * array);

/// Create array of srv/IdentifyFood messages.
/**
 * It allocates the memory for the array and calls
 * raf_interfaces__srv__IdentifyFood_Event__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
raf_interfaces__srv__IdentifyFood_Event__Sequence *
raf_interfaces__srv__IdentifyFood_Event__Sequence__create(size_t size);

/// Destroy array of srv/IdentifyFood messages.
/**
 * It calls
 * raf_interfaces__srv__IdentifyFood_Event__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
void
raf_interfaces__srv__IdentifyFood_Event__Sequence__destroy(raf_interfaces__srv__IdentifyFood_Event__Sequence * array);

/// Check for srv/IdentifyFood message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
bool
raf_interfaces__srv__IdentifyFood_Event__Sequence__are_equal(const raf_interfaces__srv__IdentifyFood_Event__Sequence * lhs, const raf_interfaces__srv__IdentifyFood_Event__Sequence * rhs);

/// Copy an array of srv/IdentifyFood messages.
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
raf_interfaces__srv__IdentifyFood_Event__Sequence__copy(
  const raf_interfaces__srv__IdentifyFood_Event__Sequence * input,
  raf_interfaces__srv__IdentifyFood_Event__Sequence * output);
#ifdef __cplusplus
}
#endif

#endif  // RAF_INTERFACES__SRV__DETAIL__IDENTIFY_FOOD__FUNCTIONS_H_
