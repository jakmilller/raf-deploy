// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from raf_interfaces:srv/SetJointAngles.idl
// generated code does not contain a copyright notice
#include "raf_interfaces/srv/detail/set_joint_angles__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `joint_angles`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
raf_interfaces__srv__SetJointAngles_Request__init(raf_interfaces__srv__SetJointAngles_Request * msg)
{
  if (!msg) {
    return false;
  }
  // joint_angles
  if (!rosidl_runtime_c__double__Sequence__init(&msg->joint_angles, 0)) {
    raf_interfaces__srv__SetJointAngles_Request__fini(msg);
    return false;
  }
  return true;
}

void
raf_interfaces__srv__SetJointAngles_Request__fini(raf_interfaces__srv__SetJointAngles_Request * msg)
{
  if (!msg) {
    return;
  }
  // joint_angles
  rosidl_runtime_c__double__Sequence__fini(&msg->joint_angles);
}

bool
raf_interfaces__srv__SetJointAngles_Request__are_equal(const raf_interfaces__srv__SetJointAngles_Request * lhs, const raf_interfaces__srv__SetJointAngles_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // joint_angles
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->joint_angles), &(rhs->joint_angles)))
  {
    return false;
  }
  return true;
}

bool
raf_interfaces__srv__SetJointAngles_Request__copy(
  const raf_interfaces__srv__SetJointAngles_Request * input,
  raf_interfaces__srv__SetJointAngles_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // joint_angles
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->joint_angles), &(output->joint_angles)))
  {
    return false;
  }
  return true;
}

raf_interfaces__srv__SetJointAngles_Request *
raf_interfaces__srv__SetJointAngles_Request__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__srv__SetJointAngles_Request * msg = (raf_interfaces__srv__SetJointAngles_Request *)allocator.allocate(sizeof(raf_interfaces__srv__SetJointAngles_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(raf_interfaces__srv__SetJointAngles_Request));
  bool success = raf_interfaces__srv__SetJointAngles_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
raf_interfaces__srv__SetJointAngles_Request__destroy(raf_interfaces__srv__SetJointAngles_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    raf_interfaces__srv__SetJointAngles_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
raf_interfaces__srv__SetJointAngles_Request__Sequence__init(raf_interfaces__srv__SetJointAngles_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__srv__SetJointAngles_Request * data = NULL;

  if (size) {
    data = (raf_interfaces__srv__SetJointAngles_Request *)allocator.zero_allocate(size, sizeof(raf_interfaces__srv__SetJointAngles_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = raf_interfaces__srv__SetJointAngles_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        raf_interfaces__srv__SetJointAngles_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
raf_interfaces__srv__SetJointAngles_Request__Sequence__fini(raf_interfaces__srv__SetJointAngles_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      raf_interfaces__srv__SetJointAngles_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

raf_interfaces__srv__SetJointAngles_Request__Sequence *
raf_interfaces__srv__SetJointAngles_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__srv__SetJointAngles_Request__Sequence * array = (raf_interfaces__srv__SetJointAngles_Request__Sequence *)allocator.allocate(sizeof(raf_interfaces__srv__SetJointAngles_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = raf_interfaces__srv__SetJointAngles_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
raf_interfaces__srv__SetJointAngles_Request__Sequence__destroy(raf_interfaces__srv__SetJointAngles_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    raf_interfaces__srv__SetJointAngles_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
raf_interfaces__srv__SetJointAngles_Request__Sequence__are_equal(const raf_interfaces__srv__SetJointAngles_Request__Sequence * lhs, const raf_interfaces__srv__SetJointAngles_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!raf_interfaces__srv__SetJointAngles_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
raf_interfaces__srv__SetJointAngles_Request__Sequence__copy(
  const raf_interfaces__srv__SetJointAngles_Request__Sequence * input,
  raf_interfaces__srv__SetJointAngles_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(raf_interfaces__srv__SetJointAngles_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    raf_interfaces__srv__SetJointAngles_Request * data =
      (raf_interfaces__srv__SetJointAngles_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!raf_interfaces__srv__SetJointAngles_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          raf_interfaces__srv__SetJointAngles_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!raf_interfaces__srv__SetJointAngles_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

bool
raf_interfaces__srv__SetJointAngles_Response__init(raf_interfaces__srv__SetJointAngles_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    raf_interfaces__srv__SetJointAngles_Response__fini(msg);
    return false;
  }
  return true;
}

void
raf_interfaces__srv__SetJointAngles_Response__fini(raf_interfaces__srv__SetJointAngles_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // message
  rosidl_runtime_c__String__fini(&msg->message);
}

bool
raf_interfaces__srv__SetJointAngles_Response__are_equal(const raf_interfaces__srv__SetJointAngles_Response * lhs, const raf_interfaces__srv__SetJointAngles_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->message), &(rhs->message)))
  {
    return false;
  }
  return true;
}

bool
raf_interfaces__srv__SetJointAngles_Response__copy(
  const raf_interfaces__srv__SetJointAngles_Response * input,
  raf_interfaces__srv__SetJointAngles_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // message
  if (!rosidl_runtime_c__String__copy(
      &(input->message), &(output->message)))
  {
    return false;
  }
  return true;
}

raf_interfaces__srv__SetJointAngles_Response *
raf_interfaces__srv__SetJointAngles_Response__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__srv__SetJointAngles_Response * msg = (raf_interfaces__srv__SetJointAngles_Response *)allocator.allocate(sizeof(raf_interfaces__srv__SetJointAngles_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(raf_interfaces__srv__SetJointAngles_Response));
  bool success = raf_interfaces__srv__SetJointAngles_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
raf_interfaces__srv__SetJointAngles_Response__destroy(raf_interfaces__srv__SetJointAngles_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    raf_interfaces__srv__SetJointAngles_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
raf_interfaces__srv__SetJointAngles_Response__Sequence__init(raf_interfaces__srv__SetJointAngles_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__srv__SetJointAngles_Response * data = NULL;

  if (size) {
    data = (raf_interfaces__srv__SetJointAngles_Response *)allocator.zero_allocate(size, sizeof(raf_interfaces__srv__SetJointAngles_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = raf_interfaces__srv__SetJointAngles_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        raf_interfaces__srv__SetJointAngles_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
raf_interfaces__srv__SetJointAngles_Response__Sequence__fini(raf_interfaces__srv__SetJointAngles_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      raf_interfaces__srv__SetJointAngles_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

raf_interfaces__srv__SetJointAngles_Response__Sequence *
raf_interfaces__srv__SetJointAngles_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__srv__SetJointAngles_Response__Sequence * array = (raf_interfaces__srv__SetJointAngles_Response__Sequence *)allocator.allocate(sizeof(raf_interfaces__srv__SetJointAngles_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = raf_interfaces__srv__SetJointAngles_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
raf_interfaces__srv__SetJointAngles_Response__Sequence__destroy(raf_interfaces__srv__SetJointAngles_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    raf_interfaces__srv__SetJointAngles_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
raf_interfaces__srv__SetJointAngles_Response__Sequence__are_equal(const raf_interfaces__srv__SetJointAngles_Response__Sequence * lhs, const raf_interfaces__srv__SetJointAngles_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!raf_interfaces__srv__SetJointAngles_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
raf_interfaces__srv__SetJointAngles_Response__Sequence__copy(
  const raf_interfaces__srv__SetJointAngles_Response__Sequence * input,
  raf_interfaces__srv__SetJointAngles_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(raf_interfaces__srv__SetJointAngles_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    raf_interfaces__srv__SetJointAngles_Response * data =
      (raf_interfaces__srv__SetJointAngles_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!raf_interfaces__srv__SetJointAngles_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          raf_interfaces__srv__SetJointAngles_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!raf_interfaces__srv__SetJointAngles_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `info`
#include "service_msgs/msg/detail/service_event_info__functions.h"
// Member `request`
// Member `response`
// already included above
// #include "raf_interfaces/srv/detail/set_joint_angles__functions.h"

bool
raf_interfaces__srv__SetJointAngles_Event__init(raf_interfaces__srv__SetJointAngles_Event * msg)
{
  if (!msg) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__init(&msg->info)) {
    raf_interfaces__srv__SetJointAngles_Event__fini(msg);
    return false;
  }
  // request
  if (!raf_interfaces__srv__SetJointAngles_Request__Sequence__init(&msg->request, 0)) {
    raf_interfaces__srv__SetJointAngles_Event__fini(msg);
    return false;
  }
  // response
  if (!raf_interfaces__srv__SetJointAngles_Response__Sequence__init(&msg->response, 0)) {
    raf_interfaces__srv__SetJointAngles_Event__fini(msg);
    return false;
  }
  return true;
}

void
raf_interfaces__srv__SetJointAngles_Event__fini(raf_interfaces__srv__SetJointAngles_Event * msg)
{
  if (!msg) {
    return;
  }
  // info
  service_msgs__msg__ServiceEventInfo__fini(&msg->info);
  // request
  raf_interfaces__srv__SetJointAngles_Request__Sequence__fini(&msg->request);
  // response
  raf_interfaces__srv__SetJointAngles_Response__Sequence__fini(&msg->response);
}

bool
raf_interfaces__srv__SetJointAngles_Event__are_equal(const raf_interfaces__srv__SetJointAngles_Event * lhs, const raf_interfaces__srv__SetJointAngles_Event * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__are_equal(
      &(lhs->info), &(rhs->info)))
  {
    return false;
  }
  // request
  if (!raf_interfaces__srv__SetJointAngles_Request__Sequence__are_equal(
      &(lhs->request), &(rhs->request)))
  {
    return false;
  }
  // response
  if (!raf_interfaces__srv__SetJointAngles_Response__Sequence__are_equal(
      &(lhs->response), &(rhs->response)))
  {
    return false;
  }
  return true;
}

bool
raf_interfaces__srv__SetJointAngles_Event__copy(
  const raf_interfaces__srv__SetJointAngles_Event * input,
  raf_interfaces__srv__SetJointAngles_Event * output)
{
  if (!input || !output) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__copy(
      &(input->info), &(output->info)))
  {
    return false;
  }
  // request
  if (!raf_interfaces__srv__SetJointAngles_Request__Sequence__copy(
      &(input->request), &(output->request)))
  {
    return false;
  }
  // response
  if (!raf_interfaces__srv__SetJointAngles_Response__Sequence__copy(
      &(input->response), &(output->response)))
  {
    return false;
  }
  return true;
}

raf_interfaces__srv__SetJointAngles_Event *
raf_interfaces__srv__SetJointAngles_Event__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__srv__SetJointAngles_Event * msg = (raf_interfaces__srv__SetJointAngles_Event *)allocator.allocate(sizeof(raf_interfaces__srv__SetJointAngles_Event), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(raf_interfaces__srv__SetJointAngles_Event));
  bool success = raf_interfaces__srv__SetJointAngles_Event__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
raf_interfaces__srv__SetJointAngles_Event__destroy(raf_interfaces__srv__SetJointAngles_Event * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    raf_interfaces__srv__SetJointAngles_Event__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
raf_interfaces__srv__SetJointAngles_Event__Sequence__init(raf_interfaces__srv__SetJointAngles_Event__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__srv__SetJointAngles_Event * data = NULL;

  if (size) {
    data = (raf_interfaces__srv__SetJointAngles_Event *)allocator.zero_allocate(size, sizeof(raf_interfaces__srv__SetJointAngles_Event), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = raf_interfaces__srv__SetJointAngles_Event__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        raf_interfaces__srv__SetJointAngles_Event__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
raf_interfaces__srv__SetJointAngles_Event__Sequence__fini(raf_interfaces__srv__SetJointAngles_Event__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      raf_interfaces__srv__SetJointAngles_Event__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

raf_interfaces__srv__SetJointAngles_Event__Sequence *
raf_interfaces__srv__SetJointAngles_Event__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__srv__SetJointAngles_Event__Sequence * array = (raf_interfaces__srv__SetJointAngles_Event__Sequence *)allocator.allocate(sizeof(raf_interfaces__srv__SetJointAngles_Event__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = raf_interfaces__srv__SetJointAngles_Event__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
raf_interfaces__srv__SetJointAngles_Event__Sequence__destroy(raf_interfaces__srv__SetJointAngles_Event__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    raf_interfaces__srv__SetJointAngles_Event__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
raf_interfaces__srv__SetJointAngles_Event__Sequence__are_equal(const raf_interfaces__srv__SetJointAngles_Event__Sequence * lhs, const raf_interfaces__srv__SetJointAngles_Event__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!raf_interfaces__srv__SetJointAngles_Event__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
raf_interfaces__srv__SetJointAngles_Event__Sequence__copy(
  const raf_interfaces__srv__SetJointAngles_Event__Sequence * input,
  raf_interfaces__srv__SetJointAngles_Event__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(raf_interfaces__srv__SetJointAngles_Event);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    raf_interfaces__srv__SetJointAngles_Event * data =
      (raf_interfaces__srv__SetJointAngles_Event *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!raf_interfaces__srv__SetJointAngles_Event__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          raf_interfaces__srv__SetJointAngles_Event__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!raf_interfaces__srv__SetJointAngles_Event__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
