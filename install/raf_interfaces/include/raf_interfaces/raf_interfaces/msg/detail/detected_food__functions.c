// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice
#include "raf_interfaces/msg/detail/detected_food__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `food_items`
#include "rosidl_runtime_c/string_functions.h"

bool
raf_interfaces__msg__DetectedFood__init(raf_interfaces__msg__DetectedFood * msg)
{
  if (!msg) {
    return false;
  }
  // food_items
  if (!rosidl_runtime_c__String__init(&msg->food_items)) {
    raf_interfaces__msg__DetectedFood__fini(msg);
    return false;
  }
  return true;
}

void
raf_interfaces__msg__DetectedFood__fini(raf_interfaces__msg__DetectedFood * msg)
{
  if (!msg) {
    return;
  }
  // food_items
  rosidl_runtime_c__String__fini(&msg->food_items);
}

bool
raf_interfaces__msg__DetectedFood__are_equal(const raf_interfaces__msg__DetectedFood * lhs, const raf_interfaces__msg__DetectedFood * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // food_items
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->food_items), &(rhs->food_items)))
  {
    return false;
  }
  return true;
}

bool
raf_interfaces__msg__DetectedFood__copy(
  const raf_interfaces__msg__DetectedFood * input,
  raf_interfaces__msg__DetectedFood * output)
{
  if (!input || !output) {
    return false;
  }
  // food_items
  if (!rosidl_runtime_c__String__copy(
      &(input->food_items), &(output->food_items)))
  {
    return false;
  }
  return true;
}

raf_interfaces__msg__DetectedFood *
raf_interfaces__msg__DetectedFood__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__msg__DetectedFood * msg = (raf_interfaces__msg__DetectedFood *)allocator.allocate(sizeof(raf_interfaces__msg__DetectedFood), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(raf_interfaces__msg__DetectedFood));
  bool success = raf_interfaces__msg__DetectedFood__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
raf_interfaces__msg__DetectedFood__destroy(raf_interfaces__msg__DetectedFood * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    raf_interfaces__msg__DetectedFood__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
raf_interfaces__msg__DetectedFood__Sequence__init(raf_interfaces__msg__DetectedFood__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__msg__DetectedFood * data = NULL;

  if (size) {
    data = (raf_interfaces__msg__DetectedFood *)allocator.zero_allocate(size, sizeof(raf_interfaces__msg__DetectedFood), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = raf_interfaces__msg__DetectedFood__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        raf_interfaces__msg__DetectedFood__fini(&data[i - 1]);
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
raf_interfaces__msg__DetectedFood__Sequence__fini(raf_interfaces__msg__DetectedFood__Sequence * array)
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
      raf_interfaces__msg__DetectedFood__fini(&array->data[i]);
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

raf_interfaces__msg__DetectedFood__Sequence *
raf_interfaces__msg__DetectedFood__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  raf_interfaces__msg__DetectedFood__Sequence * array = (raf_interfaces__msg__DetectedFood__Sequence *)allocator.allocate(sizeof(raf_interfaces__msg__DetectedFood__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = raf_interfaces__msg__DetectedFood__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
raf_interfaces__msg__DetectedFood__Sequence__destroy(raf_interfaces__msg__DetectedFood__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    raf_interfaces__msg__DetectedFood__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
raf_interfaces__msg__DetectedFood__Sequence__are_equal(const raf_interfaces__msg__DetectedFood__Sequence * lhs, const raf_interfaces__msg__DetectedFood__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!raf_interfaces__msg__DetectedFood__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
raf_interfaces__msg__DetectedFood__Sequence__copy(
  const raf_interfaces__msg__DetectedFood__Sequence * input,
  raf_interfaces__msg__DetectedFood__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(raf_interfaces__msg__DetectedFood);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    raf_interfaces__msg__DetectedFood * data =
      (raf_interfaces__msg__DetectedFood *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!raf_interfaces__msg__DetectedFood__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          raf_interfaces__msg__DetectedFood__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!raf_interfaces__msg__DetectedFood__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
