// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from raf_interfaces:msg/DetectedFood.idl
// generated code does not contain a copyright notice

#include "raf_interfaces/msg/detail/detected_food__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_raf_interfaces
const rosidl_type_hash_t *
raf_interfaces__msg__DetectedFood__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xe9, 0x8a, 0x6e, 0x91, 0x65, 0x0f, 0x24, 0xd1,
      0x90, 0x07, 0xb8, 0x69, 0x56, 0x3a, 0xde, 0xb2,
      0x60, 0x29, 0x88, 0xd1, 0x94, 0xdd, 0xf9, 0x68,
      0xe9, 0x1b, 0x1d, 0x9a, 0x96, 0x87, 0xbc, 0x4d,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char raf_interfaces__msg__DetectedFood__TYPE_NAME[] = "raf_interfaces/msg/DetectedFood";

// Define type names, field names, and default values
static char raf_interfaces__msg__DetectedFood__FIELD_NAME__food_items[] = "food_items";

static rosidl_runtime_c__type_description__Field raf_interfaces__msg__DetectedFood__FIELDS[] = {
  {
    {raf_interfaces__msg__DetectedFood__FIELD_NAME__food_items, 10, 10},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
raf_interfaces__msg__DetectedFood__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {raf_interfaces__msg__DetectedFood__TYPE_NAME, 31, 31},
      {raf_interfaces__msg__DetectedFood__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "string food_items";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
raf_interfaces__msg__DetectedFood__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {raf_interfaces__msg__DetectedFood__TYPE_NAME, 31, 31},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 17, 17},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
raf_interfaces__msg__DetectedFood__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *raf_interfaces__msg__DetectedFood__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
