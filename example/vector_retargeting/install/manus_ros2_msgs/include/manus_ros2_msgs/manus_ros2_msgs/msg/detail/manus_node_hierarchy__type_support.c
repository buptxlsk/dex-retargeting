// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from manus_ros2_msgs:msg/ManusNodeHierarchy.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "manus_ros2_msgs/msg/detail/manus_node_hierarchy__rosidl_typesupport_introspection_c.h"
#include "manus_ros2_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "manus_ros2_msgs/msg/detail/manus_node_hierarchy__functions.h"
#include "manus_ros2_msgs/msg/detail/manus_node_hierarchy__struct.h"


// Include directives for member types
// Member `node_ids`
// Member `parent_node_ids`
#include "rosidl_runtime_c/primitives_sequence_functions.h"
// Member `poses`
#include "geometry_msgs/msg/pose.h"
// Member `poses`
#include "geometry_msgs/msg/detail/pose__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  manus_ros2_msgs__msg__ManusNodeHierarchy__init(message_memory);
}

void manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_fini_function(void * message_memory)
{
  manus_ros2_msgs__msg__ManusNodeHierarchy__fini(message_memory);
}

size_t manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__size_function__ManusNodeHierarchy__node_ids(
  const void * untyped_member)
{
  const rosidl_runtime_c__uint32__Sequence * member =
    (const rosidl_runtime_c__uint32__Sequence *)(untyped_member);
  return member->size;
}

const void * manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_const_function__ManusNodeHierarchy__node_ids(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__uint32__Sequence * member =
    (const rosidl_runtime_c__uint32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_function__ManusNodeHierarchy__node_ids(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__uint32__Sequence * member =
    (rosidl_runtime_c__uint32__Sequence *)(untyped_member);
  return &member->data[index];
}

void manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__fetch_function__ManusNodeHierarchy__node_ids(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const uint32_t * item =
    ((const uint32_t *)
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_const_function__ManusNodeHierarchy__node_ids(untyped_member, index));
  uint32_t * value =
    (uint32_t *)(untyped_value);
  *value = *item;
}

void manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__assign_function__ManusNodeHierarchy__node_ids(
  void * untyped_member, size_t index, const void * untyped_value)
{
  uint32_t * item =
    ((uint32_t *)
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_function__ManusNodeHierarchy__node_ids(untyped_member, index));
  const uint32_t * value =
    (const uint32_t *)(untyped_value);
  *item = *value;
}

bool manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__resize_function__ManusNodeHierarchy__node_ids(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__uint32__Sequence * member =
    (rosidl_runtime_c__uint32__Sequence *)(untyped_member);
  rosidl_runtime_c__uint32__Sequence__fini(member);
  return rosidl_runtime_c__uint32__Sequence__init(member, size);
}

size_t manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__size_function__ManusNodeHierarchy__parent_node_ids(
  const void * untyped_member)
{
  const rosidl_runtime_c__uint32__Sequence * member =
    (const rosidl_runtime_c__uint32__Sequence *)(untyped_member);
  return member->size;
}

const void * manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_const_function__ManusNodeHierarchy__parent_node_ids(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__uint32__Sequence * member =
    (const rosidl_runtime_c__uint32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_function__ManusNodeHierarchy__parent_node_ids(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__uint32__Sequence * member =
    (rosidl_runtime_c__uint32__Sequence *)(untyped_member);
  return &member->data[index];
}

void manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__fetch_function__ManusNodeHierarchy__parent_node_ids(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const uint32_t * item =
    ((const uint32_t *)
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_const_function__ManusNodeHierarchy__parent_node_ids(untyped_member, index));
  uint32_t * value =
    (uint32_t *)(untyped_value);
  *value = *item;
}

void manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__assign_function__ManusNodeHierarchy__parent_node_ids(
  void * untyped_member, size_t index, const void * untyped_value)
{
  uint32_t * item =
    ((uint32_t *)
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_function__ManusNodeHierarchy__parent_node_ids(untyped_member, index));
  const uint32_t * value =
    (const uint32_t *)(untyped_value);
  *item = *value;
}

bool manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__resize_function__ManusNodeHierarchy__parent_node_ids(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__uint32__Sequence * member =
    (rosidl_runtime_c__uint32__Sequence *)(untyped_member);
  rosidl_runtime_c__uint32__Sequence__fini(member);
  return rosidl_runtime_c__uint32__Sequence__init(member, size);
}

size_t manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__size_function__ManusNodeHierarchy__poses(
  const void * untyped_member)
{
  const geometry_msgs__msg__Pose__Sequence * member =
    (const geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return member->size;
}

const void * manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_const_function__ManusNodeHierarchy__poses(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Pose__Sequence * member =
    (const geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return &member->data[index];
}

void * manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_function__ManusNodeHierarchy__poses(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Pose__Sequence * member =
    (geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  return &member->data[index];
}

void manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__fetch_function__ManusNodeHierarchy__poses(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Pose * item =
    ((const geometry_msgs__msg__Pose *)
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_const_function__ManusNodeHierarchy__poses(untyped_member, index));
  geometry_msgs__msg__Pose * value =
    (geometry_msgs__msg__Pose *)(untyped_value);
  *value = *item;
}

void manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__assign_function__ManusNodeHierarchy__poses(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Pose * item =
    ((geometry_msgs__msg__Pose *)
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_function__ManusNodeHierarchy__poses(untyped_member, index));
  const geometry_msgs__msg__Pose * value =
    (const geometry_msgs__msg__Pose *)(untyped_value);
  *item = *value;
}

bool manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__resize_function__ManusNodeHierarchy__poses(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Pose__Sequence * member =
    (geometry_msgs__msg__Pose__Sequence *)(untyped_member);
  geometry_msgs__msg__Pose__Sequence__fini(member);
  return geometry_msgs__msg__Pose__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_message_member_array[5] = {
  {
    "glove_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusNodeHierarchy, glove_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "node_count",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusNodeHierarchy, node_count),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "node_ids",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusNodeHierarchy, node_ids),  // bytes offset in struct
    NULL,  // default value
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__size_function__ManusNodeHierarchy__node_ids,  // size() function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_const_function__ManusNodeHierarchy__node_ids,  // get_const(index) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_function__ManusNodeHierarchy__node_ids,  // get(index) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__fetch_function__ManusNodeHierarchy__node_ids,  // fetch(index, &value) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__assign_function__ManusNodeHierarchy__node_ids,  // assign(index, value) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__resize_function__ManusNodeHierarchy__node_ids  // resize(index) function pointer
  },
  {
    "parent_node_ids",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusNodeHierarchy, parent_node_ids),  // bytes offset in struct
    NULL,  // default value
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__size_function__ManusNodeHierarchy__parent_node_ids,  // size() function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_const_function__ManusNodeHierarchy__parent_node_ids,  // get_const(index) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_function__ManusNodeHierarchy__parent_node_ids,  // get(index) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__fetch_function__ManusNodeHierarchy__parent_node_ids,  // fetch(index, &value) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__assign_function__ManusNodeHierarchy__parent_node_ids,  // assign(index, value) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__resize_function__ManusNodeHierarchy__parent_node_ids  // resize(index) function pointer
  },
  {
    "poses",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(manus_ros2_msgs__msg__ManusNodeHierarchy, poses),  // bytes offset in struct
    NULL,  // default value
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__size_function__ManusNodeHierarchy__poses,  // size() function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_const_function__ManusNodeHierarchy__poses,  // get_const(index) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__get_function__ManusNodeHierarchy__poses,  // get(index) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__fetch_function__ManusNodeHierarchy__poses,  // fetch(index, &value) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__assign_function__ManusNodeHierarchy__poses,  // assign(index, value) function pointer
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__resize_function__ManusNodeHierarchy__poses  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_message_members = {
  "manus_ros2_msgs__msg",  // message namespace
  "ManusNodeHierarchy",  // message name
  5,  // number of fields
  sizeof(manus_ros2_msgs__msg__ManusNodeHierarchy),
  manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_message_member_array,  // message members
  manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_init_function,  // function to initialize message memory (memory has to be allocated)
  manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_message_type_support_handle = {
  0,
  &manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_manus_ros2_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, manus_ros2_msgs, msg, ManusNodeHierarchy)() {
  manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  if (!manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_message_type_support_handle.typesupport_identifier) {
    manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &manus_ros2_msgs__msg__ManusNodeHierarchy__rosidl_typesupport_introspection_c__ManusNodeHierarchy_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
