// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from manus_ros2_msgs:msg/ManusNodePoses.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__STRUCT_H_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'node_ids'
#include "rosidl_runtime_c/primitives_sequence.h"
// Member 'poses'
#include "geometry_msgs/msg/detail/pose__struct.h"

/// Struct defined in msg/ManusNodePoses in the package manus_ros2_msgs.
typedef struct manus_ros2_msgs__msg__ManusNodePoses
{
  uint32_t glove_id;
  int32_t node_count;
  rosidl_runtime_c__uint32__Sequence node_ids;
  geometry_msgs__msg__Pose__Sequence poses;
} manus_ros2_msgs__msg__ManusNodePoses;

// Struct for a sequence of manus_ros2_msgs__msg__ManusNodePoses.
typedef struct manus_ros2_msgs__msg__ManusNodePoses__Sequence
{
  manus_ros2_msgs__msg__ManusNodePoses * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} manus_ros2_msgs__msg__ManusNodePoses__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__STRUCT_H_
