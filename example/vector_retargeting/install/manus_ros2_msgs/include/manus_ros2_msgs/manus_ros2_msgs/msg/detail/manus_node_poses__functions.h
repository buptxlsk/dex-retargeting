// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from manus_ros2_msgs:msg/ManusNodePoses.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__FUNCTIONS_H_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "manus_ros2_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "manus_ros2_msgs/msg/detail/manus_node_poses__struct.h"

/// Initialize msg/ManusNodePoses message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * manus_ros2_msgs__msg__ManusNodePoses
 * )) before or use
 * manus_ros2_msgs__msg__ManusNodePoses__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusNodePoses__init(manus_ros2_msgs__msg__ManusNodePoses * msg);

/// Finalize msg/ManusNodePoses message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusNodePoses__fini(manus_ros2_msgs__msg__ManusNodePoses * msg);

/// Create msg/ManusNodePoses message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * manus_ros2_msgs__msg__ManusNodePoses__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
manus_ros2_msgs__msg__ManusNodePoses *
manus_ros2_msgs__msg__ManusNodePoses__create();

/// Destroy msg/ManusNodePoses message.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusNodePoses__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusNodePoses__destroy(manus_ros2_msgs__msg__ManusNodePoses * msg);

/// Check for msg/ManusNodePoses message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusNodePoses__are_equal(const manus_ros2_msgs__msg__ManusNodePoses * lhs, const manus_ros2_msgs__msg__ManusNodePoses * rhs);

/// Copy a msg/ManusNodePoses message.
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
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusNodePoses__copy(
  const manus_ros2_msgs__msg__ManusNodePoses * input,
  manus_ros2_msgs__msg__ManusNodePoses * output);

/// Initialize array of msg/ManusNodePoses messages.
/**
 * It allocates the memory for the number of elements and calls
 * manus_ros2_msgs__msg__ManusNodePoses__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusNodePoses__Sequence__init(manus_ros2_msgs__msg__ManusNodePoses__Sequence * array, size_t size);

/// Finalize array of msg/ManusNodePoses messages.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusNodePoses__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusNodePoses__Sequence__fini(manus_ros2_msgs__msg__ManusNodePoses__Sequence * array);

/// Create array of msg/ManusNodePoses messages.
/**
 * It allocates the memory for the array and calls
 * manus_ros2_msgs__msg__ManusNodePoses__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
manus_ros2_msgs__msg__ManusNodePoses__Sequence *
manus_ros2_msgs__msg__ManusNodePoses__Sequence__create(size_t size);

/// Destroy array of msg/ManusNodePoses messages.
/**
 * It calls
 * manus_ros2_msgs__msg__ManusNodePoses__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
void
manus_ros2_msgs__msg__ManusNodePoses__Sequence__destroy(manus_ros2_msgs__msg__ManusNodePoses__Sequence * array);

/// Check for msg/ManusNodePoses message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusNodePoses__Sequence__are_equal(const manus_ros2_msgs__msg__ManusNodePoses__Sequence * lhs, const manus_ros2_msgs__msg__ManusNodePoses__Sequence * rhs);

/// Copy an array of msg/ManusNodePoses messages.
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
ROSIDL_GENERATOR_C_PUBLIC_manus_ros2_msgs
bool
manus_ros2_msgs__msg__ManusNodePoses__Sequence__copy(
  const manus_ros2_msgs__msg__ManusNodePoses__Sequence * input,
  manus_ros2_msgs__msg__ManusNodePoses__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__FUNCTIONS_H_
