// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from manus_ros2_msgs:msg/ManusNodeHierarchy.idl
// generated code does not contain a copyright notice
#include "manus_ros2_msgs/msg/detail/manus_node_hierarchy__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `node_ids`
// Member `parent_node_ids`
#include "rosidl_runtime_c/primitives_sequence_functions.h"
// Member `poses`
#include "geometry_msgs/msg/detail/pose__functions.h"

bool
manus_ros2_msgs__msg__ManusNodeHierarchy__init(manus_ros2_msgs__msg__ManusNodeHierarchy * msg)
{
  if (!msg) {
    return false;
  }
  // glove_id
  // node_count
  // node_ids
  if (!rosidl_runtime_c__uint32__Sequence__init(&msg->node_ids, 0)) {
    manus_ros2_msgs__msg__ManusNodeHierarchy__fini(msg);
    return false;
  }
  // parent_node_ids
  if (!rosidl_runtime_c__uint32__Sequence__init(&msg->parent_node_ids, 0)) {
    manus_ros2_msgs__msg__ManusNodeHierarchy__fini(msg);
    return false;
  }
  // poses
  if (!geometry_msgs__msg__Pose__Sequence__init(&msg->poses, 0)) {
    manus_ros2_msgs__msg__ManusNodeHierarchy__fini(msg);
    return false;
  }
  return true;
}

void
manus_ros2_msgs__msg__ManusNodeHierarchy__fini(manus_ros2_msgs__msg__ManusNodeHierarchy * msg)
{
  if (!msg) {
    return;
  }
  // glove_id
  // node_count
  // node_ids
  rosidl_runtime_c__uint32__Sequence__fini(&msg->node_ids);
  // parent_node_ids
  rosidl_runtime_c__uint32__Sequence__fini(&msg->parent_node_ids);
  // poses
  geometry_msgs__msg__Pose__Sequence__fini(&msg->poses);
}

bool
manus_ros2_msgs__msg__ManusNodeHierarchy__are_equal(const manus_ros2_msgs__msg__ManusNodeHierarchy * lhs, const manus_ros2_msgs__msg__ManusNodeHierarchy * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // glove_id
  if (lhs->glove_id != rhs->glove_id) {
    return false;
  }
  // node_count
  if (lhs->node_count != rhs->node_count) {
    return false;
  }
  // node_ids
  if (!rosidl_runtime_c__uint32__Sequence__are_equal(
      &(lhs->node_ids), &(rhs->node_ids)))
  {
    return false;
  }
  // parent_node_ids
  if (!rosidl_runtime_c__uint32__Sequence__are_equal(
      &(lhs->parent_node_ids), &(rhs->parent_node_ids)))
  {
    return false;
  }
  // poses
  if (!geometry_msgs__msg__Pose__Sequence__are_equal(
      &(lhs->poses), &(rhs->poses)))
  {
    return false;
  }
  return true;
}

bool
manus_ros2_msgs__msg__ManusNodeHierarchy__copy(
  const manus_ros2_msgs__msg__ManusNodeHierarchy * input,
  manus_ros2_msgs__msg__ManusNodeHierarchy * output)
{
  if (!input || !output) {
    return false;
  }
  // glove_id
  output->glove_id = input->glove_id;
  // node_count
  output->node_count = input->node_count;
  // node_ids
  if (!rosidl_runtime_c__uint32__Sequence__copy(
      &(input->node_ids), &(output->node_ids)))
  {
    return false;
  }
  // parent_node_ids
  if (!rosidl_runtime_c__uint32__Sequence__copy(
      &(input->parent_node_ids), &(output->parent_node_ids)))
  {
    return false;
  }
  // poses
  if (!geometry_msgs__msg__Pose__Sequence__copy(
      &(input->poses), &(output->poses)))
  {
    return false;
  }
  return true;
}

manus_ros2_msgs__msg__ManusNodeHierarchy *
manus_ros2_msgs__msg__ManusNodeHierarchy__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusNodeHierarchy * msg = (manus_ros2_msgs__msg__ManusNodeHierarchy *)allocator.allocate(sizeof(manus_ros2_msgs__msg__ManusNodeHierarchy), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(manus_ros2_msgs__msg__ManusNodeHierarchy));
  bool success = manus_ros2_msgs__msg__ManusNodeHierarchy__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
manus_ros2_msgs__msg__ManusNodeHierarchy__destroy(manus_ros2_msgs__msg__ManusNodeHierarchy * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    manus_ros2_msgs__msg__ManusNodeHierarchy__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence__init(manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusNodeHierarchy * data = NULL;

  if (size) {
    data = (manus_ros2_msgs__msg__ManusNodeHierarchy *)allocator.zero_allocate(size, sizeof(manus_ros2_msgs__msg__ManusNodeHierarchy), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = manus_ros2_msgs__msg__ManusNodeHierarchy__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        manus_ros2_msgs__msg__ManusNodeHierarchy__fini(&data[i - 1]);
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
manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence__fini(manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence * array)
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
      manus_ros2_msgs__msg__ManusNodeHierarchy__fini(&array->data[i]);
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

manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence *
manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence * array = (manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence *)allocator.allocate(sizeof(manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence__destroy(manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence__are_equal(const manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence * lhs, const manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!manus_ros2_msgs__msg__ManusNodeHierarchy__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence__copy(
  const manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence * input,
  manus_ros2_msgs__msg__ManusNodeHierarchy__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(manus_ros2_msgs__msg__ManusNodeHierarchy);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    manus_ros2_msgs__msg__ManusNodeHierarchy * data =
      (manus_ros2_msgs__msg__ManusNodeHierarchy *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!manus_ros2_msgs__msg__ManusNodeHierarchy__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          manus_ros2_msgs__msg__ManusNodeHierarchy__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!manus_ros2_msgs__msg__ManusNodeHierarchy__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
