// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from manus_ros2_msgs:msg/ManusNodeHierarchy.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_HIERARCHY__STRUCT_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_HIERARCHY__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'poses'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__manus_ros2_msgs__msg__ManusNodeHierarchy __attribute__((deprecated))
#else
# define DEPRECATED__manus_ros2_msgs__msg__ManusNodeHierarchy __declspec(deprecated)
#endif

namespace manus_ros2_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ManusNodeHierarchy_
{
  using Type = ManusNodeHierarchy_<ContainerAllocator>;

  explicit ManusNodeHierarchy_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->glove_id = 0ul;
      this->node_count = 0l;
    }
  }

  explicit ManusNodeHierarchy_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->glove_id = 0ul;
      this->node_count = 0l;
    }
  }

  // field types and members
  using _glove_id_type =
    uint32_t;
  _glove_id_type glove_id;
  using _node_count_type =
    int32_t;
  _node_count_type node_count;
  using _node_ids_type =
    std::vector<uint32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint32_t>>;
  _node_ids_type node_ids;
  using _parent_node_ids_type =
    std::vector<uint32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint32_t>>;
  _parent_node_ids_type parent_node_ids;
  using _poses_type =
    std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Pose_<ContainerAllocator>>>;
  _poses_type poses;

  // setters for named parameter idiom
  Type & set__glove_id(
    const uint32_t & _arg)
  {
    this->glove_id = _arg;
    return *this;
  }
  Type & set__node_count(
    const int32_t & _arg)
  {
    this->node_count = _arg;
    return *this;
  }
  Type & set__node_ids(
    const std::vector<uint32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint32_t>> & _arg)
  {
    this->node_ids = _arg;
    return *this;
  }
  Type & set__parent_node_ids(
    const std::vector<uint32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint32_t>> & _arg)
  {
    this->parent_node_ids = _arg;
    return *this;
  }
  Type & set__poses(
    const std::vector<geometry_msgs::msg::Pose_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Pose_<ContainerAllocator>>> & _arg)
  {
    this->poses = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator> *;
  using ConstRawPtr =
    const manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__manus_ros2_msgs__msg__ManusNodeHierarchy
    std::shared_ptr<manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__manus_ros2_msgs__msg__ManusNodeHierarchy
    std::shared_ptr<manus_ros2_msgs::msg::ManusNodeHierarchy_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ManusNodeHierarchy_ & other) const
  {
    if (this->glove_id != other.glove_id) {
      return false;
    }
    if (this->node_count != other.node_count) {
      return false;
    }
    if (this->node_ids != other.node_ids) {
      return false;
    }
    if (this->parent_node_ids != other.parent_node_ids) {
      return false;
    }
    if (this->poses != other.poses) {
      return false;
    }
    return true;
  }
  bool operator!=(const ManusNodeHierarchy_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ManusNodeHierarchy_

// alias to use template instance with default allocator
using ManusNodeHierarchy =
  manus_ros2_msgs::msg::ManusNodeHierarchy_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace manus_ros2_msgs

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_HIERARCHY__STRUCT_HPP_
