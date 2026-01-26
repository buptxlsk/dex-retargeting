// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from manus_ros2_msgs:msg/ManusNodePoses.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__BUILDER_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "manus_ros2_msgs/msg/detail/manus_node_poses__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace manus_ros2_msgs
{

namespace msg
{

namespace builder
{

class Init_ManusNodePoses_poses
{
public:
  explicit Init_ManusNodePoses_poses(::manus_ros2_msgs::msg::ManusNodePoses & msg)
  : msg_(msg)
  {}
  ::manus_ros2_msgs::msg::ManusNodePoses poses(::manus_ros2_msgs::msg::ManusNodePoses::_poses_type arg)
  {
    msg_.poses = std::move(arg);
    return std::move(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusNodePoses msg_;
};

class Init_ManusNodePoses_node_ids
{
public:
  explicit Init_ManusNodePoses_node_ids(::manus_ros2_msgs::msg::ManusNodePoses & msg)
  : msg_(msg)
  {}
  Init_ManusNodePoses_poses node_ids(::manus_ros2_msgs::msg::ManusNodePoses::_node_ids_type arg)
  {
    msg_.node_ids = std::move(arg);
    return Init_ManusNodePoses_poses(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusNodePoses msg_;
};

class Init_ManusNodePoses_node_count
{
public:
  explicit Init_ManusNodePoses_node_count(::manus_ros2_msgs::msg::ManusNodePoses & msg)
  : msg_(msg)
  {}
  Init_ManusNodePoses_node_ids node_count(::manus_ros2_msgs::msg::ManusNodePoses::_node_count_type arg)
  {
    msg_.node_count = std::move(arg);
    return Init_ManusNodePoses_node_ids(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusNodePoses msg_;
};

class Init_ManusNodePoses_glove_id
{
public:
  Init_ManusNodePoses_glove_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ManusNodePoses_node_count glove_id(::manus_ros2_msgs::msg::ManusNodePoses::_glove_id_type arg)
  {
    msg_.glove_id = std::move(arg);
    return Init_ManusNodePoses_node_count(msg_);
  }

private:
  ::manus_ros2_msgs::msg::ManusNodePoses msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::manus_ros2_msgs::msg::ManusNodePoses>()
{
  return manus_ros2_msgs::msg::builder::Init_ManusNodePoses_glove_id();
}

}  // namespace manus_ros2_msgs

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__BUILDER_HPP_
