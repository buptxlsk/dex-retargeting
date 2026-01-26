// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from manus_ros2_msgs:msg/ManusNodePoses.idl
// generated code does not contain a copyright notice

#ifndef MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__TRAITS_HPP_
#define MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "manus_ros2_msgs/msg/detail/manus_node_poses__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'poses'
#include "geometry_msgs/msg/detail/pose__traits.hpp"

namespace manus_ros2_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ManusNodePoses & msg,
  std::ostream & out)
{
  out << "{";
  // member: glove_id
  {
    out << "glove_id: ";
    rosidl_generator_traits::value_to_yaml(msg.glove_id, out);
    out << ", ";
  }

  // member: node_count
  {
    out << "node_count: ";
    rosidl_generator_traits::value_to_yaml(msg.node_count, out);
    out << ", ";
  }

  // member: node_ids
  {
    if (msg.node_ids.size() == 0) {
      out << "node_ids: []";
    } else {
      out << "node_ids: [";
      size_t pending_items = msg.node_ids.size();
      for (auto item : msg.node_ids) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: poses
  {
    if (msg.poses.size() == 0) {
      out << "poses: []";
    } else {
      out << "poses: [";
      size_t pending_items = msg.poses.size();
      for (auto item : msg.poses) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ManusNodePoses & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: glove_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "glove_id: ";
    rosidl_generator_traits::value_to_yaml(msg.glove_id, out);
    out << "\n";
  }

  // member: node_count
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "node_count: ";
    rosidl_generator_traits::value_to_yaml(msg.node_count, out);
    out << "\n";
  }

  // member: node_ids
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.node_ids.size() == 0) {
      out << "node_ids: []\n";
    } else {
      out << "node_ids:\n";
      for (auto item : msg.node_ids) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: poses
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.poses.size() == 0) {
      out << "poses: []\n";
    } else {
      out << "poses:\n";
      for (auto item : msg.poses) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ManusNodePoses & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace manus_ros2_msgs

namespace rosidl_generator_traits
{

[[deprecated("use manus_ros2_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const manus_ros2_msgs::msg::ManusNodePoses & msg,
  std::ostream & out, size_t indentation = 0)
{
  manus_ros2_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use manus_ros2_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const manus_ros2_msgs::msg::ManusNodePoses & msg)
{
  return manus_ros2_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<manus_ros2_msgs::msg::ManusNodePoses>()
{
  return "manus_ros2_msgs::msg::ManusNodePoses";
}

template<>
inline const char * name<manus_ros2_msgs::msg::ManusNodePoses>()
{
  return "manus_ros2_msgs/msg/ManusNodePoses";
}

template<>
struct has_fixed_size<manus_ros2_msgs::msg::ManusNodePoses>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<manus_ros2_msgs::msg::ManusNodePoses>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<manus_ros2_msgs::msg::ManusNodePoses>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MANUS_ROS2_MSGS__MSG__DETAIL__MANUS_NODE_POSES__TRAITS_HPP_
