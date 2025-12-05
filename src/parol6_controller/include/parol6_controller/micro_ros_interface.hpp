#ifndef MICRO_ROS_INTERFACE_HPP_
#define MICRO_ROS_INTERFACE_HPP_

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <std_msgs/msg/bool.hpp>

namespace micro_ros_controller
{
class MicroRosInterface : public hardware_interface::SystemInterface
{
public:
  MicroRosInterface();
  virtual ~MicroRosInterface() override;

  virtual hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & hardware_info) override;

  virtual std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  virtual std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  virtual hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;

  virtual hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

  virtual hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;

  virtual hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // Joint state subscriber callback
  void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg);

  // ROS 2 node and communication
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectoryPoint>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr mov_trigger_pub_;

  // Joint command and state storage
  std::vector<double> position_commands_;
  std::vector<double> velocity_commands_;
  std::vector<double> position_states_;
  std::vector<double> velocity_states_;

  // Previous commands to track changes
  std::vector<double> prev_position_commands_;
  std::vector<double> prev_velocity_commands_;

  // Latest received joint states
  sensor_msgs::msg::JointState latest_joint_states_;
  std::mutex joint_states_mutex_;
};
}  // namespace micro_ros_controller

#endif  