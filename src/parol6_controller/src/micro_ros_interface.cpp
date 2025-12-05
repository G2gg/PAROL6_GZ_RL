#include "parol6_controller/micro_ros_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <mutex>

namespace micro_ros_controller
{
  
MicroRosInterface::MicroRosInterface() : node_(nullptr)
{
}

MicroRosInterface::~MicroRosInterface()
{
}

hardware_interface::CallbackReturn MicroRosInterface::on_init(
  const hardware_interface::HardwareInfo &hardware_info)
{
    // Standard initialization from parent class
    CallbackReturn result = hardware_interface::SystemInterface::on_init(hardware_info);
    if (result != CallbackReturn::SUCCESS)
    {
        return result;
    }

    // Create ROS 2 node for this hardware interface
    node_ = rclcpp::Node::make_shared("micro_ros_hardware_interface");

    // Reserve space for joint commands and states
    position_commands_.reserve(info_.joints.size());
    velocity_commands_.reserve(info_.joints.size());
    position_states_.reserve(info_.joints.size());
    velocity_states_.reserve(info_.joints.size());
    prev_position_commands_.reserve(info_.joints.size());
    prev_velocity_commands_.reserve(info_.joints.size());

    // Subscribe to joint states topic
    //   joint_states_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    //     "/state_interface_pos_vel", 10, 
    //     std::bind(&MicroRosInterface::joint_states_callback, this, std::placeholders::_1)
    //   );

    // Create publisher for joint trajectory commands
    trajectory_pub_ = node_->create_publisher<trajectory_msgs::msg::JointTrajectoryPoint>(
        "/parol6_controller/joint_trajectory_point", 10);

  
    return CallbackReturn::SUCCESS;
}

void MicroRosInterface::joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(joint_states_mutex_);
  latest_joint_states_ = *msg;
}

std::vector<hardware_interface::StateInterface> MicroRosInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  // Export position and velocity state interfaces
  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_states_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_states_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> MicroRosInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  // Export position and velocity command interfaces
  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &position_commands_[i]));
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &velocity_commands_[i]));
  }

  return command_interfaces;
}

hardware_interface::CallbackReturn MicroRosInterface::on_activate(
  const rclcpp_lifecycle::State & previous_state)
{
  RCLCPP_INFO(node_->get_logger(), "Starting robot hardware interface...");

  // Reset commands and states
  position_commands_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  velocity_commands_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  position_states_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  velocity_states_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  prev_position_commands_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  prev_velocity_commands_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  RCLCPP_INFO(node_->get_logger(), "Hardware interface started, ready to take commands");
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MicroRosInterface::on_deactivate(
  const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(node_->get_logger(), "Stopping robot hardware interface...");

  // Reset everything to zero
  position_commands_.clear();
  velocity_commands_.clear();
  position_states_.clear();
  velocity_states_.clear();

  RCLCPP_INFO(node_->get_logger(), "Hardware interface stopped");
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type MicroRosInterface::read(
  const rclcpp::Time &time, const rclcpp::Duration &period)
{
  // Read joint states from the latest received message
//   std::lock_guard<std::mutex> lock(joint_states_mutex_);
  
//   if (!latest_joint_states_.name.empty())
//   {
//     // Update position and velocity states
//     for (size_t i = 0; i < info_.joints.size(); i++)
//     {
//       // Find the joint in the received message
//       auto it = std::find(
//         latest_joint_states_.name.begin(), 
//         latest_joint_states_.name.end(), 
//         info_.joints[i].name
//       );
      
//       if (it != latest_joint_states_.name.end())
//       {
//         size_t index = std::distance(latest_joint_states_.name.begin(), it);
        
//         // Update position state if available
//         if (index < latest_joint_states_.position.size())
//         {
//           position_states_[i] = latest_joint_states_.position[index];
//         }
        
//         // Update velocity state if available
//         if (index < latest_joint_states_.velocity.size())
//         {
//           velocity_states_[i] = latest_joint_states_.velocity[index];
//         }
//       }
//     }
//   }

    position_states_ = position_commands_; // (OPEN LOOP)
    velocity_states_ = velocity_commands_; // (OPEN LOOP)
    return hardware_interface::return_type::OK;

    return hardware_interface::return_type::OK;
}

hardware_interface::return_type MicroRosInterface::write(
  const rclcpp::Time &time, const rclcpp::Duration &period)
{
  // Check if commands have changed
  bool commands_changed = 
    (position_commands_ != prev_position_commands_) || 
    (velocity_commands_ != prev_velocity_commands_);


  if (commands_changed)
  {
    // Log positions
    std::stringstream position_ss;
    position_ss << "Joint Positions: [";
    for (size_t i = 0; i < position_commands_.size(); ++i) {
      position_ss << position_commands_[i];
      if (i < position_commands_.size() - 1) {
        position_ss << ", ";
      }
    }
    position_ss << "]";
    RCLCPP_INFO(node_->get_logger(), "%s", position_ss.str().c_str());

    // Log velocities
    std::stringstream velocity_ss;
    velocity_ss << "Joint Velocities: [";
    for (size_t i = 0; i < velocity_commands_.size(); ++i) {
      velocity_ss << velocity_commands_[i];
      if (i < velocity_commands_.size() - 1) {
        velocity_ss << ", ";
      }
    }
    velocity_ss << "]";
    RCLCPP_INFO(node_->get_logger(), "%s", velocity_ss.str().c_str());

    // Alternative compact logging method
    RCLCPP_INFO_STREAM(node_->get_logger(), 
      "Detailed Joint Commands:" << 
      "\nPositions: " << position_ss.str() << 
      "\nVelocities: " << velocity_ss.str()
    );

        
    // Create a single trajectory point
    trajectory_msgs::msg::JointTrajectoryPoint point;
    
    // Set positions
    point.positions = position_commands_;
    
    // Set velocities (optional)
    point.velocities = velocity_commands_;
    
    // Set a default duration (you might want to adjust this)
    // point.time_from_start = rclcpp::Duration::from_seconds(0.05); // 0.5 seconds
    
    

    // Publish the trajectory
    trajectory_pub_->publish(point);

    // Update previous commands
    prev_position_commands_ = position_commands_;
    prev_velocity_commands_ = velocity_commands_;
  }

  return hardware_interface::return_type::OK;
}
}  // namespace micro_ros_controller

// Export the plugin
PLUGINLIB_EXPORT_CLASS(micro_ros_controller::MicroRosInterface, hardware_interface::SystemInterface)