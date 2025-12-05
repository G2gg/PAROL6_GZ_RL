#include "parol6_controller/parol6_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>


namespace parol6_controller
{
  
Parol6Interface::Parol6Interface()
{
}


Parol6Interface::~Parol6Interface()
{
  if (teensy_.IsOpen())
  {
    try
    {
      teensy_.Close();
    }
    catch (...)
    {
      RCLCPP_FATAL_STREAM(rclcpp::get_logger("Parol6Interface"),
                          "Something went wrong while closing connection with port " << port_);
    }
  }
}


CallbackReturn Parol6Interface::on_init(const hardware_interface::HardwareInfo &hardware_info)
{
  CallbackReturn result = hardware_interface::SystemInterface::on_init(hardware_info);
  if (result != CallbackReturn::SUCCESS)
  {
    return result;
  }

  try
  {
    port_ = info_.hardware_parameters.at("port");
  }
  catch (const std::out_of_range &e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("Parol6Interface"), "No Serial Port provided! Aborting");
    return CallbackReturn::FAILURE;
  }

  position_commands_.reserve(info_.joints.size());
  position_states_.reserve(info_.joints.size());
  prev_position_commands_.reserve(info_.joints.size());

  return CallbackReturn::SUCCESS;
}


std::vector<hardware_interface::StateInterface> Parol6Interface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  // Provide only a position Interafce
  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, & position_states_[i]));
  }

  return state_interfaces;
}


std::vector<hardware_interface::CommandInterface> Parol6Interface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  // Provide only a position Interafce
  for (size_t i = 0; i < info_.joints.size(); i++)
  {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, & position_commands_[i]));
  }

  return command_interfaces;
}


CallbackReturn Parol6Interface::on_activate(const rclcpp_lifecycle::State & previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("Parol6Interface"), "Starting robot hardware ...");

  // Reset commands and states
  position_commands_ = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  prev_position_commands_ = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  position_states_ = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  try
  {
    teensy_.Open(port_);
    teensy_.SetBaudRate(LibSerial::BaudRate::BAUD_115200);
  }
  catch (...)
  {
    RCLCPP_FATAL_STREAM(rclcpp::get_logger("Parol6Interface"),
                        "Something went wrong while interacting with port " << port_);
    return CallbackReturn::FAILURE;
  }

  RCLCPP_INFO(rclcpp::get_logger("Parol6Interface"),
              "Hardware started, ready to take commands");
  return CallbackReturn::SUCCESS;
}


CallbackReturn Parol6Interface::on_deactivate(const rclcpp_lifecycle::State &previous_state)
{
  RCLCPP_INFO(rclcpp::get_logger("Parol6Interface"), "Stopping robot hardware ...");

  if (teensy_.IsOpen())
  {
    try
    {
      teensy_.Close();
    }
    catch (...)
    {
      RCLCPP_FATAL_STREAM(rclcpp::get_logger("Parol6Interface"),
                          "Something went wrong while closing connection with port " << port_);
    }
  }

  RCLCPP_INFO(rclcpp::get_logger("Parol6Interface"), "Hardware stopped");
  return CallbackReturn::SUCCESS;
}


hardware_interface::return_type Parol6Interface::read(const rclcpp::Time &time,
                                                          const rclcpp::Duration &period)
{
  // Open Loop Control - assuming the robot is always where we command to be
  position_states_ = position_commands_;
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type Parol6Interface::write(const rclcpp::Time &time,
                                                           const rclcpp::Duration &period)
{
  if (position_commands_ == prev_position_commands_)
  {
    // Nothing changed, do not send any command
    return hardware_interface::return_type::OK;
  }

  char buffer[32];
  std::string msg;

  snprintf(buffer, sizeof(buffer), "%.3f", position_commands_.at(0) * 180.0 / M_PI);
  msg.append(buffer).append(",");

  snprintf(buffer, sizeof(buffer), "%.3f", position_commands_.at(1) * 180.0 / M_PI);
  msg.append(buffer).append(",");

  snprintf(buffer, sizeof(buffer), "%.3f", position_commands_.at(2) * 180.0 / M_PI);
  msg.append(buffer).append(",");

  snprintf(buffer, sizeof(buffer), "%.3f", position_commands_.at(3) * 180.0 / M_PI);
  msg.append(buffer).append(",");

  snprintf(buffer, sizeof(buffer), "%.3f", position_commands_.at(4) * 180.0 / M_PI);
  msg.append(buffer).append(",");

  snprintf(buffer, sizeof(buffer), "%.3f", position_commands_.at(5) * 180.0 / M_PI);
  msg.append(buffer).append("\n");


  try
  {
    RCLCPP_INFO_STREAM(rclcpp::get_logger("Parol6Interface"), "Sending new command " << msg);
    teensy_.Write(msg);
  }
  catch (...)
  {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("Parol6Interface"),
                        "Something went wrong while sending the message "
                            << msg << " to the port " << port_);
    return hardware_interface::return_type::ERROR;
  }

  prev_position_commands_ = position_commands_;

  return hardware_interface::return_type::OK;
}
}  // namespace parol6_controller
PLUGINLIB_EXPORT_CLASS(parol6_controller::Parol6Interface, hardware_interface::SystemInterface)
