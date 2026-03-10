#pragma once

#include <rclcpp/rclcpp.hpp>

#include "concrete_block_registration_teaser/registration_types.hpp"

namespace concrete_block_registration_teaser
{

BlockRegistrationConfig
load_registration_config(rclcpp::Node & node);

}  // namespace concrete_block_registration_teaser
