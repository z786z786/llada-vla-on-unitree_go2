#pragma once

#include <filesystem>
#include <optional>

namespace collector::input
{

std::optional<std::filesystem::path> FindDefaultKeyboardDevice();
float SlewTowards(float current, float target, float accelPerSecond, float decelPerSecond, float deltaSeconds);
float QuantizeWirelessAxisToFullScale(float value, float threshold);

}  // namespace collector::input
