#include "collector_input_utils.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace collector::input
{

std::optional<std::filesystem::path> FindDefaultKeyboardDevice()
{
    namespace fs = std::filesystem;

    const fs::path byIdDir("/dev/input/by-id");
    if (fs::exists(byIdDir))
    {
        std::vector<std::pair<std::string, fs::path>> keyboardDevices;
        for (const auto& entry : fs::directory_iterator(byIdDir))
        {
            if (!entry.is_symlink())
            {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.find("-event-kbd") == std::string::npos)
            {
                continue;
            }
            std::error_code ec;
            const fs::path resolved = fs::canonical(entry.path(), ec);
            if (!ec)
            {
                keyboardDevices.emplace_back(name, resolved);
            }
        }

        if (!keyboardDevices.empty())
        {
            std::sort(
                keyboardDevices.begin(),
                keyboardDevices.end(),
                [](const auto& lhs, const auto& rhs)
                {
                    const bool lhsPrefersInterface = lhs.first.find("-if") != std::string::npos;
                    const bool rhsPrefersInterface = rhs.first.find("-if") != std::string::npos;
                    if (lhsPrefersInterface != rhsPrefersInterface)
                    {
                        return lhsPrefersInterface > rhsPrefersInterface;
                    }
                    return lhs.first < rhs.first;
                });
            return keyboardDevices.front().second;
        }
    }

    const fs::path inputDir("/dev/input");
    if (!fs::exists(inputDir))
    {
        return std::nullopt;
    }

    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        const std::string name = entry.path().filename().string();
        if (name.rfind("event", 0) == 0)
        {
            return entry.path();
        }
    }
    return std::nullopt;
}

float SlewTowards(float current, float target, float accelPerSecond, float decelPerSecond, float deltaSeconds)
{
    const float delta = target - current;
    if (std::fabs(delta) <= 1e-6f)
    {
        return target;
    }

    const bool accelerating = std::fabs(target) > std::fabs(current) ||
                              (std::fabs(target) > 1e-6f && current * target < 0.0f);
    const float maxStep = (accelerating ? accelPerSecond : decelPerSecond) * std::max(deltaSeconds, 0.0f);
    if (std::fabs(delta) <= maxStep)
    {
        return target;
    }
    return current + (delta > 0.0f ? maxStep : -maxStep);
}

float QuantizeWirelessAxisToFullScale(float value, float threshold)
{
    if (value > threshold)
    {
        return 1.0f;
    }
    if (value < -threshold)
    {
        return -1.0f;
    }
    return 0.0f;
}

}  // namespace collector::input
