#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <unitree/idl/go2/WirelessController_.hpp>

namespace collector::input
{

enum class BackendKind
{
    WirelessController,
    Evdev,
};

enum class TeleopEventType
{
    StartCapture,
    StopCapture,
    CancelSegment,
    EmergencyStop,
    ClearFault,
    ToggleStand,
    PrintStatus,
    PrintHelp,
    Quit,
    SubmitLabelShortcut,
};

struct TeleopEvent
{
    TeleopEventType type;
    char shortcut = 0;
};

struct MotionState
{
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool yawLeft = false;
    bool yawRight = false;
};

struct CommandIntent
{
    double timestamp = 0.0;
    float vx = 0.0f;
    float vy = 0.0f;
    float wz = 0.0f;
    float sendVx = 0.0f;
    float sendVy = 0.0f;
    float sendWz = 0.0f;
    bool shouldSendMotion = false;
    bool valid = false;
};

struct BackendConfig
{
    BackendKind kind = BackendKind::WirelessController;
    std::filesystem::path inputDevice;
    bool terminalRawEnabled = false;
    double wirelessTimeoutSeconds = 1.0;
    float keyboardLinearAccelPerSecond = 0.0f;
    float keyboardLinearDecelPerSecond = 0.0f;
    float keyboardYawAccelPerSecond = 0.0f;
    float keyboardYawDecelPerSecond = 0.0f;
    float linearAccelPerSecond = 0.0f;
    float linearDecelPerSecond = 0.0f;
    float yawAccelPerSecond = 0.0f;
    float yawDecelPerSecond = 0.0f;
    float wirelessDiscreteThreshold = 0.0f;
    float wirelessStickDeadZone = 0.0f;
    float wirelessStickSmoothing = 0.0f;
    float wirelessCollectorDeadZone = 0.0f;
    float wirelessCollectorReleaseZone = 0.0f;
    float wirelessAxisExponent = 1.0f;
};

struct ComputeCommandOptions
{
    bool stopRequested = false;
    double nowSeconds = 0.0;
    std::chrono::steady_clock::time_point nowSteady{};
    bool wirelessNativePassthrough = false;
    float cmdVxMax = 0.0f;
    float cmdVyMax = 0.0f;
    float cmdWzMax = 0.0f;
};

class InputBackend
{
public:
    virtual ~InputBackend() = default;

    virtual void Start() = 0;
    virtual void Stop() = 0;
    virtual std::vector<TeleopEvent> PollEvents(bool promptActive, std::chrono::milliseconds timeout) = 0;
    virtual MotionState GetMotionState(double nowSeconds, std::chrono::steady_clock::time_point nowSteady) const = 0;
    virtual CommandIntent ComputeCommand(const ComputeCommandOptions& options) = 0;
    virtual void ResetMotionState() = 0;
    virtual void AppendStatus(std::ostream& output, double nowSeconds) const = 0;
    virtual void AppendHelp(std::ostream& output) const = 0;
    virtual std::string BackendName() const = 0;

    virtual void HandleWirelessControllerMessage(const unitree_go::msg::dds_::WirelessController_& message)
    {
        (void)message;
    }
};

std::unique_ptr<InputBackend> CreateInputBackend(const BackendConfig& config);

}  // namespace collector::input
