#include "collector_input_backend.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <optional>
#include <poll.h>
#include <thread>
#include <unistd.h>

#include "collector_input_utils.h"
#include "wireless_gamepad.h"

namespace collector::input
{

std::optional<TeleopEvent> EventFromTerminalChar(char ch);

namespace
{

constexpr float kWirelessButtonYawScale = 0.6f;

struct WirelessControllerSnapshot
{
    Gamepad gamepad;
    double timestamp = 0.0;
    uint64_t sequence = 0;
    bool valid = false;
};

struct PendingWirelessControllerActions
{
    bool start = false;
    bool A = false;
    bool Y = false;
    bool R2 = false;
    bool F1 = false;
    bool F2 = false;
    bool up = false;
    bool right = false;
    bool down = false;
    bool left = false;

    bool Any() const
    {
        return start || A || Y || R2 || F1 || F2 || up || right || down || left;
    }

    void MergeFrom(const Gamepad& gamepad)
    {
        start = start || gamepad.start.onPress;
        A = A || gamepad.A.onPress;
        Y = Y || gamepad.Y.onPress;
        R2 = R2 || gamepad.R2.onPress;
        F1 = F1 || gamepad.F1.onPress;
        F2 = F2 || gamepad.F2.onPress;
        up = up || gamepad.up.onPress;
        right = right || gamepad.right.onPress;
        down = down || gamepad.down.onPress;
        left = left || gamepad.left.onPress;
    }
};

class WirelessInputBackend final : public InputBackend
{
public:
    explicit WirelessInputBackend(const BackendConfig& config)
        : config_(config)
    {
        snapshot_.gamepad.deadZone = config_.wirelessStickDeadZone;
        snapshot_.gamepad.smooth = config_.wirelessStickSmoothing;
    }

    void Start() override
    {
        ResetMotionState();
    }

    void Stop() override
    {
        ResetMotionState();
    }

    std::vector<TeleopEvent> PollEvents(bool promptActive, std::chrono::milliseconds timeout) override
    {
        std::vector<TeleopEvent> events;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            labelModeActive_ = promptActive;
            if (pendingActions_.start)
            {
                events.push_back({TeleopEventType::StartCapture});
            }
            if (pendingActions_.A)
            {
                events.push_back({TeleopEventType::StopCapture});
            }
            if (pendingActions_.Y)
            {
                events.push_back({TeleopEventType::CancelSegment});
            }
            if (pendingActions_.R2)
            {
                events.push_back({TeleopEventType::EmergencyStop});
            }
            if (pendingActions_.F1)
            {
                events.push_back({TeleopEventType::ClearFault});
            }
            if (pendingActions_.F2)
            {
                events.push_back({TeleopEventType::ToggleStand});
            }
            if (promptActive && pendingActions_.up)
            {
                events.push_back({TeleopEventType::SubmitLabelShortcut, '1'});
            }
            if (promptActive && pendingActions_.right)
            {
                events.push_back({TeleopEventType::SubmitLabelShortcut, '2'});
            }
            if (promptActive && pendingActions_.down)
            {
                events.push_back({TeleopEventType::SubmitLabelShortcut, '3'});
            }
            if (promptActive && pendingActions_.left)
            {
                events.push_back({TeleopEventType::SubmitLabelShortcut, '4'});
            }
            pendingActions_ = PendingWirelessControllerActions{};
        }

        if (config_.terminalRawEnabled && !promptActive)
        {
            pollfd pfd{STDIN_FILENO, POLLIN, 0};
            const int pollResult = ::poll(&pfd, 1, 0);
            if (pollResult > 0 && (pfd.revents & POLLIN) != 0)
            {
                char ch = 0;
                const ssize_t bytesRead = ::read(STDIN_FILENO, &ch, 1);
                if (bytesRead > 0)
                {
                    if (const auto event = EventFromTerminalChar(ch); event.has_value())
                    {
                        events.push_back(event.value());
                    }
                }
            }
        }

        if (events.empty() && timeout.count() > 0)
        {
            std::this_thread::sleep_for(timeout);
        }
        return events;
    }

    MotionState GetMotionState(double nowSeconds, std::chrono::steady_clock::time_point nowSteady) const override
    {
        (void)nowSteady;
        MotionState motion;
        WirelessControllerSnapshot snapshot;
        bool labelModeActive = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot = snapshot_;
            labelModeActive = labelModeActive_;
        }
        const double ageSeconds = nowSeconds - snapshot.timestamp;
        if (!snapshot.valid || ageSeconds < 0.0 || ageSeconds > config_.wirelessTimeoutSeconds)
        {
            return motion;
        }

        if (!labelModeActive)
        {
            motion.forward = snapshot.gamepad.up.pressed;
            motion.backward = snapshot.gamepad.down.pressed;
            motion.left = snapshot.gamepad.left.pressed;
            motion.right = snapshot.gamepad.right.pressed;
            motion.yawLeft = snapshot.gamepad.X.pressed;
            motion.yawRight = snapshot.gamepad.B.pressed;
        }
        return motion;
    }

    CommandIntent ComputeCommand(const ComputeCommandOptions& options) override
    {
        WirelessControllerSnapshot snapshot;
        bool labelModeActive = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot = snapshot_;
            labelModeActive = labelModeActive_;
        }

        float targetSendVx = 0.0f;
        float targetSendVy = 0.0f;
        float targetSendWz = 0.0f;
        const double ageSeconds = options.nowSeconds - snapshot.timestamp;
        if (snapshot.valid && ageSeconds >= 0.0 && ageSeconds <= config_.wirelessTimeoutSeconds)
        {
            const bool discreteMotionRequested =
                !labelModeActive && !options.stopRequested &&
                (snapshot.gamepad.up.pressed || snapshot.gamepad.down.pressed ||
                 snapshot.gamepad.left.pressed || snapshot.gamepad.right.pressed ||
                 snapshot.gamepad.X.pressed || snapshot.gamepad.B.pressed);
            if (discreteMotionRequested)
            {
                targetSendVx = (snapshot.gamepad.up.pressed ? 1.0f : 0.0f) -
                               (snapshot.gamepad.down.pressed ? 1.0f : 0.0f);
                targetSendVy = (snapshot.gamepad.left.pressed ? 1.0f : 0.0f) -
                               (snapshot.gamepad.right.pressed ? 1.0f : 0.0f);
                targetSendWz = ((snapshot.gamepad.X.pressed ? 1.0f : 0.0f) -
                                (snapshot.gamepad.B.pressed ? 1.0f : 0.0f)) * kWirelessButtonYawScale;
            }
        }

        targetSendVx = std::clamp(targetSendVx, -1.0f, 1.0f);
        targetSendVy = std::clamp(targetSendVy, -1.0f, 1.0f);
        targetSendWz = std::clamp(targetSendWz, -1.0f, 1.0f);

        const float planarNorm = std::sqrt(targetSendVx * targetSendVx + targetSendVy * targetSendVy);
        if (planarNorm > 1.0f)
        {
            targetSendVx /= planarNorm;
            targetSendVy /= planarNorm;
        }

        const float deltaSeconds = lastCommandUpdate_.time_since_epoch().count() == 0
                                       ? 0.0f
                                       : std::chrono::duration<float>(options.nowSteady - lastCommandUpdate_).count();
        lastCommandUpdate_ = options.nowSteady;

        // Match the evdev backend feel: button-derived wireless commands use the
        // same slew-limited ramp instead of hard step changes.
        smoothedVx_ = SlewTowards(
            smoothedVx_, targetSendVx, config_.keyboardLinearAccelPerSecond, config_.keyboardLinearDecelPerSecond, deltaSeconds);
        smoothedVy_ = SlewTowards(
            smoothedVy_, targetSendVy, config_.keyboardLinearAccelPerSecond, config_.keyboardLinearDecelPerSecond, deltaSeconds);
        smoothedWz_ = SlewTowards(
            smoothedWz_, targetSendWz, config_.keyboardYawAccelPerSecond, config_.keyboardYawDecelPerSecond, deltaSeconds);

        const bool sendingMotion =
                                   (std::fabs(smoothedVx_) > 1e-6f ||
                                    std::fabs(smoothedVy_) > 1e-6f ||
                                    std::fabs(smoothedWz_) > 1e-6f);

        CommandIntent intent;
        intent.timestamp = options.nowSeconds;
        intent.vx = smoothedVx_ * options.cmdVxMax;
        intent.vy = smoothedVy_ * options.cmdVyMax;
        intent.wz = smoothedWz_ * options.cmdWzMax;
        intent.sendVx = smoothedVx_ * options.cmdVxMax;
        intent.sendVy = smoothedVy_ * options.cmdVyMax;
        intent.sendWz = smoothedWz_ * options.cmdWzMax;
        intent.shouldSendMotion = sendingMotion;
        intent.valid = true;
        return intent;
    }

    void ResetMotionState() override
    {
        smoothedVx_ = 0.0f;
        smoothedVy_ = 0.0f;
        smoothedWz_ = 0.0f;
        lastCommandUpdate_ = std::chrono::steady_clock::now();
    }

    void AppendStatus(std::ostream& output, double nowSeconds) const override
    {
        WirelessControllerSnapshot snapshot;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot = snapshot_;
        }
        const double ageSeconds = nowSeconds - snapshot.timestamp;
        if (!snapshot.valid || ageSeconds < 0.0 || ageSeconds > config_.wirelessTimeoutSeconds)
        {
            output << " controller_axes=(stale)";
            return;
        }
        output << " controller_axes=("
               << "ly=" << snapshot.gamepad.rawLy
               << ",lx=" << snapshot.gamepad.rawLx
               << ",rx=" << snapshot.gamepad.rawRx
               << ")";
    }

    void AppendHelp(std::ostream& output) const override
    {
        output << "  sticks keep native behavior" << std::endl;
        output << "  D-pad up/down/left/right = W/S/A/D style motion, X/B = Q/E style yaw" << std::endl;
        output << "  when label_ready, D-pad up/right/down/left switches to label 1/2/3/4" << std::endl;
        output << "  Start start capture  A stop capture  Y discard current segment" << std::endl;
        output << "  R2 emergency stop  F1 clear fault  F2 toggle stand up/down" << std::endl;
    }

    std::string BackendName() const override
    {
        return "wireless_controller";
    }

    void HandleWirelessControllerMessage(const unitree_go::msg::dds_::WirelessController_& message) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.gamepad.Update(message);
        pendingActions_.MergeFrom(snapshot_.gamepad);
        snapshot_.timestamp = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        snapshot_.sequence += 1;
        snapshot_.valid = true;
    }

private:
    BackendConfig config_;
    mutable std::mutex mutex_;
    WirelessControllerSnapshot snapshot_{};
    PendingWirelessControllerActions pendingActions_{};
    bool labelModeActive_ = false;
    std::chrono::steady_clock::time_point lastCommandUpdate_{};
    float smoothedVx_ = 0.0f;
    float smoothedVy_ = 0.0f;
    float smoothedWz_ = 0.0f;
};

}  // namespace

std::unique_ptr<InputBackend> CreateWirelessInputBackend(const BackendConfig& config)
{
    return std::make_unique<WirelessInputBackend>(config);
}

}  // namespace collector::input
