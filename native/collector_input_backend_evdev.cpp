#include "collector_input_backend.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <linux/input.h>
#include <mutex>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#include "collector_input_utils.h"

namespace collector::input
{

std::optional<TeleopEvent> EventFromTerminalChar(char ch);
std::unique_ptr<InputBackend> CreateWirelessInputBackend(const BackendConfig& config);

namespace
{

class EvdevInputBackend final : public InputBackend
{
public:
    explicit EvdevInputBackend(const BackendConfig& config)
        : config_(config)
    {
    }

    void Start() override
    {
        if (config_.inputDevice.empty())
        {
            throw std::runtime_error("evdev input device path is empty");
        }
        fd_ = ::open(config_.inputDevice.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd_ < 0)
        {
            throw std::runtime_error("打开输入设备失败：" + config_.inputDevice.string());
        }
        ResetMotionState();
    }

    void Stop() override
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
        ResetMotionState();
    }

    std::vector<TeleopEvent> PollEvents(bool promptActive, std::chrono::milliseconds timeout) override
    {
        std::vector<TeleopEvent> events;
        std::vector<pollfd> pfds;
        pfds.push_back(pollfd{fd_, POLLIN, 0});
        if (config_.terminalRawEnabled)
        {
            pfds.push_back(pollfd{STDIN_FILENO, POLLIN, 0});
        }

        const int pollResult = ::poll(pfds.data(), static_cast<nfds_t>(pfds.size()), static_cast<int>(timeout.count()));
        if (pollResult <= 0)
        {
            return events;
        }

        if ((pfds[0].revents & POLLIN) != 0)
        {
            input_event event{};
            const ssize_t bytesRead = ::read(fd_, &event, sizeof(event));
            if (bytesRead == static_cast<ssize_t>(sizeof(event)) && event.type == EV_KEY)
            {
                const bool pressed = event.value != 0;
                HandleEvdevKey(event.code, pressed, promptActive, events);
            }
        }

        if (config_.terminalRawEnabled && pfds.size() > 1 && (pfds[1].revents & POLLIN) != 0 && !promptActive)
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

        return events;
    }

    MotionState GetMotionState(double nowSeconds, std::chrono::steady_clock::time_point nowSteady) const override
    {
        (void)nowSeconds;
        MotionState motion;
        std::lock_guard<std::mutex> lock(mutex_);
        motion.forward = keyW_.pressed || keyW_.deadline > nowSteady;
        motion.backward = keyS_.pressed || keyS_.deadline > nowSteady;
        motion.left = keyA_.pressed || keyA_.deadline > nowSteady;
        motion.right = keyD_.pressed || keyD_.deadline > nowSteady;
        motion.yawLeft = keyQ_.pressed || keyQ_.deadline > nowSteady;
        motion.yawRight = keyE_.pressed || keyE_.deadline > nowSteady;
        return motion;
    }

    CommandIntent ComputeCommand(const ComputeCommandOptions& options) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const float deltaSeconds = lastCommandUpdate_.time_since_epoch().count() == 0
                                       ? 0.0f
                                       : std::chrono::duration<float>(options.nowSteady - lastCommandUpdate_).count();
        lastCommandUpdate_ = options.nowSteady;

        const float forward = options.stopRequested ? 0.0f : ((keyW_.pressed || keyW_.deadline > options.nowSteady) ? 1.0f : 0.0f);
        const float backward = options.stopRequested ? 0.0f : ((keyS_.pressed || keyS_.deadline > options.nowSteady) ? 1.0f : 0.0f);
        const float left = options.stopRequested ? 0.0f : ((keyA_.pressed || keyA_.deadline > options.nowSteady) ? 1.0f : 0.0f);
        const float right = options.stopRequested ? 0.0f : ((keyD_.pressed || keyD_.deadline > options.nowSteady) ? 1.0f : 0.0f);
        const float yawLeft = options.stopRequested ? 0.0f : ((keyQ_.pressed || keyQ_.deadline > options.nowSteady) ? 1.0f : 0.0f);
        const float yawRight = options.stopRequested ? 0.0f : ((keyE_.pressed || keyE_.deadline > options.nowSteady) ? 1.0f : 0.0f);

        float targetVx = forward - backward;
        float targetVy = left - right;
        float targetWz = yawLeft - yawRight;

        const float planarNorm = std::sqrt(targetVx * targetVx + targetVy * targetVy);
        if (planarNorm > 1.0f)
        {
            targetVx /= planarNorm;
            targetVy /= planarNorm;
        }

        smoothedVx_ = SlewTowards(
            smoothedVx_, targetVx, config_.keyboardLinearAccelPerSecond, config_.keyboardLinearDecelPerSecond, deltaSeconds);
        smoothedVy_ = SlewTowards(
            smoothedVy_, targetVy, config_.keyboardLinearAccelPerSecond, config_.keyboardLinearDecelPerSecond, deltaSeconds);
        smoothedWz_ = SlewTowards(
            smoothedWz_, targetWz, config_.keyboardYawAccelPerSecond, config_.keyboardYawDecelPerSecond, deltaSeconds);

        CommandIntent intent;
        intent.timestamp = options.nowSeconds;
        intent.vx = smoothedVx_ * options.cmdVxMax;
        intent.vy = smoothedVy_ * options.cmdVyMax;
        intent.wz = smoothedWz_ * options.cmdWzMax;
        intent.valid = true;
        return intent;
    }

    void ResetMotionState() override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        keyW_ = KeyActivity{};
        keyS_ = KeyActivity{};
        keyA_ = KeyActivity{};
        keyD_ = KeyActivity{};
        keyQ_ = KeyActivity{};
        keyE_ = KeyActivity{};
        smoothedVx_ = 0.0f;
        smoothedVy_ = 0.0f;
        smoothedWz_ = 0.0f;
        lastCommandUpdate_ = std::chrono::steady_clock::now();
    }

    void AppendStatus(std::ostream& output, double nowSeconds) const override
    {
        (void)nowSeconds;
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        output << " motion_keys=("
               << ((keyW_.pressed || keyW_.deadline > now) ? "W" : "")
               << ((keyS_.pressed || keyS_.deadline > now) ? "S" : "")
               << ((keyA_.pressed || keyA_.deadline > now) ? "A" : "")
               << ((keyD_.pressed || keyD_.deadline > now) ? "D" : "")
               << ((keyQ_.pressed || keyQ_.deadline > now) ? "Q" : "")
               << ((keyE_.pressed || keyE_.deadline > now) ? "E" : "")
               << ")";
    }

    void AppendHelp(std::ostream& output) const override
    {
        output << "  W/S forward-backward  A/D strafe  Q/E yaw" << std::endl;
        output << "  R start capture flow  T stop capture  ESC discard current segment" << std::endl;
        output << "  Space emergency stop  C clear fault  V toggle stand up/down  P status  H help  X quit" << std::endl;
        output << "  pending label: 1 good demo  2 usable imperfect  3 failed but valuable  4 discard" << std::endl;
    }

    std::string BackendName() const override
    {
        return "evdev";
    }

private:
    struct KeyActivity
    {
        bool pressed = false;
        std::chrono::steady_clock::time_point deadline{};
    };

    void SetKeyState(char ch, bool pressed)
    {
        KeyActivity* target = nullptr;
        switch (ch)
        {
        case 'w':
            target = &keyW_;
            break;
        case 's':
            target = &keyS_;
            break;
        case 'a':
            target = &keyA_;
            break;
        case 'd':
            target = &keyD_;
            break;
        case 'q':
            target = &keyQ_;
            break;
        case 'e':
            target = &keyE_;
            break;
        default:
            break;
        }
        if (target)
        {
            target->pressed = pressed;
            target->deadline = std::chrono::steady_clock::time_point{};
        }
    }

    void HandleEvdevKey(uint16_t code, bool pressed, bool promptActive, std::vector<TeleopEvent>& events)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        switch (code)
        {
        case KEY_W:
            SetKeyState('w', pressed);
            return;
        case KEY_S:
            SetKeyState('s', pressed);
            return;
        case KEY_A:
            SetKeyState('a', pressed);
            return;
        case KEY_D:
            SetKeyState('d', pressed);
            return;
        case KEY_Q:
            SetKeyState('q', pressed);
            return;
        case KEY_E:
            SetKeyState('e', pressed);
            return;
        case KEY_ESC:
            if (pressed)
            {
                events.push_back({TeleopEventType::CancelSegment});
            }
            return;
        case KEY_SPACE:
            if (pressed)
            {
                events.push_back({TeleopEventType::EmergencyStop});
            }
            return;
        case KEY_R:
            if (pressed)
            {
                events.push_back({TeleopEventType::StartCapture});
            }
            return;
        case KEY_T:
            if (pressed)
            {
                events.push_back({TeleopEventType::StopCapture});
            }
            return;
        case KEY_C:
            if (pressed)
            {
                events.push_back({TeleopEventType::ClearFault});
            }
            return;
        case KEY_P:
            if (pressed)
            {
                events.push_back({TeleopEventType::PrintStatus});
            }
            return;
        case KEY_H:
            if (pressed)
            {
                events.push_back({TeleopEventType::PrintHelp});
            }
            return;
        case KEY_X:
            if (pressed)
            {
                events.push_back({TeleopEventType::Quit});
            }
            return;
        default:
            return;
        }
    }

    BackendConfig config_;
    mutable std::mutex mutex_;
    int fd_ = -1;
    KeyActivity keyW_{};
    KeyActivity keyS_{};
    KeyActivity keyA_{};
    KeyActivity keyD_{};
    KeyActivity keyQ_{};
    KeyActivity keyE_{};
    std::chrono::steady_clock::time_point lastCommandUpdate_{};
    float smoothedVx_ = 0.0f;
    float smoothedVy_ = 0.0f;
    float smoothedWz_ = 0.0f;
};

}  // namespace

std::unique_ptr<InputBackend> CreateInputBackend(const BackendConfig& config)
{
    if (config.kind == BackendKind::WirelessController)
    {
        return CreateWirelessInputBackend(config);
    }
    return std::make_unique<EvdevInputBackend>(config);
}

}  // namespace collector::input
