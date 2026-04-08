#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <linux/input.h>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <poll.h>
#include <sstream>
#include <string>
#include <thread>
#include <termios.h>
#include <unistd.h>
#include <vector>

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/video/video_client.hpp>

namespace
{

namespace fs = std::filesystem;
constexpr char kDefaultDataDirName[] = "data";

constexpr char kSportStateTopic[] = "rt/sportmodestate";
constexpr char kSchemaVersion[] = "go2_local_dataset_v1";
constexpr double kStateTimeoutSeconds = 1.0;
constexpr double kMaxSafeAbsRollRad = 0.75;
constexpr double kMaxSafeAbsPitchRad = 0.75;
constexpr double kDefaultCmdVxMax = 0.5;
constexpr double kDefaultCmdVyMax = 0.3;
constexpr double kDefaultCmdWzMax = 0.8;
constexpr float kStandingBodyHeightThreshold = 0.18f;
constexpr auto kKeyboardHoldTimeout = std::chrono::milliseconds(900);
constexpr float kLinearAccelPerSecond = 3.0f;
constexpr float kLinearDecelPerSecond = 4.5f;
constexpr float kYawAccelPerSecond = 4.5f;
constexpr float kYawDecelPerSecond = 6.0f;
constexpr auto kCaptureStartDelay = std::chrono::milliseconds(500);
constexpr char kEscapeKey = 27;
constexpr char kBackspace = 127;
constexpr char kCtrlH = 8;

const std::vector<std::string> kAllowedInstructions = {
    "go forward",
    "move backward",
    "strafe left",
    "strafe right",
    "stand up",
    "lie down",
    "turn left",
    "turn right",
    "stay still",
};

struct Config
{
    enum class InputBackend
    {
        Evdev,
        Tty,
    };

    enum class CaptureMode
    {
        SingleAction,
        Trajectory,
    };

    std::string networkInterface;
    fs::path outputDir;
    double loopHz = 50.0;
    double videoPollHz = 10.0;
    InputBackend inputBackend = InputBackend::Evdev;
    CaptureMode captureMode = CaptureMode::SingleAction;
    fs::path inputDevice;
    std::string sceneId;
    std::string operatorId;
    std::string instruction;
    std::string taskFamily;
    std::string targetType;
    std::string targetDescription;
    std::string targetInstanceId;
    std::string collectorNotes;
    std::vector<std::string> taskTags;
    double cmdVxMax = kDefaultCmdVxMax;
    double cmdVyMax = kDefaultCmdVyMax;
    double cmdWzMax = kDefaultCmdWzMax;
};

struct TaskMetadata
{
    std::string instruction;
    std::string captureMode;
    std::string taskFamily;
    std::string targetType;
    std::string targetDescription;
    std::string targetInstanceId;
    std::vector<std::string> taskTags;
    std::string collectorNotes;
    std::string instructionSource;
};

struct LatestState
{
    double timestamp = 0.0;
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float velocityX = 0.0f;
    float velocityY = 0.0f;
    float velocityZ = 0.0f;
    float yawSpeed = 0.0f;
    float bodyHeight = 0.0f;
    uint8_t gaitType = 0;
    bool valid = false;
};

struct LatestImage
{
    double timestamp = 0.0;
    uint64_t sequence = 0;
    std::vector<uint8_t> jpegBytes;
    bool valid = false;
};

struct VelocityCommand
{
    double timestamp = 0.0;
    float vx = 0.0f;
    float vy = 0.0f;
    float wz = 0.0f;
    bool valid = false;
};

struct EffectiveControlAction
{
    VelocityCommand command;
    double timestamp = 0.0;
};

struct EpisodeFrame
{
    double timestamp = 0.0;
    double stateTimestamp = 0.0;
    double actionTimestamp = 0.0;
    double rawActionTimestamp = 0.0;
    double imageTimestamp = 0.0;
    float stateVx = 0.0f;
    float stateVy = 0.0f;
    float stateWz = 0.0f;
    float stateYaw = 0.0f;
    float rawActionVx = 0.0f;
    float rawActionVy = 0.0f;
    float rawActionWz = 0.0f;
    float controlActionVx = 0.0f;
    float controlActionVy = 0.0f;
    float controlActionWz = 0.0f;
    std::vector<uint8_t> jpegBytes;
};

struct EpisodeSummary
{
    std::string episodeId;
    std::string instruction;
    std::string captureMode;
    std::string taskFamily;
    std::string targetType;
    std::string targetDescription;
    std::string targetInstanceId;
    std::vector<std::string> taskTags;
    std::string collectorNotes;
    std::string instructionSource;
    std::string sceneId;
    std::string operatorId;
    size_t numFrames = 0;
    double startTimestamp = 0.0;
    double endTimestamp = 0.0;
};

enum class SafetyState
{
    SafeReady,
    FaultLatched,
    EstopLatched,
};

enum class CaptureState
{
    Idle,
    Armed,
    DelayBeforeLog,
    Capturing,
    Fault,
};

enum class MotionInstruction
{
    None,
    GoForward,
    MoveBackward,
    StrafeLeft,
    StrafeRight,
    TurnLeft,
    TurnRight,
    Mixed,
};

double NowSeconds()
{
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string Trim(const std::string& value)
{
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])))
    {
        ++begin;
    }

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])))
    {
        --end;
    }

    return value.substr(begin, end - begin);
}

std::string JsonEscape(const std::string& input)
{
    std::ostringstream oss;
    for (const char ch : input)
    {
        switch (ch)
        {
        case '\\':
            oss << "\\\\";
            break;
        case '"':
            oss << "\\\"";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            oss << ch;
            break;
        }
    }
    return oss.str();
}

std::string JsonString(const std::string& input)
{
    return "\"" + JsonEscape(input) + "\"";
}

std::string JsonStringArray(const std::vector<std::string>& values)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t index = 0; index < values.size(); ++index)
    {
        oss << JsonString(values[index]);
        if (index + 1 != values.size())
        {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

std::vector<std::string> SplitCsvList(const std::string& input)
{
    std::vector<std::string> values;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        const std::string trimmed = Trim(item);
        if (!trimmed.empty())
        {
            values.push_back(trimmed);
        }
    }
    return values;
}

std::string InferTaskFamily(const std::string& instruction)
{
    const std::string trimmed = Trim(instruction);
    if (std::find(kAllowedInstructions.begin(), kAllowedInstructions.end(), trimmed) != kAllowedInstructions.end())
    {
        return "legacy_motion";
    }

    std::string lowered;
    lowered.reserve(trimmed.size());
    for (const char ch : trimmed)
    {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (lowered.rfind("go to ", 0) == 0 || lowered.rfind("go to the ", 0) == 0 || lowered.rfind("approach ", 0) == 0 || lowered.rfind("approach the ", 0) == 0)
    {
        return "goal_navigation";
    }
    if (lowered.rfind("follow ", 0) == 0 || lowered.rfind("follow the ", 0) == 0)
    {
        return "visual_following";
    }
    if (lowered.rfind("go around ", 0) == 0 || lowered.rfind("go around the ", 0) == 0 || lowered.rfind("avoid ", 0) == 0 || lowered.rfind("avoid the ", 0) == 0)
    {
        return "obstacle_aware_navigation";
    }
    return std::string{};
}

TaskMetadata ResolveTaskMetadata(const TaskMetadata& configured, const std::string& fallbackInstruction)
{
    TaskMetadata resolved = configured;
    resolved.instruction = Trim(resolved.instruction);
    if (resolved.instruction.empty())
    {
        resolved.instruction = Trim(fallbackInstruction);
        resolved.instructionSource = "motion_label";
    }
    else if (resolved.instructionSource.empty())
    {
        resolved.instructionSource = "semantic_text";
    }
    if (resolved.taskFamily.empty())
    {
        resolved.taskFamily = InferTaskFamily(resolved.instruction);
    }
    return resolved;
}

double AgeSeconds(double timestampSeconds, double nowSeconds)
{
    if (timestampSeconds <= 0.0)
    {
        return -1.0;
    }
    return std::max(0.0, nowSeconds - timestampSeconds);
}

std::tm LocalTimeFromEpochSeconds(std::time_t seconds)
{
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &seconds);
#else
    localtime_r(&seconds, &localTime);
#endif
    return localTime;
}

std::string FormatTimestampForPath(double timestampSeconds, const char* format)
{
    const std::time_t seconds = static_cast<std::time_t>(timestampSeconds);
    const std::tm localTime = LocalTimeFromEpochSeconds(seconds);
    char buffer[64];
    if (std::strftime(buffer, sizeof(buffer), format, &localTime) == 0)
    {
        throw std::runtime_error("failed to format timestamp");
    }
    return buffer;
}

fs::path CollectorRootFromArgv0(const char* argv0)
{
    const fs::path executablePath = fs::absolute(argv0);
    return executablePath.parent_path().parent_path().parent_path();
}

std::string SafetyStateName(SafetyState state)
{
    switch (state)
    {
    case SafetyState::SafeReady:
        return "safe_ready";
    case SafetyState::FaultLatched:
        return "fault_latched";
    case SafetyState::EstopLatched:
        return "estop_latched";
    }
    return "unknown";
}

std::string CaptureStateName(CaptureState state)
{
    switch (state)
    {
    case CaptureState::Idle:
        return "idle";
    case CaptureState::Armed:
        return "armed";
    case CaptureState::DelayBeforeLog:
        return "delay_before_log";
    case CaptureState::Capturing:
        return "capturing";
    case CaptureState::Fault:
        return "fault";
    }
    return "unknown";
}

std::string MotionInstructionName(MotionInstruction motion)
{
    switch (motion)
    {
    case MotionInstruction::None:
        return "";
    case MotionInstruction::GoForward:
        return "go forward";
    case MotionInstruction::MoveBackward:
        return "move backward";
    case MotionInstruction::StrafeLeft:
        return "strafe left";
    case MotionInstruction::StrafeRight:
        return "strafe right";
    case MotionInstruction::TurnLeft:
        return "turn left";
    case MotionInstruction::TurnRight:
        return "turn right";
    case MotionInstruction::Mixed:
        return "mixed";
    }
    return "";
}

std::string InputBackendName(Config::InputBackend backend)
{
    switch (backend)
    {
    case Config::InputBackend::Evdev:
        return "evdev";
    case Config::InputBackend::Tty:
        return "tty";
    }
    return "unknown";
}

std::string CaptureModeName(Config::CaptureMode mode)
{
    switch (mode)
    {
    case Config::CaptureMode::SingleAction:
        return "single_action";
    case Config::CaptureMode::Trajectory:
        return "trajectory";
    }
    return "unknown";
}

std::optional<Config::InputBackend> ParseInputBackend(const std::string& value)
{
    if (value == "evdev")
    {
        return Config::InputBackend::Evdev;
    }
    if (value == "tty")
    {
        return Config::InputBackend::Tty;
    }
    return std::nullopt;
}

std::optional<Config::CaptureMode> ParseCaptureMode(const std::string& value)
{
    if (value == "single_action")
    {
        return Config::CaptureMode::SingleAction;
    }
    if (value == "trajectory")
    {
        return Config::CaptureMode::Trajectory;
    }
    return std::nullopt;
}

std::optional<fs::path> FindDefaultKeyboardDevice()
{
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
    const float maxStep = (accelerating ? accelPerSecond : decelPerSecond) *
                          std::max(deltaSeconds, 0.0f);
    if (std::fabs(delta) <= maxStep)
    {
        return target;
    }
    return current + (delta > 0.0f ? maxStep : -maxStep);
}

EffectiveControlAction ResolveControlAction(const VelocityCommand& rawAction, double sampleTimestamp)
{
    EffectiveControlAction resolved;
    resolved.timestamp = sampleTimestamp;
    resolved.command = rawAction;
    resolved.command.timestamp = sampleTimestamp;
    resolved.command.valid = true;
    return resolved;
}

void PrintUsage(const char* program)
{
    std::cout
        << "Usage: " << program << " --network-interface IFACE --scene-id SCENE --operator-id OPERATOR [options]\n\n"
        << "Options:\n"
        << "  --output-dir PATH        Dataset root directory (default: <collector>/" << kDefaultDataDirName << ")\n"
        << "  --loop-hz FLOAT          Control and logging loop frequency (default: 50.0)\n"
        << "  --video-poll-hz FLOAT    Camera polling frequency (default: 10.0)\n"
        << "  --input-backend MODE     Input backend: evdev or tty (default: evdev)\n"
        << "  --capture-mode MODE      Capture mode: single_action or trajectory (default: single_action)\n"
        << "  --input-device PATH      evdev device path (default: auto-detect keyboard)\n"
        << "  --scene-id TEXT          Required scene identifier\n"
        << "  --operator-id TEXT       Required operator identifier\n"
        << "  --instruction TEXT       Optional semantic instruction for all collected episodes in this run\n"
        << "  --task-family TEXT       Optional task family, e.g. goal_navigation / visual_following / obstacle_aware_navigation\n"
        << "  --target-type TEXT       Optional target type, e.g. door / person / obstacle\n"
        << "  --target-description TEXT  Optional free-text target description, e.g. red door on the left\n"
        << "  --target-instance-id TEXT  Optional target instance identifier used by the operator during collection\n"
        << "  --task-tags CSV          Optional comma-separated tags, e.g. occluded,left_turn,low_light\n"
        << "  --collector-notes TEXT   Optional free-text notes stored with each episode\n"
        << "  --cmd-vx-max FLOAT       Max forward/backward speed in m/s\n"
        << "  --cmd-vy-max FLOAT       Max strafe speed in m/s\n"
        << "  --cmd-wz-max FLOAT       Max yaw speed in rad/s\n"
        << "  --help                   Show this help\n\n"
        << "Keyboard:\n"
        << "  W/S  forward/backward\n"
        << "  A/D  strafe left/right\n"
        << "  Q/E  turn left/right\n"
        << "  R    start capture flow for the selected mode\n"
        << "  T    manually end the current capture if already recording\n"
        << "  ESC  cancel current armed/capture segment\n"
        << "  Space emergency stop and latch safety fault\n"
        << "  C    clear fault or toggle stand up/down\n"
        << "  P    print status\n"
        << "  H    print help\n"
        << "  X    quit\n"
        << "Input:\n"
        << "  evdev supports true multi-key press/release and smoother diagonal motion\n"
        << "  tty is a fallback mode and may feel less stable for combined keys\n"
        << "Capture modes:\n"
        << "  single_action arms on R, locks one motion key, waits 0.5s, records until key release\n"
        << "  trajectory starts recording immediately on R, allows turning/strafe changes, and ends on T\n";
}

std::optional<Config> ParseArgs(int argc, char** argv, std::string& error)
{
    Config config;
    const fs::path collectorRoot = CollectorRootFromArgv0(argv[0]);
    config.outputDir = collectorRoot / kDefaultDataDirName;

    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        if (arg == "--network-interface")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --network-interface";
                return std::nullopt;
            }
            config.networkInterface = argv[++index];
        }
        else if (arg == "--output-dir")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --output-dir";
                return std::nullopt;
            }
            fs::path outputDir = argv[++index];
            if (!outputDir.is_absolute())
            {
                outputDir = collectorRoot / outputDir;
            }
            config.outputDir = outputDir;
        }
        else if (arg == "--loop-hz")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --loop-hz";
                return std::nullopt;
            }
            config.loopHz = std::stod(argv[++index]);
        }
        else if (arg == "--video-poll-hz")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --video-poll-hz";
                return std::nullopt;
            }
            config.videoPollHz = std::stod(argv[++index]);
        }
        else if (arg == "--input-backend")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --input-backend";
                return std::nullopt;
            }
            const auto backend = ParseInputBackend(argv[++index]);
            if (!backend.has_value())
            {
                error = "input backend must be one of: evdev, tty";
                return std::nullopt;
            }
            config.inputBackend = backend.value();
        }
        else if (arg == "--capture-mode")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --capture-mode";
                return std::nullopt;
            }
            const auto captureMode = ParseCaptureMode(argv[++index]);
            if (!captureMode.has_value())
            {
                error = "capture mode must be one of: single_action, trajectory";
                return std::nullopt;
            }
            config.captureMode = captureMode.value();
        }
        else if (arg == "--input-device")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --input-device";
                return std::nullopt;
            }
            config.inputDevice = argv[++index];
        }
        else if (arg == "--scene-id")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --scene-id";
                return std::nullopt;
            }
            config.sceneId = argv[++index];
        }
        else if (arg == "--operator-id")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --operator-id";
                return std::nullopt;
            }
            config.operatorId = argv[++index];
        }
        else if (arg == "--instruction")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --instruction";
                return std::nullopt;
            }
            config.instruction = argv[++index];
        }
        else if (arg == "--task-family")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --task-family";
                return std::nullopt;
            }
            config.taskFamily = argv[++index];
        }
        else if (arg == "--target-type")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --target-type";
                return std::nullopt;
            }
            config.targetType = argv[++index];
        }
        else if (arg == "--target-description")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --target-description";
                return std::nullopt;
            }
            config.targetDescription = argv[++index];
        }
        else if (arg == "--target-instance-id")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --target-instance-id";
                return std::nullopt;
            }
            config.targetInstanceId = argv[++index];
        }
        else if (arg == "--task-tags")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --task-tags";
                return std::nullopt;
            }
            config.taskTags = SplitCsvList(argv[++index]);
        }
        else if (arg == "--collector-notes")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --collector-notes";
                return std::nullopt;
            }
            config.collectorNotes = argv[++index];
        }
        else if (arg == "--cmd-vx-max")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --cmd-vx-max";
                return std::nullopt;
            }
            config.cmdVxMax = std::stod(argv[++index]);
        }
        else if (arg == "--cmd-vy-max")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --cmd-vy-max";
                return std::nullopt;
            }
            config.cmdVyMax = std::stod(argv[++index]);
        }
        else if (arg == "--cmd-wz-max")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --cmd-wz-max";
                return std::nullopt;
            }
            config.cmdWzMax = std::stod(argv[++index]);
        }
        else if (arg == "--help")
        {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        else
        {
            error = "unknown argument: " + arg;
            return std::nullopt;
        }
    }

    config.sceneId = Trim(config.sceneId);
    config.operatorId = Trim(config.operatorId);
    config.instruction = Trim(config.instruction);
    config.taskFamily = Trim(config.taskFamily);
    config.targetType = Trim(config.targetType);
    config.targetDescription = Trim(config.targetDescription);
    config.targetInstanceId = Trim(config.targetInstanceId);
    config.collectorNotes = Trim(config.collectorNotes);

    if (config.networkInterface.empty())
    {
        error = "--network-interface is required";
        return std::nullopt;
    }
    if (config.sceneId.empty())
    {
        error = "--scene-id is required";
        return std::nullopt;
    }
    if (config.operatorId.empty())
    {
        error = "--operator-id is required";
        return std::nullopt;
    }
    if (config.captureMode == Config::CaptureMode::Trajectory && config.instruction.empty())
    {
        error = "--instruction is required when --capture-mode trajectory is used";
        return std::nullopt;
    }
    if (config.loopHz <= 0.0 || config.videoPollHz <= 0.0)
    {
        error = "frequencies must be positive";
        return std::nullopt;
    }
    if (config.cmdVxMax <= 0.0 || config.cmdVyMax <= 0.0 || config.cmdWzMax <= 0.0)
    {
        error = "command limits must be positive";
        return std::nullopt;
    }
    if (config.inputBackend == Config::InputBackend::Evdev && config.inputDevice.empty())
    {
        const auto detected = FindDefaultKeyboardDevice();
        if (!detected.has_value())
        {
            error = "failed to auto-detect keyboard input device under /dev/input";
            return std::nullopt;
        }
        config.inputDevice = detected.value();
    }

    return config;
}

class RawTerminalGuard
{
public:
    RawTerminalGuard() = default;

    bool TryEnable()
    {
        if (!isatty(STDIN_FILENO))
        {
            return false;
        }
        Enable();
        return true;
    }

    void Enable()
    {
        if (!isatty(STDIN_FILENO))
        {
            throw std::runtime_error("stdin must be a tty for keyboard teleop");
        }
        if (enabled_)
        {
            return;
        }
        if (tcgetattr(STDIN_FILENO, &original_) != 0)
        {
            throw std::runtime_error("failed to get terminal attributes");
        }
        termios raw = original_;
        raw.c_lflag &= static_cast<unsigned int>(~(ICANON | ECHO));
        raw.c_iflag &= static_cast<unsigned int>(~(IXON | ICRNL));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 1;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0)
        {
            throw std::runtime_error("failed to set raw terminal mode");
        }
        enabled_ = true;
    }

    void Disable()
    {
        if (!enabled_)
        {
            return;
        }
        tcsetattr(STDIN_FILENO, TCSANOW, &original_);
        enabled_ = false;
    }

    ~RawTerminalGuard()
    {
        Disable();
    }

private:
    bool enabled_ = false;
    termios original_{};
};

class TrajectoryLogger
{
public:
    explicit TrajectoryLogger(const fs::path& outputDir)
        : sessionId_(FormatTimestampForPath(NowSeconds(), "%Y%m%d_%H%M%S")),
          outputDir_(outputDir / sessionId_),
          episodesDir_(outputDir_ / "episodes"),
          imagesDir_(outputDir_ / "images"),
          indexPath_(outputDir_ / "index.json")
    {
        fs::create_directories(outputDir_);
        fs::create_directories(episodesDir_);
        fs::create_directories(imagesDir_);
        RewriteIndexLocked();
    }

    const fs::path& OutputDir() const
    {
        return outputDir_;
    }

    bool HasRecordedEpisodes() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return !episodeHistory_.empty();
    }

    void CleanupIfEmpty()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!episodeHistory_.empty() || capturing_ || !pendingFrames_.empty())
        {
            return;
        }
        std::error_code ec;
        fs::remove_all(outputDir_, ec);
    }

    struct Status
    {
        bool capturing = false;
        bool pendingLabel = false;
        size_t bufferedFrames = 0;
        std::string pendingEpisodeId;
        std::string lastEpisodeId;
    };

    Status GetStatus() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Status status;
        status.capturing = capturing_;
        status.pendingLabel = pendingLabel_;
        status.bufferedFrames = pendingFrames_.size();
        status.pendingEpisodeId = pendingEpisodeId_;
        status.lastEpisodeId = lastRecordedEpisodeId_;
        return status;
    }

    void BeginSegment(const std::string& sceneId, const std::string& operatorId, const TaskMetadata& taskMetadata)
    {
        const std::string trimmedSceneId = Trim(sceneId);
        const std::string trimmedOperatorId = Trim(operatorId);
        if (trimmedSceneId.empty())
        {
            throw std::runtime_error("scene_id must not be empty");
        }
        if (trimmedOperatorId.empty())
        {
            throw std::runtime_error("operator_id must not be empty");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        pendingFrames_.clear();
        pendingSceneId_ = trimmedSceneId;
        pendingOperatorId_ = trimmedOperatorId;
        pendingTaskMetadata_ = taskMetadata;
        pendingEpisodeId_.clear();
        capturing_ = true;
        pendingLabel_ = false;
    }

    bool LogStep(
        double timestamp,
        const LatestState& state,
        const VelocityCommand& rawAction,
        const EffectiveControlAction& controlAction,
        const LatestImage& image,
        double stateTimestamp,
        double rawActionTimestamp,
        double imageTimestamp)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!capturing_)
        {
            return false;
        }
        if (!image.valid || image.jpegBytes.empty())
        {
            return false;
        }

        pendingFrames_.push_back(EpisodeFrame{
            timestamp,
            stateTimestamp,
            controlAction.timestamp,
            rawActionTimestamp,
            imageTimestamp,
            state.velocityX,
            state.velocityY,
            state.yawSpeed,
            state.yaw,
            rawAction.vx,
            rawAction.vy,
            rawAction.wz,
            controlAction.command.vx,
            controlAction.command.vy,
            controlAction.command.wz,
            image.jpegBytes,
        });
        return true;
    }

    size_t EndSegmentForLabel()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!capturing_)
        {
            return 0;
        }
        capturing_ = false;
        pendingLabel_ = !pendingFrames_.empty();
        return pendingFrames_.size();
    }

    void DiscardPendingSegment()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        capturing_ = false;
        pendingLabel_ = false;
        pendingFrames_.clear();
        pendingEpisodeId_.clear();
        pendingSceneId_.clear();
        pendingOperatorId_.clear();
        pendingTaskMetadata_ = TaskMetadata{};
    }

    std::optional<std::string> FinalizePendingSegment(const std::string& fallbackInstruction)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pendingLabel_ || pendingFrames_.empty())
        {
            return std::nullopt;
        }
        const TaskMetadata resolvedTaskMetadata = ResolveTaskMetadata(pendingTaskMetadata_, fallbackInstruction);
        const std::string trimmedInstruction = Trim(resolvedTaskMetadata.instruction);
        if (trimmedInstruction.empty())
        {
            throw std::runtime_error("instruction must not be empty");
        }

        std::ostringstream episodeId;
        episodeId << "ep_" << std::setw(6) << std::setfill('0') << nextEpisodeIndex_++;
        pendingEpisodeId_ = episodeId.str();

        const fs::path episodeImageDir = imagesDir_ / pendingEpisodeId_;
        fs::create_directories(episodeImageDir);

        std::ostringstream episodeJson;
        episodeJson << std::fixed << std::setprecision(6);
        episodeJson << "{\n"
                    << "  \"schema_version\": " << JsonString(kSchemaVersion) << ",\n"
                    << "  \"episode_id\": " << JsonString(pendingEpisodeId_) << ",\n"
                    << "  \"instruction\": " << JsonString(trimmedInstruction) << ",\n"
                    << "  \"capture_mode\": " << JsonString(resolvedTaskMetadata.captureMode) << ",\n"
                    << "  \"task_family\": " << JsonString(resolvedTaskMetadata.taskFamily) << ",\n"
                    << "  \"target_type\": " << JsonString(resolvedTaskMetadata.targetType) << ",\n"
                    << "  \"target_description\": " << JsonString(resolvedTaskMetadata.targetDescription) << ",\n"
                    << "  \"target_instance_id\": " << JsonString(resolvedTaskMetadata.targetInstanceId) << ",\n"
                    << "  \"task_tags\": " << JsonStringArray(resolvedTaskMetadata.taskTags) << ",\n"
                    << "  \"collector_notes\": " << JsonString(resolvedTaskMetadata.collectorNotes) << ",\n"
                    << "  \"instruction_source\": " << JsonString(resolvedTaskMetadata.instructionSource) << ",\n"
                    << "  \"scene_id\": " << JsonString(pendingSceneId_) << ",\n"
                    << "  \"operator_id\": " << JsonString(pendingOperatorId_) << ",\n"
                    << "  \"frames\": [\n";

        for (size_t index = 0; index < pendingFrames_.size(); ++index)
        {
            const auto& frame = pendingFrames_[index];
            std::ostringstream filename;
            filename << FormatTimestampForPath(frame.timestamp, "%Y%m%d_%H%M%S")
                     << "_" << std::setw(3) << std::setfill('0') << (index + 1) << ".jpg";
            const fs::path imagePath = episodeImageDir / filename.str();
            std::ofstream imageFile(imagePath, std::ios::binary);
            if (!imageFile.is_open())
            {
                throw std::runtime_error("failed to write image file");
            }
            imageFile.write(reinterpret_cast<const char*>(frame.jpegBytes.data()), static_cast<std::streamsize>(frame.jpegBytes.size()));
            const std::string imageRelPath = fs::relative(imagePath, outputDir_).string();

            episodeJson << "    {\n"
                        << "      \"timestamp\": " << frame.timestamp << ",\n"
                        << "      \"image\": " << JsonString(imageRelPath) << ",\n"
                        << "      \"instruction\": " << JsonString(trimmedInstruction) << ",\n"
                        << "      \"state\": {\n"
                        << "        \"vx\": " << frame.stateVx << ",\n"
                        << "        \"vy\": " << frame.stateVy << ",\n"
                        << "        \"wz\": " << frame.stateWz << ",\n"
                        << "        \"yaw\": " << frame.stateYaw << "\n"
                        << "      },\n"
                        << "      \"raw_action\": {\n"
                        << "        \"vx\": " << frame.rawActionVx << ",\n"
                        << "        \"vy\": " << frame.rawActionVy << ",\n"
                        << "        \"wz\": " << frame.rawActionWz << ",\n"
                        << "        \"camera_pitch\": 0.0,\n"
                        << "        \"keys\": 0\n"
                        << "      },\n"
                        << "      \"control_action\": {\n"
                        << "        \"vx\": " << frame.controlActionVx << ",\n"
                        << "        \"vy\": " << frame.controlActionVy << ",\n"
                        << "        \"wz\": " << frame.controlActionWz << "\n"
                        << "      },\n"
                        << "      \"meta\": {\n"
                        << "        \"schema_version\": " << JsonString(kSchemaVersion) << ",\n"
                        << "        \"episode_id\": " << JsonString(pendingEpisodeId_) << ",\n"
                        << "        \"capture_mode\": " << JsonString(resolvedTaskMetadata.captureMode) << ",\n"
                        << "        \"task_family\": " << JsonString(resolvedTaskMetadata.taskFamily) << ",\n"
                        << "        \"target_type\": " << JsonString(resolvedTaskMetadata.targetType) << ",\n"
                        << "        \"target_description\": " << JsonString(resolvedTaskMetadata.targetDescription) << ",\n"
                        << "        \"target_instance_id\": " << JsonString(resolvedTaskMetadata.targetInstanceId) << ",\n"
                        << "        \"instruction_source\": " << JsonString(resolvedTaskMetadata.instructionSource) << ",\n"
                        << "        \"scene_id\": " << JsonString(pendingSceneId_) << ",\n"
                        << "        \"operator_id\": " << JsonString(pendingOperatorId_) << ",\n"
                        << "        \"state_timestamp\": " << frame.stateTimestamp << ",\n"
                        << "        \"action_timestamp\": " << frame.actionTimestamp << ",\n"
                        << "        \"raw_action_timestamp\": " << frame.rawActionTimestamp << ",\n"
                        << "        \"control_action_timestamp\": " << frame.actionTimestamp << ",\n"
                        << "        \"image_timestamp\": " << frame.imageTimestamp << "\n"
                        << "      }\n"
                        << "    }";
            if (index + 1 != pendingFrames_.size())
            {
                episodeJson << ",";
            }
            episodeJson << "\n";
        }
        episodeJson << "  ]\n"
                    << "}\n";

        std::ofstream episodeFile(episodesDir_ / (pendingEpisodeId_ + ".json"), std::ios::out | std::ios::trunc);
        if (!episodeFile.is_open())
        {
            throw std::runtime_error("failed to write episode file");
        }
        episodeFile << episodeJson.str();
        episodeFile.flush();

        EpisodeSummary summary;
        summary.episodeId = pendingEpisodeId_;
        summary.instruction = trimmedInstruction;
        summary.captureMode = resolvedTaskMetadata.captureMode;
        summary.taskFamily = resolvedTaskMetadata.taskFamily;
        summary.targetType = resolvedTaskMetadata.targetType;
        summary.targetDescription = resolvedTaskMetadata.targetDescription;
        summary.targetInstanceId = resolvedTaskMetadata.targetInstanceId;
        summary.taskTags = resolvedTaskMetadata.taskTags;
        summary.collectorNotes = resolvedTaskMetadata.collectorNotes;
        summary.instructionSource = resolvedTaskMetadata.instructionSource;
        summary.sceneId = pendingSceneId_;
        summary.operatorId = pendingOperatorId_;
        summary.numFrames = pendingFrames_.size();
        summary.startTimestamp = pendingFrames_.front().timestamp;
        summary.endTimestamp = pendingFrames_.back().timestamp;
        episodeHistory_[summary.episodeId] = summary;
        lastRecordedEpisodeId_ = summary.episodeId;
        RewriteIndexLocked();

        pendingFrames_.clear();
        pendingLabel_ = false;
        pendingSceneId_.clear();
        pendingOperatorId_.clear();
        pendingTaskMetadata_ = TaskMetadata{};

        return summary.episodeId;
    }

private:
    void RewriteIndexLocked() const
    {
        std::vector<EpisodeSummary> summaries;
        summaries.reserve(episodeHistory_.size());
        for (const auto& item : episodeHistory_)
        {
            summaries.push_back(item.second);
        }
        std::sort(
            summaries.begin(),
            summaries.end(),
            [](const EpisodeSummary& lhs, const EpisodeSummary& rhs)
            {
                return lhs.episodeId < rhs.episodeId;
            });

        std::ofstream indexFile(indexPath_, std::ios::out | std::ios::trunc);
        if (!indexFile.is_open())
        {
            throw std::runtime_error("failed to rewrite index.json");
        }

        indexFile << std::fixed << std::setprecision(6);
        indexFile << "{\n"
                  << "  \"schema_version\": " << JsonString(kSchemaVersion) << ",\n"
                  << "  \"session_id\": " << JsonString(sessionId_) << ",\n"
                  << "  \"episodes\": [\n";
        for (size_t index = 0; index < summaries.size(); ++index)
        {
            const auto& summary = summaries[index];
            indexFile << "    {\n"
                      << "      \"episode_id\": " << JsonString(summary.episodeId) << ",\n"
                      << "      \"instruction\": " << JsonString(summary.instruction) << ",\n"
                      << "      \"capture_mode\": " << JsonString(summary.captureMode) << ",\n"
                      << "      \"task_family\": " << JsonString(summary.taskFamily) << ",\n"
                      << "      \"target_type\": " << JsonString(summary.targetType) << ",\n"
                      << "      \"target_description\": " << JsonString(summary.targetDescription) << ",\n"
                      << "      \"target_instance_id\": " << JsonString(summary.targetInstanceId) << ",\n"
                      << "      \"task_tags\": " << JsonStringArray(summary.taskTags) << ",\n"
                      << "      \"collector_notes\": " << JsonString(summary.collectorNotes) << ",\n"
                      << "      \"instruction_source\": " << JsonString(summary.instructionSource) << ",\n"
                      << "      \"scene_id\": " << JsonString(summary.sceneId) << ",\n"
                      << "      \"operator_id\": " << JsonString(summary.operatorId) << ",\n"
                      << "      \"num_frames\": " << summary.numFrames << ",\n"
                      << "      \"start_timestamp\": " << summary.startTimestamp << ",\n"
                      << "      \"end_timestamp\": " << summary.endTimestamp << "\n"
                      << "    }";
            if (index + 1 != summaries.size())
            {
                indexFile << ",";
            }
            indexFile << "\n";
        }
        indexFile << "  ]\n"
                  << "}\n";
    }

    std::string sessionId_;
    fs::path outputDir_;
    fs::path episodesDir_;
    fs::path imagesDir_;
    fs::path indexPath_;
    mutable std::mutex mutex_;
    std::vector<EpisodeFrame> pendingFrames_;
    std::map<std::string, EpisodeSummary> episodeHistory_;
    int nextEpisodeIndex_ = 1;
    bool capturing_ = false;
    bool pendingLabel_ = false;
    std::string pendingSceneId_;
    std::string pendingOperatorId_;
    TaskMetadata pendingTaskMetadata_;
    std::string pendingEpisodeId_;
    std::string lastRecordedEpisodeId_;
};

class CollectorApp
{
public:
    explicit CollectorApp(const Config& config)
        : config_(config), logger_(config.outputDir)
    {
    }

    ~CollectorApp()
    {
        Shutdown();
    }

    bool ShouldQuit() const
    {
        return quitRequested_.load();
    }

    void RequestQuit()
    {
        quitRequested_.store(true);
    }

    void Start()
    {
        if (running_.load())
        {
            return;
        }

        if (config_.inputBackend == Config::InputBackend::Tty)
        {
            terminalGuard_.Enable();
        }
        else
        {
            terminalRawEnabled_ = terminalGuard_.TryEnable();
            OpenEvdevInput();
        }

        unitree::robot::ChannelFactory::Instance()->Init(0, config_.networkInterface);

        sportClient_ = std::make_unique<unitree::robot::go2::SportClient>();
        sportClient_->SetTimeout(10.0f);
        sportClient_->Init();

        sportStateSubscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(kSportStateTopic);
        sportStateSubscriber_->InitChannel(std::bind(&CollectorApp::OnSportState, this, std::placeholders::_1), 1);

        videoClient_ = std::make_unique<unitree::robot::go2::VideoClient>();
        videoClient_->SetTimeout(1.0f);
        videoClient_->Init();

        running_.store(true);
        controlThread_ = std::thread(&CollectorApp::ControlLoop, this);
        loggingThread_ = std::thread(&CollectorApp::LoggingLoop, this);
        keyboardThread_ = std::thread(&CollectorApp::KeyboardLoop, this);
        videoThread_ = std::thread(&CollectorApp::VideoLoop, this);

        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            latestCommand_.valid = true;
            latestCommand_.timestamp = NowSeconds();
        }

        PrintLine("session directory: " + logger_.OutputDir().string());
        PrintLine("input backend: " + InputBackendName(config_.inputBackend) +
                  (config_.inputDevice.empty() ? "" : " device=" + config_.inputDevice.string()));
        PrintLine("collector ready; press H for help");
        PrintHelp();
    }

    void Shutdown()
    {
        if (!running_.exchange(false))
        {
            return;
        }

        imageUpdatedCv_.notify_all();

        if (sportStateSubscriber_)
        {
            sportStateSubscriber_->CloseChannel();
            sportStateSubscriber_.reset();
        }

        if (keyboardThread_.joinable())
        {
            keyboardThread_.join();
        }
        if (controlThread_.joinable())
        {
            controlThread_.join();
        }
        if (loggingThread_.joinable())
        {
            loggingThread_.join();
        }
        if (videoThread_.joinable())
        {
            videoThread_.join();
        }

        StopMotion();
        unitree::robot::ChannelFactory::Instance()->Release();
        CloseEvdevInput();
        terminalGuard_.Disable();
        logger_.CleanupIfEmpty();
    }

private:
    struct Snapshot
    {
        LatestState state;
        LatestImage image;
        VelocityCommand action;
    };

    struct KeyActivity
    {
        bool pressed = false;
        std::chrono::steady_clock::time_point deadline{};
    };

    struct MotionStateSnapshot
    {
        bool forward = false;
        bool backward = false;
        bool left = false;
        bool right = false;
        bool yawLeft = false;
        bool yawRight = false;
    };

    MotionStateSnapshot GetMotionStateSnapshotLocked(std::chrono::steady_clock::time_point now) const
    {
        MotionStateSnapshot motionState;
        motionState.forward = keyW_.pressed || keyW_.deadline > now;
        motionState.backward = keyS_.pressed || keyS_.deadline > now;
        motionState.left = keyA_.pressed || keyA_.deadline > now;
        motionState.right = keyD_.pressed || keyD_.deadline > now;
        motionState.yawLeft = keyQ_.pressed || keyQ_.deadline > now;
        motionState.yawRight = keyE_.pressed || keyE_.deadline > now;
        return motionState;
    }

    MotionStateSnapshot GetMotionStateSnapshot() const
    {
        std::lock_guard<std::mutex> lock(keyMutex_);
        return GetMotionStateSnapshotLocked(std::chrono::steady_clock::now());
    }

    static MotionInstruction MotionInstructionFromSnapshot(const MotionStateSnapshot& motionState)
    {
        const int activeCount = static_cast<int>(motionState.forward) +
                                static_cast<int>(motionState.backward) +
                                static_cast<int>(motionState.left) +
                                static_cast<int>(motionState.right) +
                                static_cast<int>(motionState.yawLeft) +
                                static_cast<int>(motionState.yawRight);
        if (activeCount == 0)
        {
            return MotionInstruction::None;
        }
        if (activeCount != 1)
        {
            return MotionInstruction::Mixed;
        }
        if (motionState.forward)
        {
            return MotionInstruction::GoForward;
        }
        if (motionState.backward)
        {
            return MotionInstruction::MoveBackward;
        }
        if (motionState.left)
        {
            return MotionInstruction::StrafeLeft;
        }
        if (motionState.right)
        {
            return MotionInstruction::StrafeRight;
        }
        if (motionState.yawLeft)
        {
            return MotionInstruction::TurnLeft;
        }
        return MotionInstruction::TurnRight;
    }

    void PrintLine(const std::string& line) const
    {
        std::lock_guard<std::mutex> lock(outputMutex_);
        std::cout << "\r\33[2K" << line << std::endl;
    }

    void PrintHelp() const
    {
        std::lock_guard<std::mutex> lock(outputMutex_);
        std::cout << "\r\33[2KKeys:" << std::endl;
        std::cout << "  W/S forward/backward  A/D strafe  Q/E yaw" << std::endl;
        std::cout << "  R start capture flow  ESC cancel current armed/capture segment" << std::endl;
        std::cout << "  Space emergency stop  C clear fault or toggle stand up/down  P status  H help  X quit" << std::endl;
        if (config_.captureMode == Config::CaptureMode::SingleAction)
        {
            std::cout << "  single_action mode: capture starts 0.5s after first valid motion input and auto-stops on key release" << std::endl;
            std::cout << "  semantic instruction is taken from --instruction if set, otherwise falls back to the motion label" << std::endl;
        }
        else
        {
            std::cout << "  trajectory mode: capture starts immediately on R, allows multi-stage motion changes, and stops on T" << std::endl;
            std::cout << "  --instruction is required and is stored as the trajectory-level semantic label" << std::endl;
        }
        std::cout << "  input_backend=" << InputBackendName(config_.inputBackend)
                  << " capture_mode=" << CaptureModeName(config_.captureMode) << std::endl;
    }

    void PrintStatus() const
    {
        LatestState state;
        LatestImage image;
        VelocityCommand command;
        MotionStateSnapshot motionState;
        {
            std::lock_guard<std::mutex> lock(stateMutex_);
            state = latestState_;
        }
        {
            std::lock_guard<std::mutex> lock(imageMutex_);
            image = latestImage_;
        }
        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            command = latestCommand_;
        }
        {
            std::lock_guard<std::mutex> lock(keyMutex_);
            const auto now = std::chrono::steady_clock::now();
            motionState.forward = keyW_.pressed || keyW_.deadline > now;
            motionState.backward = keyS_.pressed || keyS_.deadline > now;
            motionState.left = keyA_.pressed || keyA_.deadline > now;
            motionState.right = keyD_.pressed || keyD_.deadline > now;
            motionState.yawLeft = keyQ_.pressed || keyQ_.deadline > now;
            motionState.yawRight = keyE_.pressed || keyE_.deadline > now;
        }

        const auto loggerStatus = logger_.GetStatus();
        const double nowSeconds = NowSeconds();
        SafetyState safetyState;
        std::string faultReason;
        CaptureState captureState;
        std::string activeInstruction;
        double armDelayRemaining = 0.0;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            safetyState = safetyState_;
            faultReason = latchedFaultReason_;
            captureState = captureState_;
            activeInstruction = MotionInstructionName(activeMotionInstruction_);
            if (captureState_ == CaptureState::DelayBeforeLog)
            {
                armDelayRemaining = std::max(
                    0.0,
                    std::chrono::duration<double>(captureStartDeadline_ - std::chrono::steady_clock::now()).count());
            }
        }

        std::ostringstream oss;
        oss << std::boolalpha
            << "capture_state=" << CaptureStateName(captureState)
            << " safety_state=" << SafetyStateName(safetyState)
            << " fault_reason=" << (faultReason.empty() ? "-" : faultReason)
            << " scene_id=" << config_.sceneId
            << " operator_id=" << config_.operatorId
            << " capture_mode=" << CaptureModeName(config_.captureMode)
            << " configured_instruction=" << (config_.instruction.empty() ? "-" : config_.instruction)
            << " task_family=" << (config_.taskFamily.empty() ? "-" : config_.taskFamily)
            << " target_type=" << (config_.targetType.empty() ? "-" : config_.targetType)
            << " target_description=" << (config_.targetDescription.empty() ? "-" : config_.targetDescription)
            << " buffered_frames=" << loggerStatus.bufferedFrames
            << " state=" << state.valid
            << " image=" << image.valid
            << " state_age_s=" << std::fixed << std::setprecision(3) << AgeSeconds(state.timestamp, nowSeconds)
            << " image_age_s=" << AgeSeconds(image.timestamp, nowSeconds)
            << " command=(" << command.vx << "," << command.vy << "," << command.wz << ")"
            << " motion_keys=("
            << (motionState.forward ? "W" : "")
            << (motionState.backward ? "S" : "")
            << (motionState.left ? "A" : "")
            << (motionState.right ? "D" : "")
            << (motionState.yawLeft ? "Q" : "")
            << (motionState.yawRight ? "E" : "")
            << ")"
            << " active_instruction=" << (activeInstruction.empty() ? "-" : activeInstruction)
            << " capture_delay_s=" << std::fixed << std::setprecision(3) << armDelayRemaining;
        PrintLine(oss.str());
    }

    void OpenEvdevInput()
    {
        if (config_.inputDevice.empty())
        {
            throw std::runtime_error("evdev input device path is empty");
        }
        evdevFd_ = ::open(config_.inputDevice.c_str(), O_RDONLY | O_NONBLOCK);
        if (evdevFd_ < 0)
        {
            throw std::runtime_error("failed to open input device: " + config_.inputDevice.string());
        }
    }

    void CloseEvdevInput()
    {
        if (evdevFd_ >= 0)
        {
            ::close(evdevFd_);
            evdevFd_ = -1;
        }
    }

    std::string EvaluateSafetyFault(const Snapshot& snapshot, double nowSeconds) const
    {
        if (!snapshot.state.valid)
        {
            return "state_unavailable";
        }
        const double stateAge = AgeSeconds(snapshot.state.timestamp, nowSeconds);
        if (stateAge < 0.0 || stateAge > kStateTimeoutSeconds)
        {
            return "state_timeout";
        }
        if (std::fabs(snapshot.state.roll) > kMaxSafeAbsRollRad)
        {
            return "roll_limit_exceeded";
        }
        if (std::fabs(snapshot.state.pitch) > kMaxSafeAbsPitchRad)
        {
            return "pitch_limit_exceeded";
        }
        return "";
    }

    bool CanClearFault(std::string& reason) const
    {
        Snapshot snapshot;
        {
            std::lock_guard<std::mutex> lock(stateMutex_);
            snapshot.state = latestState_;
        }
        {
            std::lock_guard<std::mutex> lock(imageMutex_);
            snapshot.image = latestImage_;
        }
        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            snapshot.action = latestCommand_;
        }
        reason = EvaluateSafetyFault(snapshot, NowSeconds());
        return reason.empty();
    }

    void ClearLatchedFault()
    {
        std::lock_guard<std::mutex> lock(stateMachineMutex_);
        safetyState_ = SafetyState::SafeReady;
        latchedFaultReason_.clear();
        if (captureState_ == CaptureState::Fault)
        {
            captureState_ = CaptureState::Idle;
        }
    }

    void LatchFault(SafetyState state, const std::string& reason)
    {
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            safetyState_ = state;
            latchedFaultReason_ = reason;
            captureState_ = CaptureState::Fault;
            activeMotionInstruction_ = MotionInstruction::None;
            captureStartDeadline_ = std::chrono::steady_clock::time_point{};
        }
        logger_.DiscardPendingSegment();
        ResetActiveMotionKeys();
        StopMotion();
    }

    void RequestEmergencyStop(const std::string& reason)
    {
        LatchFault(SafetyState::EstopLatched, reason);
        PrintLine("emergency stop latched: " + reason);
    }

    void ClearFaultOrToggleStandState()
    {
        bool hasFault = false;
        bool segmentActive = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            hasFault = safetyState_ != SafetyState::SafeReady;
            segmentActive = captureState_ == CaptureState::Armed ||
                            captureState_ == CaptureState::DelayBeforeLog ||
                            captureState_ == CaptureState::Capturing;
        }

        if (hasFault)
        {
            std::string reason;
            if (CanClearFault(reason))
            {
                ClearLatchedFault();
                PrintLine("safety fault cleared");
            }
            else
            {
                PrintLine("cannot clear safety fault: " + reason);
            }
            return;
        }

        if (segmentActive)
        {
            PrintLine("finish or discard the active segment first");
            return;
        }

        LatestState state;
        {
            std::lock_guard<std::mutex> stateLock(stateMutex_);
            state = latestState_;
        }
        const bool shouldStandDown = state.valid && state.bodyHeight >= kStandingBodyHeightThreshold;

        try
        {
            std::lock_guard<std::mutex> sportLock(sportClientMutex_);
            if (!sportClient_)
            {
                PrintLine("sport client is not ready");
                return;
            }
            sportClient_->StopMove();
            if (shouldStandDown)
            {
                sportClient_->StandDown();
                PrintLine("requested stand down");
            }
            else
            {
                sportClient_->StandUp();
                PrintLine("requested stand up");
            }
        }
        catch (...)
        {
            PrintLine("failed to toggle stand state");
        }
    }

    TaskMetadata ConfiguredTaskMetadata() const
    {
        TaskMetadata metadata;
        metadata.instruction = config_.instruction;
        metadata.captureMode = CaptureModeName(config_.captureMode);
        metadata.taskFamily = config_.taskFamily;
        metadata.targetType = config_.targetType;
        metadata.targetDescription = config_.targetDescription;
        metadata.targetInstanceId = config_.targetInstanceId;
        metadata.taskTags = config_.taskTags;
        metadata.collectorNotes = config_.collectorNotes;
        metadata.instructionSource = metadata.instruction.empty() ? "motion_label" : "semantic_text";
        return metadata;
    }

    void BeginSegment()
    {
        std::lock_guard<std::mutex> lock(stateMachineMutex_);
        if (safetyState_ != SafetyState::SafeReady)
        {
            PrintLine("cannot start segment while safety fault is latched");
            return;
        }
        if (captureState_ == CaptureState::Capturing)
        {
            PrintLine("segment already active");
            return;
        }
        if (captureState_ == CaptureState::Armed || captureState_ == CaptureState::DelayBeforeLog)
        {
            PrintLine("segment already armed");
            return;
        }
        try
        {
            if (config_.captureMode == Config::CaptureMode::Trajectory)
            {
                logger_.BeginSegment(config_.sceneId, config_.operatorId, ConfiguredTaskMetadata());
                captureState_ = CaptureState::Capturing;
                activeMotionInstruction_ = MotionInstruction::None;
                captureStartDeadline_ = std::chrono::steady_clock::time_point{};
                PrintLine("trajectory capture started; press T to save or ESC to discard");
            }
            else
            {
                captureState_ = CaptureState::Armed;
                activeMotionInstruction_ = MotionInstruction::None;
                captureStartDeadline_ = std::chrono::steady_clock::time_point{};
                PrintLine("segment armed; waiting for a single valid motion input");
            }
        }
        catch (const std::exception& ex)
        {
            PrintLine(std::string("failed to start segment: ") + ex.what());
        }
    }

    void EndSegment()
    {
        MotionInstruction instruction = MotionInstruction::None;
        std::string savedInstruction;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (captureState_ != CaptureState::Capturing)
            {
                PrintLine("no active segment to end");
                return;
            }
            instruction = activeMotionInstruction_;
            savedInstruction = config_.instruction.empty() ? MotionInstructionName(instruction) : config_.instruction;
        }

        const size_t frameCount = logger_.EndSegmentForLabel();
        if (frameCount == 0)
        {
            logger_.DiscardPendingSegment();
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState_ = SafetyState::SafeReady == safetyState_ ? CaptureState::Idle : CaptureState::Fault;
            activeMotionInstruction_ = MotionInstruction::None;
            captureStartDeadline_ = std::chrono::steady_clock::time_point{};
            PrintLine("segment discarded because it contains no frames");
            return;
        }

        try
        {
            const auto episodeId = logger_.FinalizePendingSegment(MotionInstructionName(instruction));
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState_ = SafetyState::SafeReady == safetyState_ ? CaptureState::Idle : CaptureState::Fault;
            activeMotionInstruction_ = MotionInstruction::None;
            captureStartDeadline_ = std::chrono::steady_clock::time_point{};
            if (episodeId.has_value())
            {
                PrintLine("saved episode " + episodeId.value() + " instruction=" + savedInstruction);
            }
            else
            {
                PrintLine("no pending segment to save");
            }
        }
        catch (const std::exception& ex)
        {
            logger_.DiscardPendingSegment();
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState_ = SafetyState::SafeReady == safetyState_ ? CaptureState::Idle : CaptureState::Fault;
            activeMotionInstruction_ = MotionInstruction::None;
            captureStartDeadline_ = std::chrono::steady_clock::time_point{};
            PrintLine(std::string("failed to finalize segment: ") + ex.what());
        }
    }

    void CancelCurrentSegment(const std::string& reason)
    {
        std::lock_guard<std::mutex> lock(stateMachineMutex_);
        if (captureState_ != CaptureState::Capturing &&
            captureState_ != CaptureState::Armed &&
            captureState_ != CaptureState::DelayBeforeLog)
        {
            return;
        }
        logger_.DiscardPendingSegment();
        captureState_ = SafetyState::SafeReady == safetyState_ ? CaptureState::Idle : CaptureState::Fault;
        activeMotionInstruction_ = MotionInstruction::None;
        captureStartDeadline_ = std::chrono::steady_clock::time_point{};
        PrintLine("segment discarded: " + reason);
    }

    void StopMotion()
    {
        try
        {
            std::lock_guard<std::mutex> lock(sportClientMutex_);
            if (sportClient_)
            {
                sportClient_->StopMove();
            }
        }
        catch (...)
        {
            PrintLine("StopMove failed");
        }
    }

    void LockMotionInstruction(MotionInstruction instruction)
    {
        if (instruction == MotionInstruction::None || instruction == MotionInstruction::Mixed)
        {
            return;
        }
        try
        {
            logger_.BeginSegment(config_.sceneId, config_.operatorId, ConfiguredTaskMetadata());
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                activeMotionInstruction_ = instruction;
                captureStartDeadline_ = std::chrono::steady_clock::now() + kCaptureStartDelay;
                captureState_ = CaptureState::DelayBeforeLog;
            }
            const std::string label = config_.instruction.empty() ? MotionInstructionName(instruction) : config_.instruction;
            PrintLine("locked segment label=" + label + ", logging starts in 0.5s");
        }
        catch (const std::exception& ex)
        {
            PrintLine(std::string("failed to arm segment logging: ") + ex.what());
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState_ = SafetyState::SafeReady == safetyState_ ? CaptureState::Idle : CaptureState::Fault;
            activeMotionInstruction_ = MotionInstruction::None;
            captureStartDeadline_ = std::chrono::steady_clock::time_point{};
        }
    }

    void UpdateCaptureStateFromMotion(const MotionStateSnapshot& motionState)
    {
        if (config_.captureMode == Config::CaptureMode::Trajectory)
        {
            return;
        }

        MotionInstruction motionInstruction = MotionInstructionFromSnapshot(motionState);
        CaptureState captureState;
        SafetyState safetyState;
        MotionInstruction activeInstruction;
        std::chrono::steady_clock::time_point captureStartDeadline;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState = captureState_;
            safetyState = safetyState_;
            activeInstruction = activeMotionInstruction_;
            captureStartDeadline = captureStartDeadline_;
        }

        if (safetyState != SafetyState::SafeReady)
        {
            return;
        }

        if (captureState == CaptureState::Armed)
        {
            if (motionInstruction != MotionInstruction::None && motionInstruction != MotionInstruction::Mixed)
            {
                LockMotionInstruction(motionInstruction);
            }
            return;
        }

        if (captureState == CaptureState::DelayBeforeLog)
        {
            if (motionInstruction == MotionInstruction::None)
            {
                CancelCurrentSegment("first motion ended before logging delay elapsed");
                return;
            }
            if (motionInstruction == MotionInstruction::Mixed || motionInstruction != activeInstruction)
            {
                CancelCurrentSegment("discarded due to mixed input before logging started");
                return;
            }
            if (std::chrono::steady_clock::now() >= captureStartDeadline)
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                if (captureState_ == CaptureState::DelayBeforeLog)
                {
                    captureState_ = CaptureState::Capturing;
                    PrintLine("segment capture started instruction=" + MotionInstructionName(activeMotionInstruction_));
                }
            }
            return;
        }

        if (captureState == CaptureState::Capturing)
        {
            if (motionInstruction == MotionInstruction::Mixed || (motionInstruction != MotionInstruction::None && motionInstruction != activeInstruction))
            {
                CancelCurrentSegment("discarded due to mixed input");
                return;
            }
            if (motionInstruction == MotionInstruction::None)
            {
                EndSegment();
            }
        }
    }

    void ResetActiveMotionKeys()
    {
        std::lock_guard<std::mutex> lock(keyMutex_);
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

    void SetKeyState(char ch, bool pressed)
    {
        std::lock_guard<std::mutex> lock(keyMutex_);
        KeyActivity* target = nullptr;
        switch (std::tolower(static_cast<unsigned char>(ch)))
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
        if (!target)
        {
            return;
        }
        target->pressed = pressed;
        target->deadline = pressed ? std::chrono::steady_clock::time_point{} : std::chrono::steady_clock::time_point{};
    }

    void MarkTtyKeyActive(char ch)
    {
        const auto deadline = std::chrono::steady_clock::now() + kKeyboardHoldTimeout;
        std::lock_guard<std::mutex> lock(keyMutex_);
        KeyActivity* target = nullptr;
        switch (std::tolower(static_cast<unsigned char>(ch)))
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
        if (!target)
        {
            return;
        }
        target->deadline = deadline;
    }

    VelocityCommand ComputeKeyboardCommand()
    {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(keyMutex_);
        const float deltaSeconds = lastCommandUpdate_.time_since_epoch().count() == 0
                                       ? 0.0f
                                       : std::chrono::duration<float>(now - lastCommandUpdate_).count();
        lastCommandUpdate_ = now;

        const float forward = (keyW_.pressed || keyW_.deadline > now) ? 1.0f : 0.0f;
        const float backward = (keyS_.pressed || keyS_.deadline > now) ? 1.0f : 0.0f;
        const float left = (keyA_.pressed || keyA_.deadline > now) ? 1.0f : 0.0f;
        const float right = (keyD_.pressed || keyD_.deadline > now) ? 1.0f : 0.0f;
        const float yawLeft = (keyQ_.pressed || keyQ_.deadline > now) ? 1.0f : 0.0f;
        const float yawRight = (keyE_.pressed || keyE_.deadline > now) ? 1.0f : 0.0f;

        float targetVx = forward - backward;
        float targetVy = left - right;
        const float targetWz = yawLeft - yawRight;
        const float planarNorm = std::sqrt(targetVx * targetVx + targetVy * targetVy);
        if (planarNorm > 1.0f)
        {
            targetVx /= planarNorm;
            targetVy /= planarNorm;
        }

        smoothedVx_ = SlewTowards(smoothedVx_, targetVx, kLinearAccelPerSecond, kLinearDecelPerSecond, deltaSeconds);
        smoothedVy_ = SlewTowards(smoothedVy_, targetVy, kLinearAccelPerSecond, kLinearDecelPerSecond, deltaSeconds);
        smoothedWz_ = SlewTowards(smoothedWz_, targetWz, kYawAccelPerSecond, kYawDecelPerSecond, deltaSeconds);

        VelocityCommand command;
        command.timestamp = NowSeconds();
        command.vx = smoothedVx_ * static_cast<float>(config_.cmdVxMax);
        command.vy = smoothedVy_ * static_cast<float>(config_.cmdVyMax);
        command.wz = smoothedWz_ * static_cast<float>(config_.cmdWzMax);
        command.valid = true;
        return command;
    }

    void ProcessTeleopChar(char ch, bool allowMotionKeys = true)
    {
        if (ch == kEscapeKey)
        {
            CancelCurrentSegment("cancelled by operator");
            return;
        }

        switch (std::tolower(static_cast<unsigned char>(ch)))
        {
        case 'w':
        case 'a':
        case 's':
        case 'd':
        case 'q':
        case 'e':
            if (allowMotionKeys)
            {
                MarkTtyKeyActive(ch);
            }
            break;
        case 'r':
            BeginSegment();
            break;
        case 't':
            EndSegment();
            break;
        case 'c':
            ClearFaultOrToggleStandState();
            break;
        case 'p':
            PrintStatus();
            break;
        case 'h':
            PrintHelp();
            break;
        case 'x':
            RequestQuit();
            break;
        default:
            if (ch == ' ')
            {
                RequestEmergencyStop("keyboard emergency stop");
            }
            break;
        }
    }

    void KeyboardLoop()
    {
        if (config_.inputBackend == Config::InputBackend::Evdev)
        {
            EvdevKeyboardLoop();
            return;
        }

        while (running_.load())
        {
            char ch = 0;
            const ssize_t bytesRead = ::read(STDIN_FILENO, &ch, 1);
            if (bytesRead <= 0)
            {
                continue;
            }
            ProcessTeleopChar(ch, true);
        }
    }

    void EvdevKeyboardLoop()
    {
        std::vector<pollfd> pfds;
        pfds.push_back(pollfd{evdevFd_, POLLIN, 0});
        if (terminalRawEnabled_)
        {
            pfds.push_back(pollfd{STDIN_FILENO, POLLIN, 0});
        }
        while (running_.load())
        {
            const int pollResult = ::poll(pfds.data(), static_cast<nfds_t>(pfds.size()), 100);
            if (pollResult <= 0)
            {
                continue;
            }

            if ((pfds[0].revents & POLLIN) != 0)
            {
                input_event event{};
                const ssize_t bytesRead = ::read(evdevFd_, &event, sizeof(event));
                if (bytesRead == static_cast<ssize_t>(sizeof(event)) && event.type == EV_KEY)
                {
                    const bool pressed = event.value != 0;
                    HandleEvdevKey(event.code, pressed);
                }
            }

            if (terminalRawEnabled_ && pfds.size() > 1 && (pfds[1].revents & POLLIN) != 0)
            {
                char ch = 0;
                const ssize_t bytesRead = ::read(STDIN_FILENO, &ch, 1);
                if (bytesRead > 0)
                {
                    ProcessTeleopChar(ch, false);
                }
            }
        }
    }

    bool HandleEvdevKey(uint16_t code, bool pressed)
    {
        switch (code)
        {
        case KEY_W:
            SetKeyState('w', pressed);
            return true;
        case KEY_S:
            SetKeyState('s', pressed);
            return true;
        case KEY_A:
            SetKeyState('a', pressed);
            return true;
        case KEY_D:
            SetKeyState('d', pressed);
            return true;
        case KEY_Q:
            SetKeyState('q', pressed);
            return true;
        case KEY_E:
            SetKeyState('e', pressed);
            return true;
        case KEY_ESC:
            if (pressed)
            {
                ProcessTeleopChar(kEscapeKey);
            }
            return true;
        case KEY_SPACE:
            if (pressed)
            {
                ProcessTeleopChar(' ');
            }
            return true;
        case KEY_R:
            if (pressed)
            {
                ProcessTeleopChar('r');
            }
            return true;
        case KEY_T:
            if (pressed)
            {
                ProcessTeleopChar('t');
            }
            return true;
        case KEY_C:
            if (pressed)
            {
                ProcessTeleopChar('c');
            }
            return true;
        case KEY_P:
            if (pressed)
            {
                ProcessTeleopChar('p');
            }
            return true;
        case KEY_H:
            if (pressed)
            {
                ProcessTeleopChar('h');
            }
            return true;
        case KEY_X:
            if (pressed)
            {
                ProcessTeleopChar('x');
            }
            return true;
        default:
            return false;
        }
    }

    void OnSportState(const void* message)
    {
        const auto* state = static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
        LatestState latest;
        latest.timestamp = NowSeconds();
        latest.roll = state->imu_state().rpy()[0];
        latest.pitch = state->imu_state().rpy()[1];
        latest.yaw = state->imu_state().rpy()[2];
        latest.positionX = state->position()[0];
        latest.positionY = state->position()[1];
        latest.positionZ = state->position()[2];
        latest.velocityX = state->velocity()[0];
        latest.velocityY = state->velocity()[1];
        latest.velocityZ = state->velocity()[2];
        latest.yawSpeed = state->yaw_speed();
        latest.bodyHeight = state->body_height();
        latest.gaitType = state->gait_type();
        latest.valid = true;

        std::lock_guard<std::mutex> lock(stateMutex_);
        latestState_ = latest;
    }

    void ControlLoop()
    {
        const auto period = std::chrono::duration<double>(1.0 / config_.loopHz);
        bool lastMoveActive = false;

        while (running_.load())
        {
            const auto cycleStart = std::chrono::steady_clock::now();

            const MotionStateSnapshot motionState = GetMotionStateSnapshot();
            UpdateCaptureStateFromMotion(motionState);
            VelocityCommand command = ComputeKeyboardCommand();
            CaptureState captureState;
            SafetyState safetyState;
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                captureState = captureState_;
                safetyState = safetyState_;
            }

            if (safetyState != SafetyState::SafeReady)
            {
                command = VelocityCommand{};
                command.timestamp = NowSeconds();
                command.valid = true;
            }

            {
                std::lock_guard<std::mutex> lock(commandMutex_);
                latestCommand_ = command;
            }

            const bool moveActive = std::fabs(command.vx) > 1e-6f ||
                                    std::fabs(command.vy) > 1e-6f ||
                                    std::fabs(command.wz) > 1e-6f;

            try
            {
                std::lock_guard<std::mutex> lock(sportClientMutex_);
                if (sportClient_)
                {
                    if (moveActive)
                    {
                        sportClient_->Move(command.vx, command.vy, command.wz);
                    }
                    else if (lastMoveActive)
                    {
                        sportClient_->StopMove();
                    }
                }
            }
            catch (...)
            {
                PrintLine("failed to send motion command");
            }
            lastMoveActive = moveActive;

            std::this_thread::sleep_until(cycleStart + period);
        }
    }

    void VideoLoop()
    {
        const auto period = std::chrono::duration<double>(1.0 / config_.videoPollHz);
        while (running_.load())
        {
            const auto cycleStart = std::chrono::steady_clock::now();

            std::vector<uint8_t> jpegBytes;
            int32_t ret = -1;
            try
            {
                if (videoClient_)
                {
                    ret = videoClient_->GetImageSample(jpegBytes);
                }
            }
            catch (...)
            {
                ret = -1;
            }

            if (ret == 0 && !jpegBytes.empty())
            {
                LatestImage image;
                image.timestamp = NowSeconds();
                image.sequence = nextImageSequence_++;
                image.jpegBytes = std::move(jpegBytes);
                image.valid = true;

                {
                    std::lock_guard<std::mutex> lock(imageMutex_);
                    latestImage_ = std::move(image);
                }
                imageUpdatedCv_.notify_one();
            }

            std::this_thread::sleep_until(cycleStart + period);
        }
    }

    void LoggingLoop()
    {
        const auto period = std::chrono::duration<double>(1.0 / config_.loopHz);
        uint64_t lastLoggedImageSequence = 0;
        auto nextWaitLog = std::chrono::steady_clock::now();
        CaptureState previousCaptureState = CaptureState::Idle;

        while (running_.load())
        {
            const auto cycleStart = std::chrono::steady_clock::now();
            const auto wakeDeadline = cycleStart + period;

            LatestImage image;
            std::unique_lock<std::mutex> lock(imageMutex_);
            imageUpdatedCv_.wait_until(
                lock,
                wakeDeadline,
                [&]()
                {
                    return !running_.load() ||
                           (latestImage_.valid && latestImage_.sequence > lastLoggedImageSequence);
                });
            image = latestImage_;
            lock.unlock();

            LatestState state;
            VelocityCommand action;
            {
                std::lock_guard<std::mutex> lock(stateMutex_);
                state = latestState_;
            }
            {
                std::lock_guard<std::mutex> lock(commandMutex_);
                action = latestCommand_;
            }

            CaptureState captureState;
            SafetyState safetyState;
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                captureState = captureState_;
                safetyState = safetyState_;
            }

            if (captureState == CaptureState::Capturing &&
                previousCaptureState != CaptureState::Capturing &&
                image.valid)
            {
                lastLoggedImageSequence = image.sequence;
            }

            Snapshot snapshot{state, image, action};
            if (safetyState == SafetyState::SafeReady)
            {
                const std::string reason = EvaluateSafetyFault(snapshot, NowSeconds());
                if (!reason.empty())
                {
                    LatchFault(SafetyState::FaultLatched, reason);
                    PrintLine("safety fault latched: " + reason);
                    std::this_thread::sleep_until(wakeDeadline);
                    continue;
                }
            }

            const bool hasFreshImage = image.valid && image.sequence > lastLoggedImageSequence;
            const bool ready = captureState == CaptureState::Capturing && state.valid && hasFreshImage;
            if (ready)
            {
                const double sampleTimestamp = image.timestamp;
                const EffectiveControlAction controlAction = ResolveControlAction(action, sampleTimestamp);
                try
                {
                    logger_.LogStep(
                        sampleTimestamp,
                        state,
                        action,
                        controlAction,
                        image,
                        state.timestamp,
                        action.timestamp,
                        image.timestamp);
                    lastLoggedImageSequence = image.sequence;
                }
                catch (const std::exception& ex)
                {
                    PrintLine(std::string("failed to log sample: ") + ex.what());
                }
            }
            else if (captureState == CaptureState::Capturing && std::chrono::steady_clock::now() >= nextWaitLog)
            {
                std::vector<std::string> missing;
                if (!state.valid)
                {
                    missing.emplace_back("state");
                }
                if (!image.valid)
                {
                    missing.emplace_back("image");
                }
                else if (!hasFreshImage)
                {
                    missing.emplace_back("new_image");
                }
                if (!missing.empty())
                {
                    std::ostringstream oss;
                    oss << "waiting for ";
                    for (size_t index = 0; index < missing.size(); ++index)
                    {
                        if (index > 0)
                        {
                            oss << " and ";
                        }
                        oss << missing[index];
                    }
                    PrintLine(oss.str());
                    nextWaitLog = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                }
            }

            previousCaptureState = captureState;
            std::this_thread::sleep_until(wakeDeadline);
        }
    }

    Config config_;
    TrajectoryLogger logger_;
    RawTerminalGuard terminalGuard_;
    std::atomic<bool> running_{false};
    std::atomic<bool> quitRequested_{false};

    mutable std::mutex outputMutex_;
    mutable std::mutex stateMutex_;
    mutable std::mutex imageMutex_;
    mutable std::mutex commandMutex_;
    mutable std::mutex sportClientMutex_;
    mutable std::mutex stateMachineMutex_;
    mutable std::mutex keyMutex_;
    std::condition_variable imageUpdatedCv_;

    LatestState latestState_;
    LatestImage latestImage_;
    VelocityCommand latestCommand_;
    uint64_t nextImageSequence_ = 1;

    SafetyState safetyState_ = SafetyState::SafeReady;
    std::string latchedFaultReason_;
    CaptureState captureState_ = CaptureState::Idle;
    MotionInstruction activeMotionInstruction_ = MotionInstruction::None;
    std::chrono::steady_clock::time_point captureStartDeadline_{};

    KeyActivity keyW_;
    KeyActivity keyS_;
    KeyActivity keyA_;
    KeyActivity keyD_;
    KeyActivity keyQ_;
    KeyActivity keyE_;
    std::chrono::steady_clock::time_point lastCommandUpdate_{};
    float smoothedVx_ = 0.0f;
    float smoothedVy_ = 0.0f;
    float smoothedWz_ = 0.0f;
    int evdevFd_ = -1;
    bool terminalRawEnabled_ = false;

    std::unique_ptr<unitree::robot::go2::SportClient> sportClient_;
    std::unique_ptr<unitree::robot::go2::VideoClient> videoClient_;
    std::shared_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>> sportStateSubscriber_;

    std::thread keyboardThread_;
    std::thread controlThread_;
    std::thread videoThread_;
    std::thread loggingThread_;
};

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string error;
        const auto config = ParseArgs(argc, argv, error);
        if (!config.has_value())
        {
            std::cerr << "collector argument error: " << error << std::endl;
            return 2;
        }

        CollectorApp app(config.value());
        app.Start();

        while (!app.ShouldQuit())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        app.Shutdown();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "collector fatal error: " << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "collector fatal error: unknown exception" << std::endl;
    }

    return 1;
}
