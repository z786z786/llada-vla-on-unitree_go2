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
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/video/video_client.hpp>
#include "wireless_gamepad.h"
#include "web_ui_server.h"

namespace
{

namespace fs = std::filesystem;
constexpr char kDefaultDataDirName[] = "data";

constexpr char kSportStateTopic[] = "rt/sportmodestate";
constexpr char kWirelessControllerTopic[] = "rt/wirelesscontroller";
constexpr char kSchemaVersion[] = "go2_local_dataset_v1";
constexpr double kStateTimeoutSeconds = 1.0;
constexpr double kWirelessControllerTimeoutSeconds = 1.0;
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
constexpr float kWirelessControllerDiscreteThreshold = 0.35f;
constexpr auto kCaptureStartDelay = std::chrono::milliseconds(500);
constexpr auto kStartupGateReminderInterval = std::chrono::seconds(2);
constexpr auto kTrajectoryFinalizeGracePeriod = std::chrono::milliseconds(400);
constexpr auto kTrajectoryStopReleaseTimeout = std::chrono::milliseconds(1200);
constexpr char kEscapeKey = 27;
constexpr char kBackspace = 127;
constexpr char kCtrlH = 8;
constexpr char kStartupAcknowledgeKey = 'o';

struct TrajectoryMotionGateConfig
{
    float vxStartThreshold = 0.04f;
    float vyStartThreshold = 0.04f;
    float wzStartThreshold = 0.10f;
    float vxStopThreshold = 0.02f;
    float vyStopThreshold = 0.02f;
    float wzStopThreshold = 0.06f;
    size_t startConsecutiveFrames = 4;
    size_t stopConsecutiveFrames = 6;
    size_t preRollFrames = 4;
};

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
        WirelessController,
        Evdev,
        Tty,
    };

    enum class CaptureMode
    {
        SingleAction,
        Trajectory,
    };

    std::string networkInterface;
    fs::path collectorRoot;
    fs::path outputDir;
    double loopHz = 50.0;
    double videoPollHz = 20.0;
    InputBackend inputBackend = InputBackend::WirelessController;
    CaptureMode captureMode = CaptureMode::SingleAction;
    fs::path inputDevice;
    std::string sceneId;
    std::string operatorId;
    std::string instruction;
    std::string taskFamily;
    std::string targetType;
    std::string targetDescription;
    std::string collectorNotes;
    double cmdVxMax = kDefaultCmdVxMax;
    double cmdVyMax = kDefaultCmdVyMax;
    double cmdWzMax = kDefaultCmdWzMax;
    bool webUiEnabled = false;
    int webPort = 8080;
};

struct EditableCollectorConfig
{
    Config::CaptureMode captureMode = Config::CaptureMode::SingleAction;
    std::string sceneId;
    std::string operatorId;
    std::string instruction;
    std::string taskFamily;
    std::string targetType;
    std::string targetDescription;
    std::string collectorNotes;
    double cmdVxMax = kDefaultCmdVxMax;
    double cmdVyMax = kDefaultCmdVyMax;
    double cmdWzMax = kDefaultCmdWzMax;
};

std::optional<Config::CaptureMode> ParseCaptureMode(const std::string& value);

struct TaskMetadata
{
    std::string instruction;
    std::string captureMode;
    std::string taskFamily;
    std::string targetType;
    std::string targetDescription;
    std::string collectorNotes;
    std::string instructionSource;
    std::string segmentStatus;
    std::string success;
    std::string terminationReason;
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

struct WirelessControllerSnapshot
{
    collector::input::Gamepad gamepad;
    double timestamp = 0.0;
    uint64_t sequence = 0;
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
    bool motionInputActive = false;
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
    std::string collectorNotes;
    std::string instructionSource;
    std::string segmentStatus;
    std::string success;
    std::string terminationReason;
    std::string sceneId;
    std::string operatorId;
    size_t numFrames = 0;
    double startTimestamp = 0.0;
    double endTimestamp = 0.0;
};

enum class SegmentStatus
{
    Clean,
    Usable,
    Discard,
};

enum class SuccessLabel
{
    Success,
    Partial,
    Fail,
};

enum class TerminationReason
{
    GoalReached,
    NearGoalStop,
    Occluded,
    TargetLost,
    OperatorStop,
    BadDemo,
    Unsafe,
    Timeout,
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

std::string JoinCsvList(const std::vector<std::string>& values)
{
    std::ostringstream oss;
    for (size_t index = 0; index < values.size(); ++index)
    {
        if (index > 0)
        {
            oss << ",";
        }
        oss << values[index];
    }
    return oss.str();
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
        throw std::runtime_error("格式化时间戳失败");
    }
    return buffer;
}

fs::path CollectorRootFromArgv0(const char* argv0)
{
    std::error_code error;
    fs::path executablePath = fs::read_symlink("/proc/self/exe", error);
    if (error || executablePath.empty())
    {
        error.clear();
        executablePath = fs::weakly_canonical(fs::absolute(argv0), error);
        if (error || executablePath.empty())
        {
            executablePath = fs::absolute(argv0);
        }
    }
    return executablePath.parent_path().parent_path().parent_path();
}

fs::path CollectorDefaultsPath(const fs::path& collectorRoot)
{
    return collectorRoot / "collector_defaults.json";
}

fs::path LegacyCollectorDefaultsPath(const fs::path& collectorRoot)
{
    return collectorRoot / "collector_webui_defaults.json";
}

std::string StartupUnlockHint(Config::InputBackend backend)
{
    switch (backend)
    {
    case Config::InputBackend::WirelessController:
        return "请先按一次原生手柄 Start，让 Go2 进入正常可行走状态后再开始采集";
    case Config::InputBackend::Evdev:
    case Config::InputBackend::Tty:
        return "当前输入后端无需额外解锁，可直接开始采集";
    }
    return "当前输入后端无需额外解锁，可直接开始采集";
}

std::string StartupPromptText(Config::InputBackend backend)
{
    if (backend == Config::InputBackend::WirelessController)
    {
        return "collector 已完成初始化；配置在启动时固定，当前已启用原生手柄直通，可立即自由操作；"
               "检测到一次原生 Start 后才允许开始采集";
    }
    return "collector 已完成初始化；配置在启动时固定，可直接移动和录制";
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

std::string TrajectoryStopPhaseName(bool pendingLabel, bool stopRequested, bool waitingForRelease)
{
    if (pendingLabel)
    {
        return "label_ready";
    }
    if (!stopRequested)
    {
        return "idle";
    }
    if (waitingForRelease)
    {
        return "waiting_release";
    }
    return "finalizing";
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
    case Config::InputBackend::WirelessController:
        return "wireless_controller";
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

std::string SegmentStatusName(SegmentStatus status)
{
    switch (status)
    {
    case SegmentStatus::Clean:
        return "clean";
    case SegmentStatus::Usable:
        return "usable";
    case SegmentStatus::Discard:
        return "discard";
    }
    return "discard";
}

std::string SuccessLabelName(SuccessLabel label)
{
    switch (label)
    {
    case SuccessLabel::Success:
        return "success";
    case SuccessLabel::Partial:
        return "partial";
    case SuccessLabel::Fail:
        return "fail";
    }
    return "fail";
}

std::string TerminationReasonName(TerminationReason reason)
{
    switch (reason)
    {
    case TerminationReason::GoalReached:
        return "goal_reached";
    case TerminationReason::NearGoalStop:
        return "near_goal_stop";
    case TerminationReason::Occluded:
        return "occluded";
    case TerminationReason::TargetLost:
        return "target_lost";
    case TerminationReason::OperatorStop:
        return "operator_stop";
    case TerminationReason::BadDemo:
        return "bad_demo";
    case TerminationReason::Unsafe:
        return "unsafe";
    case TerminationReason::Timeout:
        return "timeout";
    }
    return "operator_stop";
}

bool IsValidSegmentStatusValue(const std::string& value)
{
    return value == SegmentStatusName(SegmentStatus::Clean) ||
           value == SegmentStatusName(SegmentStatus::Usable) ||
           value == SegmentStatusName(SegmentStatus::Discard);
}

bool IsValidSuccessValue(const std::string& value)
{
    return value == SuccessLabelName(SuccessLabel::Success) ||
           value == SuccessLabelName(SuccessLabel::Partial) ||
           value == SuccessLabelName(SuccessLabel::Fail);
}

bool IsValidTerminationReasonValue(const std::string& value)
{
    return value == TerminationReasonName(TerminationReason::GoalReached) ||
           value == TerminationReasonName(TerminationReason::NearGoalStop) ||
           value == TerminationReasonName(TerminationReason::Occluded) ||
           value == TerminationReasonName(TerminationReason::TargetLost) ||
           value == TerminationReasonName(TerminationReason::OperatorStop) ||
           value == TerminationReasonName(TerminationReason::BadDemo) ||
           value == TerminationReasonName(TerminationReason::Unsafe) ||
           value == TerminationReasonName(TerminationReason::Timeout);
}

UiActionResult ActionOk(const std::string& message, const std::string& episodeId = "")
{
    UiActionResult result;
    result.ok = true;
    result.code = "ok";
    result.message = message;
    result.episodeId = episodeId;
    return result;
}

UiActionResult ActionError(const std::string& code, const std::string& message)
{
    UiActionResult result;
    result.ok = false;
    result.code = code;
    result.message = message;
    return result;
}

std::optional<std::string> ExtractJsonStringFieldLocal(const std::string& body, const std::string& key)
{
    const std::string marker = "\"" + key + "\"";
    const size_t keyPos = body.find(marker);
    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }
    const size_t colonPos = body.find(':', keyPos + marker.size());
    if (colonPos == std::string::npos)
    {
        return std::nullopt;
    }
    const size_t quotePos = body.find('"', colonPos + 1);
    if (quotePos == std::string::npos)
    {
        return std::nullopt;
    }
    std::string value;
    bool escaping = false;
    for (size_t index = quotePos + 1; index < body.size(); ++index)
    {
        const char ch = body[index];
        if (escaping)
        {
            value.push_back(ch);
            escaping = false;
            continue;
        }
        if (ch == '\\')
        {
            escaping = true;
            continue;
        }
        if (ch == '"')
        {
            return value;
        }
        value.push_back(ch);
    }
    return std::nullopt;
}

std::optional<double> ExtractJsonNumberFieldLocal(const std::string& body, const std::string& key)
{
    const std::string marker = "\"" + key + "\"";
    const size_t keyPos = body.find(marker);
    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }
    const size_t colonPos = body.find(':', keyPos + marker.size());
    if (colonPos == std::string::npos)
    {
        return std::nullopt;
    }
    size_t begin = body.find_first_of("-0123456789", colonPos + 1);
    if (begin == std::string::npos)
    {
        return std::nullopt;
    }
    size_t end = begin;
    while (end < body.size() &&
           (std::isdigit(static_cast<unsigned char>(body[end])) || body[end] == '.' || body[end] == '-'))
    {
        ++end;
    }
    try
    {
        return std::stod(body.substr(begin, end - begin));
    }
    catch (...)
    {
        return std::nullopt;
    }
}

void ApplyDefaultsFile(const fs::path& defaultsPath, Config& config)
{
    std::ifstream input(defaultsPath);
    if (!input.is_open())
    {
        return;
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string body = buffer.str();

    if (const auto value = ExtractJsonStringFieldLocal(body, "scene_id"); value.has_value())
    {
        config.sceneId = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "operator_id"); value.has_value())
    {
        config.operatorId = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "instruction"); value.has_value())
    {
        config.instruction = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "capture_mode"); value.has_value())
    {
        const auto captureMode = ParseCaptureMode(value.value());
        if (captureMode.has_value())
        {
            config.captureMode = captureMode.value();
        }
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "task_family"); value.has_value())
    {
        config.taskFamily = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "target_type"); value.has_value())
    {
        config.targetType = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "target_description"); value.has_value())
    {
        config.targetDescription = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "collector_notes"); value.has_value())
    {
        config.collectorNotes = value.value();
    }
    if (const auto value = ExtractJsonNumberFieldLocal(body, "cmd_vx_max"); value.has_value())
    {
        config.cmdVxMax = value.value();
    }
    if (const auto value = ExtractJsonNumberFieldLocal(body, "cmd_vy_max"); value.has_value())
    {
        config.cmdVyMax = value.value();
    }
    if (const auto value = ExtractJsonNumberFieldLocal(body, "cmd_wz_max"); value.has_value())
    {
        config.cmdWzMax = value.value();
    }
}

void ApplyCollectorDefaults(const fs::path& collectorRoot, Config& config)
{
    const fs::path defaultsPath = CollectorDefaultsPath(collectorRoot);
    if (fs::exists(defaultsPath))
    {
        ApplyDefaultsFile(defaultsPath, config);
        return;
    }
    const fs::path legacyDefaultsPath = LegacyCollectorDefaultsPath(collectorRoot);
    if (fs::exists(legacyDefaultsPath))
    {
        ApplyDefaultsFile(legacyDefaultsPath, config);
    }
}

std::optional<Config::InputBackend> ParseInputBackend(const std::string& value)
{
    if (value == "wireless" || value == "wireless_controller" || value == "controller" || value == "gamepad")
    {
        return Config::InputBackend::WirelessController;
    }
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

TrajectoryMotionGateConfig BuildTrajectoryMotionGateConfig(double sampleHz)
{
    const double safeHz = std::max(sampleHz, 1.0);
    TrajectoryMotionGateConfig config;
    config.startConsecutiveFrames = std::max<size_t>(3, static_cast<size_t>(std::lround(0.35 * safeHz)));
    config.stopConsecutiveFrames = std::max<size_t>(5, static_cast<size_t>(std::lround(0.60 * safeHz)));
    config.preRollFrames = std::max<size_t>(3, static_cast<size_t>(std::lround(0.40 * safeHz)));
    return config;
}

void PrintUsage(const char* program)
{
    std::cout
        << "用法: " << program << " --network-interface IFACE --scene-id SCENE --operator-id OPERATOR [options]\n\n"
        << "Options:\n"
        << "  --output-dir PATH        数据集根目录（默认：<collector>/" << kDefaultDataDirName << ")\n"
        << "  --loop-hz FLOAT          Control 和 logging loop frequency (default: 50.0)\n"
        << "  --video-poll-hz FLOAT    Camera polling frequency (default: 20.0)\n"
        << "  --input-backend MODE     Input backend: wireless_controller, evdev, or tty (default: wireless_controller)\n"
        << "  --capture-mode MODE      Capture mode: single_action or trajectory (default: single_action)\n"
        << "  --input-device PATH      evdev device path (default: auto-detect keyboard)\n"
        << "  --scene-id TEXT          Required scene identifier\n"
        << "  --operator-id TEXT       Required operator identifier\n"
        << "  --instruction TEXT       Optional semantic instruction for all collected episodes in this run\n"
        << "  --task-family TEXT       Optional task family, e.g. goal_navigation / visual_following / obstacle_aware_navigation\n"
        << "  --target-type TEXT       Optional coarse target type, e.g. door / person / obstacle\n"
        << "  --target-description TEXT  Optional free-text target description; left/right/near/far can be derived offline\n"
        << "  --collector-notes TEXT   Optional free-text notes stored with each episode\n"
        << "  --cmd-vx-max FLOAT       Max forward/backward speed in m/s\n"
        << "  --cmd-vy-max FLOAT       Max strafe speed in m/s\n"
        << "  --cmd-wz-max FLOAT       Max yaw speed in rad/s\n"
        << "  --web-ui                 Enable optional local Web UI\n"
        << "  --web-port INT           Local Web UI port (default: 8080)\n"
        << "  --help                   Show this help\n\n"
        << "Wireless controller:\n"
        << "  Native joystick passthrough is enabled immediately at startup\n"
        << "  Start keeps Go2 native behavior; collector only observes one Start press before allowing recording\n"
        << "  Left stick (ly/lx) forward-backward / strafe\n"
        << "  Right stick X turn left-right\n"
        << "  A start capture  B stop capture  X discard current segment\n"
        << "  R2 emergency stop  Y clear fault or toggle stand up/down\n"
        << "  D-pad Up/Right/Down/Left submit label 1/2/3/4 when label_ready\n"
        << "Keyboard fallback:\n"
        << "  W/S forward-backward  A/D strafe  Q/E turn\n"
        << "  R start  T stop  ESC discard  Space estop  C clear fault  P status  H help  X quit\n"
        << "Input:\n"
        << "  wireless_controller subscribes Go2 native controller topic rt/wirelesscontroller\n"
        << "  evdev supports true multi-key press/release 和 smoother diagonal motion\n"
        << "  tty is a fallback mode 和 may feel less stable for combined keys\n"
        << "Capture modes:\n"
        << "  single_action arms on R, locks one motion key, waits 0.5s, records until key release, 然后进入标注\n"
        << "  trajectory arms on R, auto-detects effective motion before logging, 和 on T waits for motion settle before labeling\n";
}

std::optional<Config> ParseArgs(int argc, char** argv, std::string& error)
{
    Config config;
    const fs::path collectorRoot = CollectorRootFromArgv0(argv[0]);
    config.collectorRoot = collectorRoot;
    config.outputDir = collectorRoot / kDefaultDataDirName;
    ApplyCollectorDefaults(collectorRoot, config);

    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        if (arg == "--network-interface")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--network-interface";
                return std::nullopt;
            }
            config.networkInterface = argv[++index];
        }
        else if (arg == "--output-dir")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--output-dir";
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
                error = "参数缺少取值：--loop-hz";
                return std::nullopt;
            }
            config.loopHz = std::stod(argv[++index]);
        }
        else if (arg == "--video-poll-hz")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--video-poll-hz";
                return std::nullopt;
            }
            config.videoPollHz = std::stod(argv[++index]);
        }
        else if (arg == "--input-backend")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--input-backend";
                return std::nullopt;
            }
            const auto backend = ParseInputBackend(argv[++index]);
            if (!backend.has_value())
            {
                error = "输入后端必须是 wireless_controller、evdev 或 tty";
                return std::nullopt;
            }
            config.inputBackend = backend.value();
        }
        else if (arg == "--capture-mode")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--capture-mode";
                return std::nullopt;
            }
            const auto captureMode = ParseCaptureMode(argv[++index]);
            if (!captureMode.has_value())
            {
                error = "采集模式必须是 single_action 或 trajectory";
                return std::nullopt;
            }
            config.captureMode = captureMode.value();
        }
        else if (arg == "--input-device")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--input-device";
                return std::nullopt;
            }
            config.inputDevice = argv[++index];
        }
        else if (arg == "--scene-id")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--scene-id";
                return std::nullopt;
            }
            config.sceneId = argv[++index];
        }
        else if (arg == "--operator-id")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--operator-id";
                return std::nullopt;
            }
            config.operatorId = argv[++index];
        }
        else if (arg == "--instruction")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--instruction";
                return std::nullopt;
            }
            config.instruction = argv[++index];
        }
        else if (arg == "--task-family")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--task-family";
                return std::nullopt;
            }
            config.taskFamily = argv[++index];
        }
        else if (arg == "--target-type")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--target-type";
                return std::nullopt;
            }
            config.targetType = argv[++index];
        }
        else if (arg == "--target-description")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--target-description";
                return std::nullopt;
            }
            config.targetDescription = argv[++index];
        }
        else if (arg == "--target-instance-id" || arg == "--task-tags")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：" + arg;
                return std::nullopt;
            }
            ++index;
        }
        else if (arg == "--collector-notes")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--collector-notes";
                return std::nullopt;
            }
            config.collectorNotes = argv[++index];
        }
        else if (arg == "--cmd-vx-max")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--cmd-vx-max";
                return std::nullopt;
            }
            config.cmdVxMax = std::stod(argv[++index]);
        }
        else if (arg == "--cmd-vy-max")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--cmd-vy-max";
                return std::nullopt;
            }
            config.cmdVyMax = std::stod(argv[++index]);
        }
        else if (arg == "--cmd-wz-max")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--cmd-wz-max";
                return std::nullopt;
            }
            config.cmdWzMax = std::stod(argv[++index]);
        }
        else if (arg == "--web-ui")
        {
            config.webUiEnabled = true;
        }
        else if (arg == "--web-port")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--web-port";
                return std::nullopt;
            }
            config.webPort = std::stoi(argv[++index]);
        }
        else if (arg == "--help")
        {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        else
        {
            error = "未知参数：" + arg;
            return std::nullopt;
        }
    }

    config.sceneId = Trim(config.sceneId);
    config.operatorId = Trim(config.operatorId);
    config.instruction = Trim(config.instruction);
    config.taskFamily = Trim(config.taskFamily);
    config.targetType = Trim(config.targetType);
    config.targetDescription = Trim(config.targetDescription);
    config.collectorNotes = Trim(config.collectorNotes);

    if (config.networkInterface.empty())
    {
        error = "--network-interface 为必填参数";
        return std::nullopt;
    }
    if (config.sceneId.empty())
    {
        error = "--scene-id 为必填参数";
        return std::nullopt;
    }
    if (config.operatorId.empty())
    {
        error = "--operator-id 为必填参数";
        return std::nullopt;
    }
    if (config.captureMode == Config::CaptureMode::Trajectory && config.instruction.empty())
    {
        error = "使用 --capture-mode trajectory 时必须提供 --instruction";
        return std::nullopt;
    }
    if (config.loopHz <= 0.0 || config.videoPollHz <= 0.0)
    {
        error = "频率参数必须为正数";
        return std::nullopt;
    }
    if (config.cmdVxMax <= 0.0 || config.cmdVyMax <= 0.0 || config.cmdWzMax <= 0.0)
    {
        error = "速度上限参数必须为正数";
        return std::nullopt;
    }
    if (config.webPort <= 0 || config.webPort > 65535)
    {
        error = "--web-port 必须在 1 到 65535 之间";
        return std::nullopt;
    }
    if (config.inputBackend == Config::InputBackend::Evdev && config.inputDevice.empty())
    {
        const auto detected = FindDefaultKeyboardDevice();
        if (!detected.has_value())
        {
            error = "无法在 /dev/input 下自动检测键盘输入设备";
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
            throw std::runtime_error("键盘遥操作模式下，stdin 必须是 tty");
        }
        if (enabled_)
        {
            return;
        }
        if (tcgetattr(STDIN_FILENO, &original_) != 0)
        {
            throw std::runtime_error("获取终端属性失败");
        }
        termios raw = original_;
        raw.c_lflag &= static_cast<unsigned int>(~(ICANON | ECHO));
        raw.c_iflag &= static_cast<unsigned int>(~(IXON | ICRNL));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 1;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0)
        {
            throw std::runtime_error("设置终端 raw 模式失败");
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
        double startTimestamp = 0.0;
        double endTimestamp = 0.0;
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
        if (!pendingFrames_.empty())
        {
            status.startTimestamp = pendingFrames_.front().timestamp;
            status.endTimestamp = pendingFrames_.back().timestamp;
        }
        return status;
    }

    void BeginSegment(
        const std::string& sceneId,
        const std::string& operatorId,
        const TaskMetadata& taskMetadata,
        const std::optional<TrajectoryMotionGateConfig>& trajectoryGate = std::nullopt)
    {
        const std::string trimmedSceneId = Trim(sceneId);
        const std::string trimmedOperatorId = Trim(operatorId);
        if (trimmedSceneId.empty())
        {
            throw std::runtime_error("scene_id 不能为空");
        }
        if (trimmedOperatorId.empty())
        {
            throw std::runtime_error("operator_id 不能为空");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        pendingFrames_.clear();
        pendingSceneId_ = trimmedSceneId;
        pendingOperatorId_ = trimmedOperatorId;
        pendingTaskMetadata_ = taskMetadata;
        pendingEpisodeId_.clear();
        trajectoryGateConfig_ = trajectoryGate;
        effectiveMotionStarted_ = false;
        stopReady_ = false;
        stopRequested_ = false;
        startCandidateCount_ = 0;
        stopCandidateCount_ = 0;
        preRollFrames_.clear();
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
        double imageTimestamp,
        bool motionInputActive)
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

        EpisodeFrame frame{
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
            motionInputActive,
            image.jpegBytes,
        };

        if (!trajectoryGateConfig_.has_value())
        {
            pendingFrames_.push_back(std::move(frame));
            return true;
        }

        PushPreRollFrameLocked(frame);
        if (!effectiveMotionStarted_)
        {
            if (ShouldStartEffectiveMotionLocked(frame))
            {
                effectiveMotionStarted_ = true;
                pendingFrames_.insert(pendingFrames_.end(), preRollFrames_.begin(), preRollFrames_.end());
                preRollFrames_.clear();
                return true;
            }
            return false;
        }

        pendingFrames_.push_back(std::move(frame));
        if (stopRequested_)
        {
            if (ShouldStopEffectiveMotionLocked(pendingFrames_.back()))
            {
                stopReady_ = true;
            }
        }
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
        stopRequested_ = false;
        stopReady_ = false;
        preRollFrames_.clear();
        startCandidateCount_ = 0;
        stopCandidateCount_ = 0;
        if (trajectoryGateConfig_.has_value())
        {
            TrimLeadingTrailingLowMotionFramesLocked();
        }
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
        trajectoryGateConfig_.reset();
        effectiveMotionStarted_ = false;
        stopReady_ = false;
        stopRequested_ = false;
        startCandidateCount_ = 0;
        stopCandidateCount_ = 0;
        preRollFrames_.clear();
    }

    void RequestTrajectoryStop()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (capturing_ && trajectoryGateConfig_.has_value())
        {
            stopRequested_ = true;
            stopCandidateCount_ = 0;
        }
    }

    bool IsTrajectoryStopRequested() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopRequested_;
    }

    bool HasEffectiveMotion() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return effectiveMotionStarted_ && !pendingFrames_.empty();
    }

    bool ConsumeTrajectoryStopReady()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!stopReady_)
        {
            return false;
        }
        stopReady_ = false;
        return true;
    }

    std::optional<std::string> FinalizePendingSegment(const std::string& fallbackInstruction, const TaskMetadata& labelMetadata)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pendingLabel_ || pendingFrames_.empty())
        {
            return std::nullopt;
        }
        const TaskMetadata resolvedTaskMetadata = ResolveTaskMetadata(pendingTaskMetadata_, fallbackInstruction);
        const std::string segmentStatus = Trim(labelMetadata.segmentStatus);
        const std::string success = Trim(labelMetadata.success);
        const std::string terminationReason = Trim(labelMetadata.terminationReason);
        const std::string trimmedInstruction = Trim(resolvedTaskMetadata.instruction);
        if (trimmedInstruction.empty())
        {
            throw std::runtime_error("instruction 不能为空");
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
                    << "  \"collector_notes\": " << JsonString(resolvedTaskMetadata.collectorNotes) << ",\n"
                    << "  \"instruction_source\": " << JsonString(resolvedTaskMetadata.instructionSource) << ",\n"
                    << "  \"segment_status\": " << JsonString(segmentStatus) << ",\n"
                    << "  \"success\": " << JsonString(success) << ",\n"
                    << "  \"termination_reason\": " << JsonString(terminationReason) << ",\n"
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
                throw std::runtime_error("写入图像文件失败");
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
                        << "        \"instruction_source\": " << JsonString(resolvedTaskMetadata.instructionSource) << ",\n"
                        << "        \"segment_status\": " << JsonString(segmentStatus) << ",\n"
                        << "        \"success\": " << JsonString(success) << ",\n"
                        << "        \"termination_reason\": " << JsonString(terminationReason) << ",\n"
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
            throw std::runtime_error("写入 episode 文件失败");
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
        summary.collectorNotes = resolvedTaskMetadata.collectorNotes;
        summary.instructionSource = resolvedTaskMetadata.instructionSource;
        summary.segmentStatus = segmentStatus;
        summary.success = success;
        summary.terminationReason = terminationReason;
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
    bool IsCommandAboveStartThresholdLocked(const EpisodeFrame& frame) const
    {
        const auto& config = trajectoryGateConfig_.value();
        return std::fabs(frame.rawActionVx) > config.vxStartThreshold ||
               std::fabs(frame.rawActionVy) > config.vyStartThreshold ||
               std::fabs(frame.rawActionWz) > config.wzStartThreshold;
    }

    bool IsCommandBelowStopThresholdLocked(const EpisodeFrame& frame) const
    {
        const auto& config = trajectoryGateConfig_.value();
        return std::fabs(frame.rawActionVx) <= config.vxStopThreshold &&
               std::fabs(frame.rawActionVy) <= config.vyStopThreshold &&
               std::fabs(frame.rawActionWz) <= config.wzStopThreshold;
    }

    bool IsLowMotionFrameLocked(const EpisodeFrame& frame) const
    {
        return !frame.motionInputActive && IsCommandBelowStopThresholdLocked(frame);
    }

    bool ShouldStartEffectiveMotionLocked(const EpisodeFrame& frame)
    {
        if (frame.motionInputActive && IsCommandAboveStartThresholdLocked(frame))
        {
            ++startCandidateCount_;
        }
        else
        {
            startCandidateCount_ = 0;
        }
        return startCandidateCount_ >= trajectoryGateConfig_->startConsecutiveFrames;
    }

    bool ShouldStopEffectiveMotionLocked(const EpisodeFrame& frame)
    {
        if (!frame.motionInputActive && IsCommandBelowStopThresholdLocked(frame))
        {
            ++stopCandidateCount_;
        }
        else
        {
            stopCandidateCount_ = 0;
        }
        return stopCandidateCount_ >= trajectoryGateConfig_->stopConsecutiveFrames;
    }

    void PushPreRollFrameLocked(const EpisodeFrame& frame)
    {
        preRollFrames_.push_back(frame);
        const size_t maxFrames = std::max<size_t>(1, trajectoryGateConfig_->preRollFrames);
        if (preRollFrames_.size() > maxFrames)
        {
            preRollFrames_.erase(preRollFrames_.begin());
        }
    }

    void TrimLeadingTrailingLowMotionFramesLocked()
    {
        if (!trajectoryGateConfig_.has_value() || pendingFrames_.empty())
        {
            return;
        }

        size_t startIndex = 0;
        while (startIndex < pendingFrames_.size() && IsLowMotionFrameLocked(pendingFrames_[startIndex]))
        {
            ++startIndex;
        }

        size_t endIndex = pendingFrames_.size();
        while (endIndex > startIndex && IsLowMotionFrameLocked(pendingFrames_[endIndex - 1]))
        {
            --endIndex;
        }

        if (startIndex == 0 && endIndex == pendingFrames_.size())
        {
            return;
        }

        if (startIndex >= endIndex)
        {
            pendingFrames_.clear();
            return;
        }

        pendingFrames_ = std::vector<EpisodeFrame>(pendingFrames_.begin() + static_cast<std::ptrdiff_t>(startIndex),
                                                   pendingFrames_.begin() + static_cast<std::ptrdiff_t>(endIndex));
    }

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
            throw std::runtime_error("重写 index.json 失败");
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
                      << "      \"collector_notes\": " << JsonString(summary.collectorNotes) << ",\n"
                      << "      \"instruction_source\": " << JsonString(summary.instructionSource) << ",\n"
                      << "      \"segment_status\": " << JsonString(summary.segmentStatus) << ",\n"
                      << "      \"success\": " << JsonString(summary.success) << ",\n"
                      << "      \"termination_reason\": " << JsonString(summary.terminationReason) << ",\n"
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
    std::vector<EpisodeFrame> preRollFrames_;
    std::map<std::string, EpisodeSummary> episodeHistory_;
    int nextEpisodeIndex_ = 1;
    bool capturing_ = false;
    bool pendingLabel_ = false;
    bool effectiveMotionStarted_ = false;
    bool stopReady_ = false;
    bool stopRequested_ = false;
    size_t startCandidateCount_ = 0;
    size_t stopCandidateCount_ = 0;
    std::optional<TrajectoryMotionGateConfig> trajectoryGateConfig_;
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
        : config_(config),
          logger_(config.outputDir),
          trajectoryGateConfig_(BuildTrajectoryMotionGateConfig(config.videoPollHz))
    {
        editableConfig_.captureMode = config.captureMode;
        editableConfig_.sceneId = config.sceneId;
        editableConfig_.operatorId = config.operatorId;
        editableConfig_.instruction = config.instruction;
        editableConfig_.taskFamily = config.taskFamily;
        editableConfig_.targetType = config.targetType;
        editableConfig_.targetDescription = config.targetDescription;
        editableConfig_.collectorNotes = config.collectorNotes;
        editableConfig_.cmdVxMax = config.cmdVxMax;
        editableConfig_.cmdVyMax = config.cmdVyMax;
        editableConfig_.cmdWzMax = config.cmdWzMax;
        startupGateActive_ = config_.inputBackend == Config::InputBackend::WirelessController;
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

    EditableCollectorConfig GetEditableConfigSnapshot() const
    {
        std::lock_guard<std::mutex> lock(editableConfigMutex_);
        return editableConfig_;
    }

    void MaybePrintStartupGateReminder()
    {
        bool shouldPrint = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (!startupGateActive_)
            {
                return;
            }
            const auto now = std::chrono::steady_clock::now();
            if (lastStartupGateReminder_.time_since_epoch().count() == 0 ||
                now - lastStartupGateReminder_ >= kStartupGateReminderInterval)
            {
                lastStartupGateReminder_ = now;
                shouldPrint = true;
            }
        }
        if (shouldPrint)
        {
            PrintLine("当前还不能开始采集；" + StartupUnlockHint(config_.inputBackend));
        }
    }

    void ObserveWirelessNativeStart(bool emitLog)
    {
        bool cleared = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (startupGateActive_)
            {
                startupGateActive_ = false;
                lastStartupGateReminder_ = std::chrono::steady_clock::time_point{};
                cleared = true;
            }
        }

        if (cleared && emitLog)
        {
            PrintLine("已检测到原生手柄 Start；Go2 保持原生操控，可开始采集");
        }
    }

    void PrintStartupInstructions() const
    {
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        std::lock_guard<std::mutex> lock(outputMutex_);
        std::cout << "\r\33[2KCollector Startup" << std::endl;
        std::cout << "  采集前检查：确认周围安全、机器人姿态稳定、状态已正常更新" << std::endl;
        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            std::cout << "  主控手柄：左摇杆前后/横移，右摇杆 X 转向，A 开始录制，B 结束录制，X 丢弃当前段" << std::endl;
            std::cout << "  其他按键：Start 保持 Go2 原生行为，并在首次按下后允许 collector 开始录制" << std::endl;
            std::cout << "  安全/标注：R2 急停  Y 清 fault/切换站立  方向键上右下左 = 评分 1/2/3/4" << std::endl;
        }
        else
        {
            std::cout << "  主控键位：W/S 前后  A/D 横移  Q/E 转向  R 开始录制  T 结束录制  ESC 丢弃当前段" << std::endl;
            std::cout << "  其他键位：C 清 fault/切换站立  P 状态  H 帮助  X 退出" << std::endl;
            std::cout << "  标注快捷键：待标注时按 1/2/3/4 直接完成评分" << std::endl;
            std::cout << "  安全键位：Space 急停（始终有效）" << std::endl;
        }
        std::cout << "  运行参数：启动时固定；如需修改 scene/instruction/speed，请重启 collector" << std::endl;
        std::cout << "  录制门控：" << StartupUnlockHint(config_.inputBackend) << std::endl;
        std::cout << "  当前模式：capture_mode=" << CaptureModeName(editable.captureMode)
                  << " input_backend=" << InputBackendName(config_.inputBackend) << std::endl;
        std::cout << "  " << StartupPromptText(config_.inputBackend) << std::endl;
    }

    void ResetTrajectoryStopFlowLocked()
    {
        trajectoryStopRequested_ = false;
        trajectoryStopWaitingForRelease_ = false;
        trajectoryStopFinalizeDeadline_ = std::chrono::steady_clock::time_point{};
        trajectoryStopForceFinalizeDeadline_ = std::chrono::steady_clock::time_point{};
    }

    enum class TrajectoryStopFlowAction
    {
        None,
        StartGracePeriod,
        StartGracePeriodAfterTimeout,
        Finalize,
    };

    static bool HasSteadyDeadline(const std::chrono::steady_clock::time_point& deadline)
    {
        return deadline.time_since_epoch().count() != 0;
    }

    static const char* TrajectoryStopStatusMessage(bool waitingForRelease)
    {
        return waitingForRelease ? "Stopping segment... waiting for input release"
                                 : "Finalizing segment... submit a score";
    }

    void ResetCaptureProgressLocked(const std::string& pendingLabelFallback = std::string())
    {
        activeMotionInstruction_ = MotionInstruction::None;
        captureStartDeadline_ = std::chrono::steady_clock::time_point{};
        pendingLabelFallbackInstruction_ = pendingLabelFallback;
        ResetTrajectoryStopFlowLocked();
    }

    void EnterIdleOrFaultStateLocked(const std::string& pendingLabelFallback = std::string())
    {
        captureState_ = safetyState_ == SafetyState::SafeReady ? CaptureState::Idle : CaptureState::Fault;
        ResetCaptureProgressLocked(pendingLabelFallback);
    }

    void BeginTrajectoryStopLocked(bool motionInputActive)
    {
        trajectoryStopRequested_ = true;
        trajectoryStopWaitingForRelease_ = motionInputActive;
        trajectoryStopFinalizeDeadline_ =
            motionInputActive ? std::chrono::steady_clock::time_point{}
                              : std::chrono::steady_clock::now() + kTrajectoryFinalizeGracePeriod;
        trajectoryStopForceFinalizeDeadline_ = std::chrono::steady_clock::now() + kTrajectoryStopReleaseTimeout;
    }

    void StartTrajectoryFinalizeGracePeriod(const std::string& reason, bool emitLog)
    {
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (captureState_ != CaptureState::Capturing || !trajectoryStopRequested_)
            {
                return;
            }
            trajectoryStopWaitingForRelease_ = false;
            if (trajectoryStopFinalizeDeadline_.time_since_epoch().count() == 0)
            {
                trajectoryStopFinalizeDeadline_ = std::chrono::steady_clock::now() + kTrajectoryFinalizeGracePeriod;
            }
        }
        if (emitLog)
        {
            PrintLine(reason);
        }
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
        else if (config_.inputBackend == Config::InputBackend::Evdev)
        {
            terminalRawEnabled_ = terminalGuard_.TryEnable();
            OpenEvdevInput();
        }
        else
        {
            terminalRawEnabled_ = terminalGuard_.TryEnable();
        }

        unitree::robot::ChannelFactory::Instance()->Init(0, config_.networkInterface);

        sportClient_ = std::make_unique<unitree::robot::go2::SportClient>();
        sportClient_->SetTimeout(10.0f);
        sportClient_->Init();
        SetWirelessControllerPassthroughEnabled(true, "collector startup uses native joystick");

        sportStateSubscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(kSportStateTopic);
        sportStateSubscriber_->InitChannel(std::bind(&CollectorApp::OnSportState, this, std::placeholders::_1), 1);

        wirelessControllerSubscriber_ =
            std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>>(kWirelessControllerTopic);
        wirelessControllerSubscriber_->InitChannel(std::bind(&CollectorApp::OnWirelessController, this, std::placeholders::_1), 1);

        videoClient_ = std::make_unique<unitree::robot::go2::VideoClient>();
        videoClient_->SetTimeout(1.0f);
        videoClient_->Init();

        running_.store(true);
        controlThread_ = std::thread(&CollectorApp::ControlLoop, this);
        loggingThread_ = std::thread(&CollectorApp::LoggingLoop, this);
        keyboardThread_ = std::thread(&CollectorApp::KeyboardLoop, this);
        videoThread_ = std::thread(&CollectorApp::VideoLoop, this);

        if (config_.webUiEnabled)
        {
            WebUiServerConfig webConfig;
            webConfig.port = config_.webPort;
            webConfig.assetDir = (config_.collectorRoot / "native" / "web_ui_assets").string();
            webConfig.statusProvider = [this]() { return GetUiStatusSnapshot(); };
            webConfig.startHandler = [this]() { return RequestBeginSegment(); };
            webConfig.stopHandler = [this]() { return RequestStopSegmentForLabel(); };
            webConfig.discardHandler = [this]() { return RequestDiscardSegment("discarded by web ui"); };
            webConfig.estopHandler = [this]() { return RequestEmergencyStopFromUi(); };
            webConfig.clearFaultHandler = [this]() { return RequestClearFaultFromUi(); };
            webConfig.quitHandler = [this]() { return RequestQuitFromUi(); };
            webConfig.submitLabelHandler = [this](const SegmentLabelInput& input)
            {
                return SubmitPendingLabel(input);
            };
            webConfig.latestImageJpegProvider = [this]() { return GetLatestImageJpeg(); };
            webConfig.nextImageFrameProvider = [this](uint64_t lastSequence, int timeoutMs)
            {
                return WaitForNextImageFrame(lastSequence, timeoutMs);
            };
            webUiServer_ = std::make_unique<WebUiServer>(webConfig);
            webUiServer_->Start();
        }

        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            latestCommand_.valid = true;
            latestCommand_.timestamp = NowSeconds();
        }

        PrintLine("session 目录：" + logger_.OutputDir().string());
        PrintLine("输入后端：" + InputBackendName(config_.inputBackend) +
                  (config_.inputDevice.empty() ? "" : " device=" + config_.inputDevice.string()));
        if (config_.webUiEnabled)
        {
            PrintLine("Web UI：http://127.0.0.1:" + std::to_string(config_.webPort));
        }
        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            PrintLine("collector 已就绪；原生手柄直通已启用，可立即自由操作");
            PrintLine("录制门控：检测到一次原生 Start 后才允许开始采集");
        }
        else
        {
            PrintLine("collector 已就绪；当前输入后端可直接移动和录制");
        }
        PrintLine("运行参数在启动时固定；如需修改 scene/instruction/speed，请重启 collector");
        PrintStartupInstructions();
    }

    void Shutdown()
    {
        if (!running_.exchange(false))
        {
            return;
        }

        imageUpdatedCv_.notify_all();
        if (webUiServer_)
        {
            webUiServer_->Stop();
        }

        if (sportStateSubscriber_)
        {
            sportStateSubscriber_->CloseChannel();
            sportStateSubscriber_.reset();
        }
        if (wirelessControllerSubscriber_)
        {
            wirelessControllerSubscriber_->CloseChannel();
            wirelessControllerSubscriber_.reset();
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
        SetWirelessControllerPassthroughEnabled(true, "collector shutdown restore native joystick");
        unitree::robot::ChannelFactory::Instance()->Release();
        CloseEvdevInput();
        terminalGuard_.Disable();
        logger_.CleanupIfEmpty();
        webUiServer_.reset();
    }

    UiStatusSnapshot GetUiStatusSnapshot() const
    {
        const Snapshot latest = CollectLatestSnapshot();
        const auto loggerStatus = logger_.GetStatus();
        UiStatusSnapshot snapshot;
        CaptureState captureState;
        SafetyState safetyState;
        std::string faultReason;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState = captureState_;
            safetyState = safetyState_;
            faultReason = latchedFaultReason_;
            snapshot.startupGateActive = startupGateActive_;
            snapshot.stopPhase = TrajectoryStopPhaseName(false, trajectoryStopRequested_, trajectoryStopWaitingForRelease_);
        }
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        const double nowSeconds = NowSeconds();
        snapshot.running = running_.load();
        snapshot.webUiEnabled = config_.webUiEnabled;
        snapshot.webPort = config_.webPort;
        snapshot.startupPrompt = StartupPromptText(config_.inputBackend);
        snapshot.sessionDir = logger_.OutputDir().string();
        snapshot.captureMode = CaptureModeName(editable.captureMode);
        snapshot.bufferedFrames = loggerStatus.bufferedFrames;
        snapshot.segmentDurationSeconds = loggerStatus.bufferedFrames >= 2
                                              ? std::max(0.0, loggerStatus.endTimestamp - loggerStatus.startTimestamp)
                                              : 0.0;
        snapshot.stateValid = latest.state.valid;
        snapshot.imageValid = latest.image.valid;
        snapshot.stateAgeSeconds = AgeSeconds(latest.state.timestamp, nowSeconds);
        snapshot.imageAgeSeconds = AgeSeconds(latest.image.timestamp, nowSeconds);
        snapshot.robotConnected = snapshot.stateValid && snapshot.stateAgeSeconds >= 0.0 && snapshot.stateAgeSeconds <= kStateTimeoutSeconds;
        snapshot.bodyHeight = latest.state.bodyHeight;
        snapshot.roll = latest.state.roll;
        snapshot.pitch = latest.state.pitch;
        snapshot.yaw = latest.state.yaw;
        snapshot.commandVx = latest.action.vx;
        snapshot.commandVy = latest.action.vy;
        snapshot.commandWz = latest.action.wz;
        snapshot.sceneId = editable.sceneId;
        snapshot.operatorId = editable.operatorId;
        snapshot.instruction = editable.instruction;
        snapshot.taskFamily = editable.taskFamily;
        snapshot.targetType = editable.targetType;
        snapshot.targetDescription = editable.targetDescription;
        snapshot.collectorNotes = editable.collectorNotes;
        snapshot.cmdVxMax = editable.cmdVxMax;
        snapshot.cmdVyMax = editable.cmdVyMax;
        snapshot.cmdWzMax = editable.cmdWzMax;
        snapshot.networkInterface = config_.networkInterface;
        snapshot.outputDir = config_.outputDir.string();
        snapshot.loopHz = config_.loopHz;
        snapshot.videoPollHz = config_.videoPollHz;
        snapshot.inputBackend = InputBackendName(config_.inputBackend);
        snapshot.inputDevice = config_.inputDevice.string();
        snapshot.defaultsPath = CollectorDefaultsPath(config_.collectorRoot).string();
        snapshot.pendingLabelActive = loggerStatus.pendingLabel;
        snapshot.pendingEpisodeId = loggerStatus.pendingEpisodeId;
        snapshot.pendingLabelBufferedFrames = loggerStatus.bufferedFrames;
        snapshot.captureState = CaptureStateName(captureState);
        if (loggerStatus.pendingLabel)
        {
            snapshot.stopPhase = "label_ready";
        }
        snapshot.safetyState = SafetyStateName(safetyState);
        snapshot.faultReason = faultReason;
        snapshot.recording = captureState == CaptureState::Capturing;
        snapshot.actions =
            BuildUiActionAvailability(loggerStatus, captureState, safetyState, snapshot.startupGateActive);
        return snapshot;
    }

    UiActionResult RequestBeginSegment()
    {
        return BeginSegmentInternal(true);
    }

    UiActionResult RequestStopSegmentForLabel()
    {
        return StopSegmentForLabelInternal(true);
    }

    UiActionResult RequestDiscardSegment(const std::string& reason)
    {
        return DiscardSegmentInternal(reason, true);
    }

    UiActionResult RequestEmergencyStopFromUi()
    {
        SafetyState safetyState;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            safetyState = safetyState_;
        }
        if (safetyState == SafetyState::EstopLatched)
        {
            return ActionError("already_estop", "急停已锁定");
        }
        RequestEmergencyStop("web_ui_estop");
        return ActionOk("急停已锁定");
    }

    UiActionResult RequestClearFaultFromUi()
    {
        bool hasFault = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            hasFault = safetyState_ != SafetyState::SafeReady;
        }
        if (!hasFault)
        {
            return ActionError("no_fault", "当前没有 safety fault");
        }
        std::string reason;
        if (!CanClearFault(reason))
        {
            return ActionError("cannot_clear_fault", "无法清除 safety fault：" + reason);
        }
        ClearLatchedFault();
        PrintLine("safety fault 已清除");
        return ActionOk("safety fault 已清除");
    }

    UiActionResult RequestQuitFromUi()
    {
        RequestQuit();
        PrintLine("collector 正在退出");
        return ActionOk("collector 正在退出");
    }

    UiActionResult SubmitPendingLabel(const SegmentLabelInput& input)
    {
        const auto loggerStatus = logger_.GetStatus();
        if (!loggerStatus.pendingLabel)
        {
            return ActionError("no_pending_label", "当前没有待标注区间");
        }

        const std::string segmentStatus = Trim(input.segmentStatus);
        if (!IsValidSegmentStatusValue(segmentStatus))
        {
            return ActionError("validation_error", "segment_status 非法");
        }
        if (segmentStatus == SegmentStatusName(SegmentStatus::Discard))
        {
            return DiscardSegmentInternal("discarded by web label", true);
        }

        const std::string success = Trim(input.success);
        const std::string terminationReason = Trim(input.terminationReason);
        if (!IsValidSuccessValue(success))
        {
            return ActionError("validation_error", "success 非法");
        }
        if (!IsValidTerminationReasonValue(terminationReason))
        {
            return ActionError("validation_error", "termination_reason 非法");
        }

        TaskMetadata labelMetadata;
        labelMetadata.segmentStatus = segmentStatus;
        labelMetadata.success = success;
        labelMetadata.terminationReason = terminationReason;
        return FinalizePendingLabelInternal(labelMetadata, true);
    }

    std::vector<uint8_t> GetLatestImageJpeg() const
    {
        std::lock_guard<std::mutex> lock(imageMutex_);
        if (!latestImage_.valid || latestImage_.jpegBytes.empty())
        {
            return {};
        }
        return latestImage_.jpegBytes;
    }

    UiImageFrame WaitForNextImageFrame(uint64_t lastSequence, int timeoutMs)
    {
        UiImageFrame frame;
        std::unique_lock<std::mutex> lock(imageMutex_);
        const auto hasNewFrame = [&]()
        {
            return !running_.load() ||
                   (latestImage_.valid && !latestImage_.jpegBytes.empty() && latestImage_.sequence > lastSequence);
        };

        if (!hasNewFrame())
        {
            imageUpdatedCv_.wait_for(lock, std::chrono::milliseconds(std::max(timeoutMs, 1)), hasNewFrame);
        }

        if (!running_.load() ||
            !latestImage_.valid ||
            latestImage_.jpegBytes.empty() ||
            latestImage_.sequence <= lastSequence)
        {
            return frame;
        }

        frame.timestamp = latestImage_.timestamp;
        frame.sequence = latestImage_.sequence;
        frame.jpegBytes = latestImage_.jpegBytes;
        frame.valid = true;
        return frame;
    }

private:
    struct Snapshot
    {
        LatestState state;
        LatestImage image;
        VelocityCommand action;
    };

    Snapshot CollectLatestSnapshot() const
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
        return snapshot;
    }

    static UiActionAvailability BuildUiActionAvailability(const TrajectoryLogger::Status& loggerStatus,
                                                          CaptureState captureState,
                                                          SafetyState safetyState,
                                                          bool startupGateActive)
    {
        UiActionAvailability actions;
        actions.canStartRecording = !startupGateActive &&
                                    safetyState == SafetyState::SafeReady &&
                                    captureState == CaptureState::Idle &&
                                    !loggerStatus.pendingLabel;
        actions.canStopRecording = captureState == CaptureState::Capturing;
        actions.canDiscardSegment = captureState == CaptureState::Armed ||
                                    captureState == CaptureState::DelayBeforeLog ||
                                    captureState == CaptureState::Capturing ||
                                    loggerStatus.pendingLabel;
        actions.canSubmitLabel = loggerStatus.pendingLabel;
        actions.canEstop = safetyState != SafetyState::EstopLatched;
        actions.canClearFault = safetyState != SafetyState::SafeReady;
        return actions;
    }

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

    WirelessControllerSnapshot GetWirelessControllerSnapshot() const
    {
        std::lock_guard<std::mutex> lock(controllerMutex_);
        return wirelessControllerSnapshot_;
    }

    MotionStateSnapshot GetMotionStateSnapshot() const
    {
        MotionStateSnapshot motionState;
        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            const WirelessControllerSnapshot controllerState = GetWirelessControllerSnapshot();
            const double ageSeconds = AgeSeconds(controllerState.timestamp, NowSeconds());
            if (!controllerState.valid || ageSeconds < 0.0 || ageSeconds > kWirelessControllerTimeoutSeconds)
            {
                return motionState;
            }

            motionState.forward = controllerState.gamepad.rawLy > kWirelessControllerDiscreteThreshold;
            motionState.backward = controllerState.gamepad.rawLy < -kWirelessControllerDiscreteThreshold;
            motionState.left = controllerState.gamepad.rawLx < -kWirelessControllerDiscreteThreshold;
            motionState.right = controllerState.gamepad.rawLx > kWirelessControllerDiscreteThreshold;
            motionState.yawLeft = controllerState.gamepad.rawRx < -kWirelessControllerDiscreteThreshold;
            motionState.yawRight = controllerState.gamepad.rawRx > kWirelessControllerDiscreteThreshold;
            return motionState;
        }

        std::lock_guard<std::mutex> lock(keyMutex_);
        const auto now = std::chrono::steady_clock::now();
        motionState.forward = keyW_.pressed || keyW_.deadline > now;
        motionState.backward = keyS_.pressed || keyS_.deadline > now;
        motionState.left = keyA_.pressed || keyA_.deadline > now;
        motionState.right = keyD_.pressed || keyD_.deadline > now;
        motionState.yawLeft = keyQ_.pressed || keyQ_.deadline > now;
        motionState.yawRight = keyE_.pressed || keyE_.deadline > now;
        return motionState;
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

    static bool IsMotionInputActive(const MotionStateSnapshot& motionState)
    {
        return motionState.forward || motionState.backward || motionState.left ||
               motionState.right || motionState.yawLeft || motionState.yawRight;
    }

    TrajectoryStopFlowAction EvaluateTrajectoryStopFlowLocked(const MotionStateSnapshot& motionState) const
    {
        if (captureState_ != CaptureState::Capturing || !trajectoryStopRequested_)
        {
            return TrajectoryStopFlowAction::None;
        }

        const auto now = std::chrono::steady_clock::now();
        if (trajectoryStopWaitingForRelease_)
        {
            if (!IsMotionInputActive(motionState))
            {
                return TrajectoryStopFlowAction::StartGracePeriod;
            }
            if (HasSteadyDeadline(trajectoryStopForceFinalizeDeadline_) && now >= trajectoryStopForceFinalizeDeadline_)
            {
                return TrajectoryStopFlowAction::StartGracePeriodAfterTimeout;
            }
            return TrajectoryStopFlowAction::None;
        }

        if (HasSteadyDeadline(trajectoryStopFinalizeDeadline_) && now >= trajectoryStopFinalizeDeadline_)
        {
            return TrajectoryStopFlowAction::Finalize;
        }
        return TrajectoryStopFlowAction::None;
    }

    void UpdateTrajectoryStopFlow(const MotionStateSnapshot& motionState)
    {
        TrajectoryStopFlowAction action = TrajectoryStopFlowAction::None;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            action = EvaluateTrajectoryStopFlowLocked(motionState);
        }

        if (action == TrajectoryStopFlowAction::StartGracePeriodAfterTimeout)
        {
            StartTrajectoryFinalizeGracePeriod("Stopping segment timed out; finalizing segment... submit a score", true);
            return;
        }
        if (action == TrajectoryStopFlowAction::StartGracePeriod)
        {
            StartTrajectoryFinalizeGracePeriod("Finalizing segment... submit a score", true);
            return;
        }
        if (action == TrajectoryStopFlowAction::Finalize)
        {
            CompleteTrajectoryStopIfReady(true);
        }
    }

    void PrintLine(const std::string& line) const
    {
        std::lock_guard<std::mutex> lock(outputMutex_);
        std::cout << "\r\33[2K" << line << std::endl;
    }

    std::optional<std::string> PromptForSelection(
        const std::string& title,
        const std::vector<std::pair<std::string, std::string>>& options)
    {
        if (!isatty(STDIN_FILENO))
        {
            PrintLine("stdin 不是 tty，无法进行区间标注");
            return std::nullopt;
        }

        promptCancelRequested_.store(false);
        promptActive_.store(true);
        struct PromptRestore
        {
            std::atomic<bool>& promptActive;

            ~PromptRestore()
            {
                promptActive.store(false);
            }
        } restore{promptActive_};

        while (running_.load())
        {
            {
                std::lock_guard<std::mutex> lock(outputMutex_);
                std::cout << "\r\33[2K" << title << std::endl;
                for (const auto& option : options)
                {
                    std::cout << "  " << option.first << " = " << option.second << std::endl;
                }
                std::cout << "  ESC = discard" << std::endl;
                std::cout << "> " << std::flush;
            }

            while (running_.load())
            {
                if (promptCancelRequested_.exchange(false))
                {
                    return std::nullopt;
                }

                pollfd pfd{STDIN_FILENO, POLLIN, 0};
                const int pollResult = ::poll(&pfd, 1, 100);
                if (pollResult < 0)
                {
                    PrintLine("等待标注输入失败");
                    return std::nullopt;
                }
                if (pollResult == 0 || (pfd.revents & POLLIN) == 0)
                {
                    continue;
                }

                char ch = 0;
                const ssize_t bytesRead = ::read(STDIN_FILENO, &ch, 1);
                if (bytesRead <= 0)
                {
                    continue;
                }
                if (ch == kEscapeKey)
                {
                    return std::nullopt;
                }
                if (ch == '\r' || ch == '\n')
                {
                    continue;
                }

                const std::string input(1, ch);
                for (const auto& option : options)
                {
                    if (input == option.first)
                    {
                        return option.second;
                    }
                }
                PrintLine("输入无效，请按编号重新输入，或按 ESC 丢弃");
                break;
            }
        }
        return std::nullopt;
    }

    std::optional<SegmentLabelInput> PromptQuickLabel()
    {
        const auto selection = PromptForSelection(
            "请选择数据评分",
            {
                {"1", "好的成功示范 (clean + success + goal_reached)"},
                {"2", "可用但不完美 (usable + partial + near_goal_stop)"},
                {"3", "失败但有价值 (usable + fail + operator_stop)"},
                {"4", "丢弃 (discard)"},
            });
        if (!selection.has_value())
        {
            return std::nullopt;
        }

        if (selection.value() == "好的成功示范 (clean + success + goal_reached)")
        {
            return BuildPresetLabelFromShortcut('1');
        }
        if (selection.value() == "可用但不完美 (usable + partial + near_goal_stop)")
        {
            return BuildPresetLabelFromShortcut('2');
        }
        if (selection.value() == "失败但有价值 (usable + fail + operator_stop)")
        {
            return BuildPresetLabelFromShortcut('3');
        }
        if (selection.value() == "丢弃 (discard)")
        {
            return BuildPresetLabelFromShortcut('4');
        }
        return std::nullopt;
    }

    static std::optional<SegmentLabelInput> BuildPresetLabelFromShortcut(char shortcut)
    {
        SegmentLabelInput input;
        switch (shortcut)
        {
        case '1':
            input.segmentStatus = SegmentStatusName(SegmentStatus::Clean);
            input.success = SuccessLabelName(SuccessLabel::Success);
            input.terminationReason = TerminationReasonName(TerminationReason::GoalReached);
            return input;
        case '2':
            input.segmentStatus = SegmentStatusName(SegmentStatus::Usable);
            input.success = SuccessLabelName(SuccessLabel::Partial);
            input.terminationReason = TerminationReasonName(TerminationReason::NearGoalStop);
            return input;
        case '3':
            input.segmentStatus = SegmentStatusName(SegmentStatus::Usable);
            input.success = SuccessLabelName(SuccessLabel::Fail);
            input.terminationReason = TerminationReasonName(TerminationReason::OperatorStop);
            return input;
        case '4':
            input.segmentStatus = SegmentStatusName(SegmentStatus::Discard);
            return input;
        default:
            return std::nullopt;
        }
    }

    void PrintHelp() const
    {
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        std::lock_guard<std::mutex> lock(outputMutex_);
        std::cout << "\r\33[2KControls:" << std::endl;
        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            std::cout << "  Start 保持 Go2 原生行为；collector 仅在观察到一次 Start 后允许开始录制" << std::endl;
            std::cout << "  A start capture  B stop capture  X discard current segment" << std::endl;
            std::cout << "  left stick = forward/backward + strafe  right stick X = yaw" << std::endl;
            std::cout << "  R2 emergency stop  Y clear fault or toggle stand up/down" << std::endl;
            std::cout << "  pending label: D-pad up/right/down/left = 1/2/3/4" << std::endl;
        }
        else
        {
            std::cout << "  W/S forward/backward  A/D strafe  Q/E yaw" << std::endl;
            std::cout << "  R start capture flow  ESC cancel current armed/capture segment" << std::endl;
            std::cout << "  Space emergency stop  C clear fault or toggle stand up/down  P status  H help  X quit" << std::endl;
            std::cout << "  pending label: 1 good demo  2 usable imperfect  3 failed but valuable  4 discard" << std::endl;
        }
        if (editable.captureMode == Config::CaptureMode::SingleAction)
        {
            std::cout << "  single_action 模式：检测到第一个有效单动作输入后，等待 0.5 秒开始录制，松键自动结束" << std::endl;
            std::cout << "  若设置 --instruction，则使用该语义指令；否则回退为动作标签" << std::endl;
        }
        else
        {
            std::cout << "  trajectory 模式：按 R 进入 armed，连续有效动作后开始写入，按 T 请求结束并等待动作回落后标注" << std::endl;
            std::cout << "  该模式下必须提供 --instruction，并作为轨迹级语义标签保存" << std::endl;
        }
        std::cout << "  input_backend=" << InputBackendName(config_.inputBackend)
                  << " capture_mode=" << CaptureModeName(editable.captureMode) << std::endl;
    }

    void PrintStatus() const
    {
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        const Snapshot latest = CollectLatestSnapshot();
        const MotionStateSnapshot motionState = GetMotionStateSnapshot();
        const WirelessControllerSnapshot controllerState =
            config_.inputBackend == Config::InputBackend::WirelessController ? GetWirelessControllerSnapshot()
                                                                             : WirelessControllerSnapshot{};

        const auto loggerStatus = logger_.GetStatus();
        const double nowSeconds = NowSeconds();
        SafetyState safetyState;
        std::string faultReason;
        CaptureState captureState;
        std::string activeInstruction;
        double armDelayRemaining = 0.0;
        bool startupGateActive = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            safetyState = safetyState_;
            faultReason = latchedFaultReason_;
            captureState = captureState_;
            activeInstruction = MotionInstructionName(activeMotionInstruction_);
            startupGateActive = startupGateActive_;
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
            << " startup_gate=" << (startupGateActive ? "waiting_native_start" : "ready")
            << " safety_state=" << SafetyStateName(safetyState)
            << " fault_reason=" << (faultReason.empty() ? "-" : faultReason)
            << " scene_id=" << editable.sceneId
            << " operator_id=" << editable.operatorId
            << " capture_mode=" << CaptureModeName(editable.captureMode)
            << " configured_instruction=" << (editable.instruction.empty() ? "-" : editable.instruction)
            << " task_family=" << (editable.taskFamily.empty() ? "-" : editable.taskFamily)
            << " target_type=" << (editable.targetType.empty() ? "-" : editable.targetType)
            << " target_description=" << (editable.targetDescription.empty() ? "-" : editable.targetDescription)
            << " buffered_frames=" << loggerStatus.bufferedFrames
            << " state=" << latest.state.valid
            << " image=" << latest.image.valid
            << " state_age_s=" << std::fixed << std::setprecision(3) << AgeSeconds(latest.state.timestamp, nowSeconds)
            << " image_age_s=" << AgeSeconds(latest.image.timestamp, nowSeconds)
            << " command=(" << latest.action.vx << "," << latest.action.vy << "," << latest.action.wz << ")";
        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            oss << " controller_axes=("
                << "ly=" << controllerState.gamepad.rawLy
                << ",lx=" << controllerState.gamepad.rawLx
                << ",rx=" << controllerState.gamepad.rawRx
                << ")"
                << " wireless_mode=" << (nativeJoystickEnabled_.load() ? "native_passthrough" : "passthrough_disabled");
        }
        else
        {
            oss << " motion_keys=("
                << (motionState.forward ? "W" : "")
                << (motionState.backward ? "S" : "")
                << (motionState.left ? "A" : "")
                << (motionState.right ? "D" : "")
                << (motionState.yawLeft ? "Q" : "")
                << (motionState.yawRight ? "E" : "")
                << ")";
        }
        oss
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
            throw std::runtime_error("打开输入设备失败：" + config_.inputDevice.string());
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
        const Snapshot snapshot = CollectLatestSnapshot();
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
            ResetCaptureProgressLocked();
        }
        logger_.DiscardPendingSegment();
        ResetActiveMotionKeys();
        SetWirelessControllerPassthroughEnabled(false, "safety fault latched");
        StopMotion();
    }

    void RequestEmergencyStop(const std::string& reason)
    {
        LatchFault(SafetyState::EstopLatched, reason);
        PrintLine("急停已锁定：" + reason);
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
                if (config_.inputBackend == Config::InputBackend::WirelessController)
                {
                    SetWirelessControllerPassthroughEnabled(true, "fault cleared");
                }
                PrintLine("safety fault 已清除");
            }
            else
            {
                PrintLine("无法清除 safety fault：" + reason);
            }
            return;
        }

        if (segmentActive)
        {
            PrintLine("请先结束或丢弃当前采集段");
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
                PrintLine("sport client 尚未准备好");
                return;
            }
            sportClient_->StopMove();
            if (shouldStandDown)
            {
                sportClient_->StandDown();
                PrintLine("已请求趴下");
            }
            else
            {
                sportClient_->StandUp();
                PrintLine("已请求站起");
            }
        }
        catch (...)
        {
            PrintLine("切换站立状态失败");
        }
    }

    TaskMetadata ConfiguredTaskMetadata() const
    {
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        TaskMetadata metadata;
        metadata.instruction = editable.instruction;
        metadata.captureMode = CaptureModeName(editable.captureMode);
        metadata.taskFamily = editable.taskFamily;
        metadata.targetType = editable.targetType;
        metadata.targetDescription = editable.targetDescription;
        metadata.collectorNotes = editable.collectorNotes;
        metadata.instructionSource = metadata.instruction.empty() ? "motion_label" : "semantic_text";
        return metadata;
    }

    UiActionResult BeginSegmentInternal(bool emitLog)
    {
        const bool hasPendingLabel = logger_.GetStatus().pendingLabel;
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        const TaskMetadata configuredTaskMetadata = ConfiguredTaskMetadata();
        std::lock_guard<std::mutex> lock(stateMachineMutex_);
        if (startupGateActive_)
        {
            if (emitLog)
            {
                PrintLine("当前还不能开始采集；" + StartupUnlockHint(config_.inputBackend));
            }
            return ActionError("startup_gate_active", StartupUnlockHint(config_.inputBackend));
        }
        if (safetyState_ != SafetyState::SafeReady)
        {
            if (emitLog)
            {
                PrintLine("当前 safety fault 已锁定，无法启动采集段");
            }
            return ActionError("fault_latched", "当前 safety fault 已锁定");
        }
        if (captureState_ == CaptureState::Capturing)
        {
            if (emitLog)
            {
                PrintLine("当前已有采集段正在进行");
            }
            return ActionError("already_capturing", "当前已有采集段正在进行");
        }
        if (captureState_ == CaptureState::Armed || captureState_ == CaptureState::DelayBeforeLog)
        {
            if (emitLog)
            {
                PrintLine("当前已有采集段处于 armed 状态");
            }
            return ActionError("already_armed", "当前已有采集段处于 armed 状态");
        }
        if (hasPendingLabel)
        {
            if (emitLog)
            {
                PrintLine("当前有待标注区间，请先提交或丢弃");
            }
            return ActionError("pending_label", "当前有待标注区间");
        }

        try
        {
            if (editable.captureMode == Config::CaptureMode::Trajectory)
            {
                logger_.BeginSegment(editable.sceneId, editable.operatorId, configuredTaskMetadata, trajectoryGateConfig_);
                captureState_ = CaptureState::Capturing;
                ResetCaptureProgressLocked();
                if (emitLog)
                {
                    PrintLine("trajectory 已 armed；检测到连续有效动作后开始写入，使用停止键结束，使用丢弃键取消");
                }
            }
            else
            {
                captureState_ = CaptureState::Armed;
                ResetCaptureProgressLocked();
                if (emitLog)
                {
                    PrintLine("采集段已 armed，等待单一有效动作输入");
                }
            }
        }
        catch (const std::exception& ex)
        {
            if (emitLog)
            {
                PrintLine(std::string("启动采集段失败：") + ex.what());
            }
            return ActionError("start_failed", ex.what());
        }

        return ActionOk("segment started");
    }

    UiActionResult FinalizeCaptureForLabelInternal(MotionInstruction instruction, bool emitLog)
    {
        const size_t frameCount = logger_.EndSegmentForLabel();
        if (frameCount == 0)
        {
            logger_.DiscardPendingSegment();
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            EnterIdleOrFaultStateLocked();
            if (emitLog)
            {
                PrintLine("该段未记录到有效动作帧，已丢弃");
            }
            return ActionError("empty_segment", "该段未记录到有效动作帧");
        }

        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            EnterIdleOrFaultStateLocked(MotionInstructionName(instruction));
        }
        if (emitLog)
        {
            PrintLine("Entering labeling state... submit 1/2/3/4 or D-pad score");
        }
        return ActionOk("segment stopped for labeling");
    }

    UiActionResult RequestTrajectoryStopInternal(MotionInstruction instruction, bool emitLog)
    {
        if (!logger_.HasEffectiveMotion())
        {
            return FinalizeCaptureForLabelInternal(instruction, emitLog);
        }

        const bool waitingForRelease = IsMotionInputActive(GetMotionStateSnapshot());
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (captureState_ != CaptureState::Capturing)
            {
                return ActionError("not_capturing", "当前没有可结束的采集段");
            }
            if (trajectoryStopRequested_)
            {
                return ActionError("already_stopping", "trajectory 已收到结束请求，正在等待动作回落");
            }
            BeginTrajectoryStopLocked(waitingForRelease);
        }

        if (emitLog)
        {
            PrintLine(TrajectoryStopStatusMessage(waitingForRelease));
        }
        return ActionOk("trajectory stop requested");
    }

    UiActionResult StopSegmentForLabelInternal(bool emitLog)
    {
        const bool hasPendingLabel = logger_.GetStatus().pendingLabel;
        const Config::CaptureMode captureMode = GetEditableConfigSnapshot().captureMode;
        MotionInstruction instruction = MotionInstruction::None;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (captureState_ != CaptureState::Capturing)
            {
                if (hasPendingLabel)
                {
                    return ActionError("already_waiting_label", "当前区间正在等待标注");
                }
                if (emitLog)
                {
                    PrintLine("当前没有可结束的采集段");
                }
                return ActionError("not_capturing", "当前没有可结束的采集段");
            }
            instruction = activeMotionInstruction_;
        }

        if (captureMode == Config::CaptureMode::Trajectory)
        {
            return RequestTrajectoryStopInternal(instruction, emitLog);
        }
        return FinalizeCaptureForLabelInternal(instruction, emitLog);
    }

    void CompleteTrajectoryStopIfReady(bool emitLog)
    {
        MotionInstruction instruction = MotionInstruction::None;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (captureState_ != CaptureState::Capturing || !trajectoryStopRequested_)
            {
                return;
            }
            instruction = activeMotionInstruction_;
        }
        const UiActionResult result = FinalizeCaptureForLabelInternal(instruction, emitLog);
        if (!result.ok || config_.webUiEnabled)
        {
            return;
        }
        PromptAndFinalizePendingSegment();
    }

    UiActionResult DiscardSegmentInternal(const std::string& reason, bool emitLog)
    {
        const bool hasPendingLabel = logger_.GetStatus().pendingLabel;
        bool hasActiveOrPending = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            hasActiveOrPending = captureState_ == CaptureState::Capturing ||
                                 captureState_ == CaptureState::Armed ||
                                 captureState_ == CaptureState::DelayBeforeLog;
            if (!hasActiveOrPending)
            {
                hasActiveOrPending = hasPendingLabel;
            }
            if (!hasActiveOrPending)
            {
                return ActionError("nothing_to_discard", "当前没有可丢弃区间");
            }
            EnterIdleOrFaultStateLocked();
        }
        logger_.DiscardPendingSegment();
        if (emitLog)
        {
            PrintLine("采集段已丢弃：" + reason);
        }
        return ActionOk("segment discarded");
    }

    UiActionResult FinalizePendingLabelInternal(const TaskMetadata& labelMetadata, bool emitLog)
    {
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        const std::string savedInstruction = editable.instruction.empty()
                                                 ? pendingLabelFallbackInstruction_
                                                 : editable.instruction;
        try
        {
            const auto episodeId = logger_.FinalizePendingSegment(pendingLabelFallbackInstruction_, labelMetadata);
            if (!episodeId.has_value())
            {
                return ActionError("no_pending_label", "当前没有待保存的采集段");
            }
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                pendingLabelFallbackInstruction_.clear();
            }
            if (emitLog)
            {
                PrintLine(
                    "已保存 episode " + episodeId.value() +
                    " 指令=" + savedInstruction +
                    " segment_status=" + labelMetadata.segmentStatus +
                    " success=" + labelMetadata.success +
                    " termination_reason=" + labelMetadata.terminationReason);
            }
            return ActionOk("episode saved", episodeId.value());
        }
        catch (const std::exception& ex)
        {
            logger_.DiscardPendingSegment();
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                EnterIdleOrFaultStateLocked();
            }
            if (emitLog)
            {
                PrintLine(std::string("结束采集段失败：") + ex.what());
            }
            return ActionError("finalize_failed", ex.what());
        }
    }

    void PromptAndFinalizePendingSegment()
    {
        const auto quickLabel = PromptQuickLabel();
        if (!quickLabel.has_value())
        {
            DiscardSegmentInternal("cancelled during labeling", true);
            return;
        }

        const UiActionResult result = SubmitPendingLabel(quickLabel.value());
        if (!result.ok)
        {
            PrintLine("标注提交失败：" + result.message);
        }
    }

    void BeginSegment()
    {
        BeginSegmentInternal(true);
    }

    void EndSegment()
    {
        const UiActionResult result = StopSegmentForLabelInternal(true);
        if (!result.ok)
        {
            return;
        }
        if (!logger_.GetStatus().pendingLabel)
        {
            return;
        }
        if (config_.webUiEnabled)
        {
            return;
        }
        PromptAndFinalizePendingSegment();
    }

    void CancelCurrentSegment(const std::string& reason)
    {
        DiscardSegmentInternal(reason, true);
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
            PrintLine("StopMove 调用失败");
        }
    }

    void SetWirelessControllerPassthroughEnabled(bool enabled, const std::string& reason)
    {
        if (config_.inputBackend != Config::InputBackend::WirelessController)
        {
            return;
        }

        const bool alreadyEnabled = nativeJoystickEnabled_.load();
        const bool stateKnown = nativeJoystickStateKnown_.load();
        if (stateKnown && alreadyEnabled == enabled)
        {
            return;
        }

        std::string error;
        {
            std::lock_guard<std::mutex> lock(sportClientMutex_);
            if (!sportClient_)
            {
                return;
            }

            try
            {
                const int32_t result = sportClient_->SwitchJoystick(enabled);
                if (result == 0)
                {
                    nativeJoystickEnabled_.store(enabled);
                    nativeJoystickStateKnown_.store(true);
                    if (!enabled)
                    {
                        // When the collector latches a fault, stop any residual
                        // native joystick motion immediately.
                        sportClient_->StopMove();
                    }
                }
                else
                {
                    error = std::string("切换原生手柄直通失败，SwitchJoystick(") +
                            (enabled ? "true" : "false") + ") 返回码=" + std::to_string(result);
                }
            }
            catch (...)
            {
                error = std::string("切换原生手柄直通失败，SwitchJoystick(") +
                        (enabled ? "true" : "false") + ") 调用失败";
            }
        }

        if (!error.empty())
        {
            PrintLine(error);
            return;
        }

        PrintLine(std::string("原生手柄直通已") + (enabled ? "启用" : "禁用") + "：" + reason);
    }

    void LockMotionInstruction(MotionInstruction instruction)
    {
        if (instruction == MotionInstruction::None || instruction == MotionInstruction::Mixed)
        {
            return;
        }
        try
        {
            const EditableCollectorConfig editable = GetEditableConfigSnapshot();
            logger_.BeginSegment(editable.sceneId, editable.operatorId, ConfiguredTaskMetadata());
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                activeMotionInstruction_ = instruction;
                captureStartDeadline_ = std::chrono::steady_clock::now() + kCaptureStartDelay;
                captureState_ = CaptureState::DelayBeforeLog;
            }
            const std::string label = editable.instruction.empty() ? MotionInstructionName(instruction) : editable.instruction;
            PrintLine("已锁定采集段标签=" + label + "，将在 0.5 秒后开始记录");
        }
        catch (const std::exception& ex)
        {
            PrintLine(std::string("准备采集段记录失败：") + ex.what());
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            EnterIdleOrFaultStateLocked();
        }
    }

    void UpdateCaptureStateFromMotion(const MotionStateSnapshot& motionState)
    {
        if (GetEditableConfigSnapshot().captureMode == Config::CaptureMode::Trajectory)
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
                CancelCurrentSegment("首次动作在录制延迟结束前就已结束");
                return;
            }
            if (motionInstruction == MotionInstruction::Mixed || motionInstruction != activeInstruction)
            {
                CancelCurrentSegment("录制开始前检测到混合输入，已丢弃");
                return;
            }
            if (std::chrono::steady_clock::now() >= captureStartDeadline)
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                if (captureState_ == CaptureState::DelayBeforeLog)
                {
                    captureState_ = CaptureState::Capturing;
                    PrintLine("segment capture started 指令=" + MotionInstructionName(activeMotionInstruction_));
                }
            }
            return;
        }

        if (captureState == CaptureState::Capturing)
        {
            if (motionInstruction == MotionInstruction::Mixed || (motionInstruction != MotionInstruction::None && motionInstruction != activeInstruction))
            {
                CancelCurrentSegment("检测到混合输入，已丢弃");
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

    VelocityCommand ComputeInputCommand()
    {
        bool stopRequested = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            stopRequested = trajectoryStopRequested_;
        }
        const auto now = std::chrono::steady_clock::now();
        const float deltaSeconds = lastCommandUpdate_.time_since_epoch().count() == 0
                                       ? 0.0f
                                       : std::chrono::duration<float>(now - lastCommandUpdate_).count();
        lastCommandUpdate_ = now;

        float targetVx = 0.0f;
        float targetVy = 0.0f;
        float targetWz = 0.0f;
        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            const WirelessControllerSnapshot controllerState = GetWirelessControllerSnapshot();
            const double ageSeconds = AgeSeconds(controllerState.timestamp, NowSeconds());
            if (!stopRequested && controllerState.valid && ageSeconds >= 0.0 && ageSeconds <= kWirelessControllerTimeoutSeconds)
            {
                targetVx = controllerState.gamepad.rawLy;
                targetVy = -controllerState.gamepad.rawLx;
                targetWz = -controllerState.gamepad.rawRx;
            }
        }
        else
        {
            std::lock_guard<std::mutex> lock(keyMutex_);
            const float forward = stopRequested ? 0.0f : ((keyW_.pressed || keyW_.deadline > now) ? 1.0f : 0.0f);
            const float backward = stopRequested ? 0.0f : ((keyS_.pressed || keyS_.deadline > now) ? 1.0f : 0.0f);
            const float left = stopRequested ? 0.0f : ((keyA_.pressed || keyA_.deadline > now) ? 1.0f : 0.0f);
            const float right = stopRequested ? 0.0f : ((keyD_.pressed || keyD_.deadline > now) ? 1.0f : 0.0f);
            const float yawLeft = stopRequested ? 0.0f : ((keyQ_.pressed || keyQ_.deadline > now) ? 1.0f : 0.0f);
            const float yawRight = stopRequested ? 0.0f : ((keyE_.pressed || keyE_.deadline > now) ? 1.0f : 0.0f);
            targetVx = forward - backward;
            targetVy = left - right;
            targetWz = yawLeft - yawRight;
        }

        const float planarNorm = std::sqrt(targetVx * targetVx + targetVy * targetVy);
        if (planarNorm > 1.0f)
        {
            targetVx /= planarNorm;
            targetVy /= planarNorm;
        }

        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            smoothedVx_ = targetVx;
            smoothedVy_ = targetVy;
            smoothedWz_ = targetWz;
        }
        else
        {
            smoothedVx_ = SlewTowards(smoothedVx_, targetVx, kLinearAccelPerSecond, kLinearDecelPerSecond, deltaSeconds);
            smoothedVy_ = SlewTowards(smoothedVy_, targetVy, kLinearAccelPerSecond, kLinearDecelPerSecond, deltaSeconds);
            smoothedWz_ = SlewTowards(smoothedWz_, targetWz, kYawAccelPerSecond, kYawDecelPerSecond, deltaSeconds);
        }

        VelocityCommand command;
        command.timestamp = NowSeconds();
        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            // Native passthrough uses the robot's own joystick motion path; the
            // collector records normalized operator intent instead of issuing
            // another scaled Move() command.
            command.vx = smoothedVx_;
            command.vy = smoothedVy_;
            command.wz = smoothedWz_;
        }
        else
        {
            const EditableCollectorConfig editable = GetEditableConfigSnapshot();
            command.vx = smoothedVx_ * static_cast<float>(editable.cmdVxMax);
            command.vy = smoothedVy_ * static_cast<float>(editable.cmdVyMax);
            command.wz = smoothedWz_ * static_cast<float>(editable.cmdWzMax);
        }
        command.valid = true;
        return command;
    }

    void ProcessTeleopChar(char ch, bool allowMotionKeys = true)
    {
        const char lowered = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == kEscapeKey)
        {
            CancelCurrentSegment("cancelled by operator");
            return;
        }

        switch (lowered)
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
        case kStartupAcknowledgeKey:
            if (config_.inputBackend == Config::InputBackend::WirelessController)
            {
                PrintLine("collector 不需要 Apply；请直接按原生手柄 Start，检测到后即可开始采集");
            }
            else
            {
                PrintLine("collector 运行参数在启动时固定；请修改启动命令后重启 collector");
            }
            break;
        case 'x':
            RequestQuit();
            break;
        case '1':
        case '2':
        case '3':
        case '4':
            SubmitPresetLabelShortcut(lowered);
            break;
        default:
            if (ch == ' ')
            {
                RequestEmergencyStop("键盘急停");
            }
            break;
        }
    }

    void SubmitPresetLabelShortcut(char shortcut)
    {
        const auto preset = BuildPresetLabelFromShortcut(shortcut);
        if (!preset.has_value())
        {
            return;
        }
        if (!logger_.GetStatus().pendingLabel)
        {
            return;
        }
        const UiActionResult result = SubmitPendingLabel(preset.value());
        if (!result.ok)
        {
            PrintLine("标注提交失败：" + result.message);
        }
    }

    void HandleWirelessControllerActions(const collector::input::Gamepad& gamepad)
    {
        if (gamepad.start.onPress)
        {
            ObserveWirelessNativeStart(true);
        }
        if (gamepad.A.onPress)
        {
            ProcessTeleopChar('r', false);
        }
        if (gamepad.B.onPress)
        {
            ProcessTeleopChar('t', false);
        }
        if (gamepad.X.onPress)
        {
            ProcessTeleopChar(kEscapeKey, false);
        }
        if (gamepad.R2.onPress)
        {
            ProcessTeleopChar(' ', false);
        }
        if (gamepad.Y.onPress)
        {
            ProcessTeleopChar('c', false);
        }
        if (gamepad.up.onPress)
        {
            SubmitPresetLabelShortcut('1');
        }
        if (gamepad.right.onPress)
        {
            SubmitPresetLabelShortcut('2');
        }
        if (gamepad.down.onPress)
        {
            SubmitPresetLabelShortcut('3');
        }
        if (gamepad.left.onPress)
        {
            SubmitPresetLabelShortcut('4');
        }
    }

    void KeyboardLoop()
    {
        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            WirelessControllerLoop();
            return;
        }

        if (config_.inputBackend == Config::InputBackend::Evdev)
        {
            EvdevKeyboardLoop();
            return;
        }

        while (running_.load())
        {
            if (promptActive_.load())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            char ch = 0;
            const ssize_t bytesRead = ::read(STDIN_FILENO, &ch, 1);
            if (bytesRead <= 0)
            {
                continue;
            }
            ProcessTeleopChar(ch, true);
        }
    }

    void WirelessControllerLoop()
    {
        uint64_t lastHandledSequence = 0;
        while (running_.load())
        {
            const WirelessControllerSnapshot controllerState = GetWirelessControllerSnapshot();
            if (controllerState.valid && controllerState.sequence != lastHandledSequence)
            {
                lastHandledSequence = controllerState.sequence;
                HandleWirelessControllerActions(controllerState.gamepad);
            }

            if (terminalRawEnabled_ && !promptActive_.load())
            {
                pollfd pfd{STDIN_FILENO, POLLIN, 0};
                const int pollResult = ::poll(&pfd, 1, 0);
                if (pollResult > 0 && (pfd.revents & POLLIN) != 0)
                {
                    char ch = 0;
                    const ssize_t bytesRead = ::read(STDIN_FILENO, &ch, 1);
                    if (bytesRead > 0)
                    {
                        ProcessTeleopChar(ch, false);
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
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
                if (promptActive_.load())
                {
                    continue;
                }
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
                if (promptActive_.load())
                {
                    promptCancelRequested_.store(true);
                    return true;
                }
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
        case KEY_O:
            if (pressed)
            {
                ProcessTeleopChar(kStartupAcknowledgeKey);
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

    void OnWirelessController(const void* message)
    {
        const auto* controller = static_cast<const unitree_go::msg::dds_::WirelessController_*>(message);
        std::lock_guard<std::mutex> lock(controllerMutex_);
        wirelessControllerSnapshot_.gamepad.Update(*controller);
        wirelessControllerSnapshot_.timestamp = NowSeconds();
        wirelessControllerSnapshot_.sequence += 1;
        wirelessControllerSnapshot_.valid = true;
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
            UpdateTrajectoryStopFlow(motionState);
            VelocityCommand command = ComputeInputCommand();
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

            if (config_.inputBackend != Config::InputBackend::WirelessController)
            {
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
                    PrintLine("发送运动指令失败");
                }
                lastMoveActive = moveActive;
            }

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
                    PrintLine("safety fault 已锁定：" + reason);
                    std::this_thread::sleep_until(wakeDeadline);
                    continue;
                }
            }

            const bool hasFreshImage = image.valid && image.sequence > lastLoggedImageSequence;
            const bool ready = captureState == CaptureState::Capturing && state.valid && hasFreshImage;
            if (ready)
            {
                const MotionStateSnapshot motionState = GetMotionStateSnapshot();
                const bool motionInputActive = IsMotionInputActive(motionState);
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
                        image.timestamp,
                        motionInputActive);
                    lastLoggedImageSequence = image.sequence;
                }
                catch (const std::exception& ex)
                {
                    PrintLine(std::string("记录样本失败：") + ex.what());
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
                if (!missing.empty())
                {
                    std::ostringstream oss;
                    oss << "等待 ";
                    for (size_t index = 0; index < missing.size(); ++index)
                    {
                        if (index > 0)
                        {
                            oss << " 和 ";
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
    TrajectoryMotionGateConfig trajectoryGateConfig_;
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
    mutable std::mutex controllerMutex_;
    mutable std::mutex editableConfigMutex_;
    std::condition_variable imageUpdatedCv_;

    LatestState latestState_;
    LatestImage latestImage_;
    VelocityCommand latestCommand_;
    uint64_t nextImageSequence_ = 1;

    SafetyState safetyState_ = SafetyState::SafeReady;
    std::string latchedFaultReason_;
    CaptureState captureState_ = CaptureState::Idle;
    bool startupGateActive_ = false;
    std::chrono::steady_clock::time_point lastStartupGateReminder_{};
    MotionInstruction activeMotionInstruction_ = MotionInstruction::None;
    std::chrono::steady_clock::time_point captureStartDeadline_{};
    std::string pendingLabelFallbackInstruction_;
    bool trajectoryStopRequested_ = false;
    bool trajectoryStopWaitingForRelease_ = false;
    std::chrono::steady_clock::time_point trajectoryStopFinalizeDeadline_{};
    std::chrono::steady_clock::time_point trajectoryStopForceFinalizeDeadline_{};
    EditableCollectorConfig editableConfig_;

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
    WirelessControllerSnapshot wirelessControllerSnapshot_{};
    std::atomic<bool> nativeJoystickEnabled_{false};
    std::atomic<bool> nativeJoystickStateKnown_{false};
    int evdevFd_ = -1;
    bool terminalRawEnabled_ = false;

    std::unique_ptr<unitree::robot::go2::SportClient> sportClient_;
    std::unique_ptr<unitree::robot::go2::VideoClient> videoClient_;
    std::shared_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>> sportStateSubscriber_;
    std::shared_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>> wirelessControllerSubscriber_;

    std::thread keyboardThread_;
    std::thread controlThread_;
    std::thread videoThread_;
    std::thread loggingThread_;
    std::atomic<bool> promptActive_{false};
    std::atomic<bool> promptCancelRequested_{false};
    std::unique_ptr<WebUiServer> webUiServer_;
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
            std::cerr << "collector 参数错误：" << error << std::endl;
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
        std::cerr << "collector 致命错误：" << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "collector 致命错误：未知异常" << std::endl;
    }

    return 1;
}
