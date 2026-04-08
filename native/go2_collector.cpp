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
#include "web_ui_server.h"

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
    fs::path collectorRoot;
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
    std::string targetInstanceId;
    std::string collectorNotes;
    std::vector<std::string> taskTags;
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
    std::string targetInstanceId;
    std::vector<std::string> taskTags;
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
    const fs::path executablePath = fs::absolute(argv0);
    return executablePath.parent_path().parent_path().parent_path();
}

fs::path CollectorDefaultsPath(const fs::path& collectorRoot)
{
    return collectorRoot / "collector_webui_defaults.json";
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
    if (const auto value = ExtractJsonStringFieldLocal(body, "target_instance_id"); value.has_value())
    {
        config.targetInstanceId = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "task_tags_csv"); value.has_value())
    {
        config.taskTags = SplitCsvList(value.value());
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

void SaveDefaultsFile(const fs::path& defaultsPath, const EditableCollectorConfig& config)
{
    std::ofstream output(defaultsPath, std::ios::out | std::ios::trunc);
    if (!output.is_open())
    {
        throw std::runtime_error("写入 collector 默认配置失败");
    }

    output << "{\n"
           << "  \"scene_id\": " << JsonString(config.sceneId) << ",\n"
           << "  \"operator_id\": " << JsonString(config.operatorId) << ",\n"
           << "  \"instruction\": " << JsonString(config.instruction) << ",\n"
           << "  \"capture_mode\": " << JsonString(CaptureModeName(config.captureMode)) << ",\n"
           << "  \"task_family\": " << JsonString(config.taskFamily) << ",\n"
           << "  \"target_type\": " << JsonString(config.targetType) << ",\n"
           << "  \"target_description\": " << JsonString(config.targetDescription) << ",\n"
           << "  \"target_instance_id\": " << JsonString(config.targetInstanceId) << ",\n"
           << "  \"task_tags_csv\": " << JsonString(JoinCsvList(config.taskTags)) << ",\n"
           << "  \"collector_notes\": " << JsonString(config.collectorNotes) << ",\n"
           << "  \"cmd_vx_max\": " << config.cmdVxMax << ",\n"
           << "  \"cmd_vy_max\": " << config.cmdVyMax << ",\n"
           << "  \"cmd_wz_max\": " << config.cmdWzMax << "\n"
           << "}\n";
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
        << "用法: " << program << " --network-interface IFACE --scene-id SCENE --operator-id OPERATOR [options]\n\n"
        << "Options:\n"
        << "  --output-dir PATH        数据集根目录（默认：<collector>/" << kDefaultDataDirName << ")\n"
        << "  --loop-hz FLOAT          Control 和 logging loop frequency (default: 50.0)\n"
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
        << "  --web-ui                 Enable optional local Web UI\n"
        << "  --web-port INT           Local Web UI port (default: 8080)\n"
        << "  --help                   Show this help\n\n"
        << "Keyboard:\n"
        << "  W/S  forward/backward\n"
        << "  A/D  strafe left/right\n"
        << "  Q/E  turn left/right\n"
        << "  R    start capture flow for the selected mode\n"
        << "  T    manually end the current capture and start labeling if already recording\n"
        << "  ESC  cancel current armed/capture segment\n"
        << "  Space emergency stop 和 latch safety fault\n"
        << "  C    clear fault or toggle stand up/down\n"
        << "  P    print status\n"
        << "  H    print help\n"
        << "  X    quit\n"
        << "Input:\n"
        << "  evdev supports true multi-key press/release 和 smoother diagonal motion\n"
        << "  tty is a fallback mode 和 may feel less stable for combined keys\n"
        << "Capture modes:\n"
        << "  single_action arms on R, locks one motion key, waits 0.5s, records until key release, 然后进入标注\n"
        << "  trajectory starts recording immediately on R, allows turning/strafe changes, 和 ends on T for labeling\n";
}

std::optional<Config> ParseArgs(int argc, char** argv, std::string& error)
{
    Config config;
    const fs::path collectorRoot = CollectorRootFromArgv0(argv[0]);
    config.collectorRoot = collectorRoot;
    config.outputDir = collectorRoot / kDefaultDataDirName;
    ApplyDefaultsFile(CollectorDefaultsPath(collectorRoot), config);

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
                error = "输入后端必须是 evdev 或 tty";
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
        else if (arg == "--target-instance-id")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--target-instance-id";
                return std::nullopt;
            }
            config.targetInstanceId = argv[++index];
        }
        else if (arg == "--task-tags")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--task-tags";
                return std::nullopt;
            }
            config.taskTags = SplitCsvList(argv[++index]);
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
    config.targetInstanceId = Trim(config.targetInstanceId);
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

    void BeginSegment(const std::string& sceneId, const std::string& operatorId, const TaskMetadata& taskMetadata)
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
                    << "  \"target_instance_id\": " << JsonString(resolvedTaskMetadata.targetInstanceId) << ",\n"
                    << "  \"task_tags\": " << JsonStringArray(resolvedTaskMetadata.taskTags) << ",\n"
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
                        << "        \"target_instance_id\": " << JsonString(resolvedTaskMetadata.targetInstanceId) << ",\n"
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
        summary.targetInstanceId = resolvedTaskMetadata.targetInstanceId;
        summary.taskTags = resolvedTaskMetadata.taskTags;
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
                      << "      \"target_instance_id\": " << JsonString(summary.targetInstanceId) << ",\n"
                      << "      \"task_tags\": " << JsonStringArray(summary.taskTags) << ",\n"
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
        editableConfig_.captureMode = config.captureMode;
        editableConfig_.sceneId = config.sceneId;
        editableConfig_.operatorId = config.operatorId;
        editableConfig_.instruction = config.instruction;
        editableConfig_.taskFamily = config.taskFamily;
        editableConfig_.targetType = config.targetType;
        editableConfig_.targetDescription = config.targetDescription;
        editableConfig_.targetInstanceId = config.targetInstanceId;
        editableConfig_.collectorNotes = config.collectorNotes;
        editableConfig_.taskTags = config.taskTags;
        editableConfig_.cmdVxMax = config.cmdVxMax;
        editableConfig_.cmdVyMax = config.cmdVyMax;
        editableConfig_.cmdWzMax = config.cmdWzMax;
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

    UiActionResult UpdateEditableConfig(const UiConfigUpdateInput& input)
    {
        EditableCollectorConfig updated = GetEditableConfigSnapshot();
        const auto loggerStatus = logger_.GetStatus();
        CaptureState captureState;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState = captureState_;
        }
        updated.sceneId = Trim(input.sceneId);
        updated.operatorId = Trim(input.operatorId);
        updated.instruction = Trim(input.instruction);
        updated.taskFamily = Trim(input.taskFamily);
        updated.targetType = Trim(input.targetType);
        updated.targetDescription = Trim(input.targetDescription);
        updated.targetInstanceId = Trim(input.targetInstanceId);
        updated.collectorNotes = Trim(input.collectorNotes);
        updated.taskTags = SplitCsvList(input.taskTagsCsv);
        updated.cmdVxMax = input.cmdVxMax;
        updated.cmdVyMax = input.cmdVyMax;
        updated.cmdWzMax = input.cmdWzMax;

        const auto captureMode = ParseCaptureMode(Trim(input.captureMode));
        if (!captureMode.has_value())
        {
            return ActionError("validation_error", "capture_mode 非法");
        }
        updated.captureMode = captureMode.value();
        if (updated.captureMode != GetEditableConfigSnapshot().captureMode &&
            (captureState == CaptureState::Capturing ||
             captureState == CaptureState::Armed ||
             captureState == CaptureState::DelayBeforeLog ||
             loggerStatus.pendingLabel))
        {
            return ActionError("invalid_state", "当前有活动区间时不能切换 capture_mode");
        }

        if (updated.sceneId.empty())
        {
            return ActionError("validation_error", "scene_id 不能为空");
        }
        if (updated.operatorId.empty())
        {
            return ActionError("validation_error", "operator_id 不能为空");
        }
        if (updated.captureMode == Config::CaptureMode::Trajectory && updated.instruction.empty())
        {
            return ActionError("validation_error", "trajectory 模式下 instruction 不能为空");
        }
        if (updated.cmdVxMax <= 0.0 || updated.cmdVyMax <= 0.0 || updated.cmdWzMax <= 0.0)
        {
            return ActionError("validation_error", "速度上限必须为正数");
        }

        {
            std::lock_guard<std::mutex> lock(editableConfigMutex_);
            editableConfig_ = updated;
        }
        return ActionOk("collector 参数已更新，将用于后续采集段");
    }

    UiActionResult SaveEditableConfigAsDefaults()
    {
        try
        {
            SaveDefaultsFile(CollectorDefaultsPath(config_.collectorRoot), GetEditableConfigSnapshot());
            return ActionOk("已保存为下次启动默认值");
        }
        catch (const std::exception& ex)
        {
            return ActionError("save_defaults_failed", ex.what());
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
            webConfig.submitLabelHandler = [this](const SegmentLabelInput& input)
            {
                return SubmitPendingLabel(input);
            };
            webConfig.updateConfigHandler = [this](const UiConfigUpdateInput& input)
            {
                return UpdateEditableConfig(input);
            };
            webConfig.saveDefaultsHandler = [this]()
            {
                return SaveEditableConfigAsDefaults();
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
        PrintLine("collector 已就绪；按 H 查看帮助");
        PrintHelp();
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
        webUiServer_.reset();
    }

    UiStatusSnapshot GetUiStatusSnapshot() const
    {
        LatestState state;
        LatestImage image;
        VelocityCommand command;
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

        UiStatusSnapshot snapshot;
        CaptureState captureState;
        SafetyState safetyState;
        std::string faultReason;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState = captureState_;
            safetyState = safetyState_;
            faultReason = latchedFaultReason_;
        }
        const auto loggerStatus = logger_.GetStatus();
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        const double nowSeconds = NowSeconds();
        snapshot.running = running_.load();
        snapshot.webUiEnabled = config_.webUiEnabled;
        snapshot.webPort = config_.webPort;
        snapshot.sessionDir = logger_.OutputDir().string();
        snapshot.captureMode = CaptureModeName(editable.captureMode);
        snapshot.bufferedFrames = loggerStatus.bufferedFrames;
        snapshot.segmentDurationSeconds = loggerStatus.bufferedFrames >= 2
                                              ? std::max(0.0, loggerStatus.endTimestamp - loggerStatus.startTimestamp)
                                              : 0.0;
        snapshot.stateValid = state.valid;
        snapshot.imageValid = image.valid;
        snapshot.stateAgeSeconds = AgeSeconds(state.timestamp, nowSeconds);
        snapshot.imageAgeSeconds = AgeSeconds(image.timestamp, nowSeconds);
        snapshot.robotConnected = snapshot.stateValid && snapshot.stateAgeSeconds >= 0.0 && snapshot.stateAgeSeconds <= kStateTimeoutSeconds;
        snapshot.bodyHeight = state.bodyHeight;
        snapshot.roll = state.roll;
        snapshot.pitch = state.pitch;
        snapshot.yaw = state.yaw;
        snapshot.commandVx = command.vx;
        snapshot.commandVy = command.vy;
        snapshot.commandWz = command.wz;
        snapshot.sceneId = editable.sceneId;
        snapshot.operatorId = editable.operatorId;
        snapshot.instruction = editable.instruction;
        snapshot.taskFamily = editable.taskFamily;
        snapshot.targetType = editable.targetType;
        snapshot.targetDescription = editable.targetDescription;
        snapshot.targetInstanceId = editable.targetInstanceId;
        snapshot.taskTagsCsv = JoinCsvList(editable.taskTags);
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
        snapshot.safetyState = SafetyStateName(safetyState);
        snapshot.faultReason = faultReason;
        snapshot.recording = captureState == CaptureState::Capturing;
        snapshot.actions.canStartRecording = safetyState == SafetyState::SafeReady &&
                                             captureState == CaptureState::Idle &&
                                             !loggerStatus.pendingLabel;
        snapshot.actions.canStopRecording = captureState == CaptureState::Capturing;
        snapshot.actions.canDiscardSegment = captureState == CaptureState::Armed ||
                                             captureState == CaptureState::DelayBeforeLog ||
                                             captureState == CaptureState::Capturing ||
                                             loggerStatus.pendingLabel;
        snapshot.actions.canSubmitLabel = loggerStatus.pendingLabel;
        snapshot.actions.canEstop = safetyState != SafetyState::EstopLatched;
        snapshot.actions.canClearFault = safetyState != SafetyState::SafeReady;
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

    std::optional<std::string> PromptSegmentStatus()
    {
        return PromptForSelection(
            "请选择 segment_status",
            {
                {"1", SegmentStatusName(SegmentStatus::Clean)},
                {"2", SegmentStatusName(SegmentStatus::Usable)},
                {"3", SegmentStatusName(SegmentStatus::Discard)},
            });
    }

    std::optional<std::string> PromptSuccessLabel()
    {
        return PromptForSelection(
            "请选择 success",
            {
                {"1", SuccessLabelName(SuccessLabel::Success)},
                {"2", SuccessLabelName(SuccessLabel::Partial)},
                {"3", SuccessLabelName(SuccessLabel::Fail)},
            });
    }

    std::optional<std::string> PromptTerminationReason()
    {
        return PromptForSelection(
            "请选择 termination_reason",
            {
                {"1", TerminationReasonName(TerminationReason::GoalReached)},
                {"2", TerminationReasonName(TerminationReason::NearGoalStop)},
                {"3", TerminationReasonName(TerminationReason::Occluded)},
                {"4", TerminationReasonName(TerminationReason::TargetLost)},
                {"5", TerminationReasonName(TerminationReason::OperatorStop)},
                {"6", TerminationReasonName(TerminationReason::BadDemo)},
                {"7", TerminationReasonName(TerminationReason::Unsafe)},
                {"8", TerminationReasonName(TerminationReason::Timeout)},
            });
    }

    void PrintHelp() const
    {
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        std::lock_guard<std::mutex> lock(outputMutex_);
        std::cout << "\r\33[2KKeys:" << std::endl;
        std::cout << "  W/S forward/backward  A/D strafe  Q/E yaw" << std::endl;
        std::cout << "  R start capture flow  ESC cancel current armed/capture segment" << std::endl;
        std::cout << "  Space emergency stop  C clear fault or toggle stand up/down  P status  H help  X quit" << std::endl;
        if (editable.captureMode == Config::CaptureMode::SingleAction)
        {
            std::cout << "  single_action 模式：检测到第一个有效单动作输入后，等待 0.5 秒开始录制，松键自动结束" << std::endl;
            std::cout << "  若设置 --instruction，则使用该语义指令；否则回退为动作标签" << std::endl;
        }
        else
        {
            std::cout << "  trajectory 模式：按 R 立即开始录制，允许多阶段动作变化，按 T 结束后进行标注" << std::endl;
            std::cout << "  该模式下必须提供 --instruction，并作为轨迹级语义标签保存" << std::endl;
        }
        std::cout << "  input_backend=" << InputBackendName(config_.inputBackend)
                  << " capture_mode=" << CaptureModeName(editable.captureMode) << std::endl;
    }

    void PrintStatus() const
    {
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
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
            << " scene_id=" << editable.sceneId
            << " operator_id=" << editable.operatorId
            << " capture_mode=" << CaptureModeName(editable.captureMode)
            << " configured_instruction=" << (editable.instruction.empty() ? "-" : editable.instruction)
            << " task_family=" << (editable.taskFamily.empty() ? "-" : editable.taskFamily)
            << " target_type=" << (editable.targetType.empty() ? "-" : editable.targetType)
            << " target_description=" << (editable.targetDescription.empty() ? "-" : editable.targetDescription)
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
        metadata.targetInstanceId = editable.targetInstanceId;
        metadata.taskTags = editable.taskTags;
        metadata.collectorNotes = editable.collectorNotes;
        metadata.instructionSource = metadata.instruction.empty() ? "motion_label" : "semantic_text";
        return metadata;
    }

    UiActionResult BeginSegmentInternal(bool emitLog)
    {
        const bool hasPendingLabel = logger_.GetStatus().pendingLabel;
        std::lock_guard<std::mutex> lock(stateMachineMutex_);
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
            const EditableCollectorConfig editable = GetEditableConfigSnapshot();
            if (editable.captureMode == Config::CaptureMode::Trajectory)
            {
                logger_.BeginSegment(editable.sceneId, editable.operatorId, ConfiguredTaskMetadata());
                captureState_ = CaptureState::Capturing;
                activeMotionInstruction_ = MotionInstruction::None;
                captureStartDeadline_ = std::chrono::steady_clock::time_point{};
                pendingLabelFallbackInstruction_.clear();
                if (emitLog)
                {
                    PrintLine("trajectory 采集已开始；按 T 结束并标注，按 ESC 丢弃");
                }
            }
            else
            {
                captureState_ = CaptureState::Armed;
                activeMotionInstruction_ = MotionInstruction::None;
                captureStartDeadline_ = std::chrono::steady_clock::time_point{};
                pendingLabelFallbackInstruction_.clear();
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

    UiActionResult StopSegmentForLabelInternal(bool emitLog)
    {
        const bool hasPendingLabel = logger_.GetStatus().pendingLabel;
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

        const size_t frameCount = logger_.EndSegmentForLabel();
        if (frameCount == 0)
        {
            logger_.DiscardPendingSegment();
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState_ = SafetyState::SafeReady == safetyState_ ? CaptureState::Idle : CaptureState::Fault;
            activeMotionInstruction_ = MotionInstruction::None;
            captureStartDeadline_ = std::chrono::steady_clock::time_point{};
            pendingLabelFallbackInstruction_.clear();
            if (emitLog)
            {
                PrintLine("该段未记录到有效帧，已丢弃");
            }
            return ActionError("empty_segment", "该段未记录到有效帧");
        }

        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState_ = SafetyState::SafeReady == safetyState_ ? CaptureState::Idle : CaptureState::Fault;
            activeMotionInstruction_ = MotionInstruction::None;
            captureStartDeadline_ = std::chrono::steady_clock::time_point{};
            pendingLabelFallbackInstruction_ = MotionInstructionName(instruction);
        }
        if (emitLog)
        {
            PrintLine("采集段已结束，等待标注或丢弃");
        }
        return ActionOk("segment stopped for labeling");
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
            captureState_ = SafetyState::SafeReady == safetyState_ ? CaptureState::Idle : CaptureState::Fault;
            activeMotionInstruction_ = MotionInstruction::None;
            captureStartDeadline_ = std::chrono::steady_clock::time_point{};
            pendingLabelFallbackInstruction_.clear();
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
                captureState_ = SafetyState::SafeReady == safetyState_ ? CaptureState::Idle : CaptureState::Fault;
                activeMotionInstruction_ = MotionInstruction::None;
                captureStartDeadline_ = std::chrono::steady_clock::time_point{};
                pendingLabelFallbackInstruction_.clear();
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
        const auto segmentStatus = PromptSegmentStatus();
        if (!segmentStatus.has_value())
        {
            DiscardSegmentInternal("cancelled during labeling", true);
            return;
        }

        SegmentLabelInput input;
        input.segmentStatus = segmentStatus.value();
        if (input.segmentStatus != SegmentStatusName(SegmentStatus::Discard))
        {
            const auto success = PromptSuccessLabel();
            if (!success.has_value())
            {
                DiscardSegmentInternal("cancelled during labeling", true);
                return;
            }
            input.success = success.value();

            const auto terminationReason = PromptTerminationReason();
            if (!terminationReason.has_value())
            {
                DiscardSegmentInternal("cancelled during labeling", true);
                return;
            }
            input.terminationReason = terminationReason.value();
        }

        const UiActionResult result = SubmitPendingLabel(input);
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
            captureState_ = SafetyState::SafeReady == safetyState_ ? CaptureState::Idle : CaptureState::Fault;
            activeMotionInstruction_ = MotionInstruction::None;
            captureStartDeadline_ = std::chrono::steady_clock::time_point{};
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
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        command.vx = smoothedVx_ * static_cast<float>(editable.cmdVxMax);
        command.vy = smoothedVy_ * static_cast<float>(editable.cmdVyMax);
        command.wz = smoothedWz_ * static_cast<float>(editable.cmdWzMax);
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
                RequestEmergencyStop("键盘急停");
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
                PrintLine("发送运动指令失败");
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
                    PrintLine("safety fault 已锁定：" + reason);
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
    mutable std::mutex editableConfigMutex_;
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
    std::string pendingLabelFallbackInstruction_;
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
    int evdevFd_ = -1;
    bool terminalRawEnabled_ = false;

    std::unique_ptr<unitree::robot::go2::SportClient> sportClient_;
    std::unique_ptr<unitree::robot::go2::VideoClient> videoClient_;
    std::shared_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>> sportStateSubscriber_;

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
