#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

namespace
{

constexpr char kDefaultSportStateTopic[] = "rt/sportmodestate";

struct Config
{
    std::string networkInterface;
    std::string topic = kDefaultSportStateTopic;
    double durationSeconds = 0.0;
    double printHz = 10.0;
    double peakWindowSeconds = 3.0;
    double timeoutSeconds = 10.0;
    int speedLevel = -1;
    bool enableNativeJoystick = false;
    bool disableNativeJoystick = false;
};

struct Sample
{
    std::chrono::steady_clock::time_point monoTime;
    float vx = 0.0f;
    float vy = 0.0f;
    float wz = 0.0f;
};

struct StateSnapshot
{
    bool hasState = false;
    uint8_t mode = 0;
    uint8_t gaitType = 0;
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    float yawSpeed = 0.0f;
    std::chrono::steady_clock::time_point monoTime{};
};

std::atomic<bool> gStopRequested{false};

void HandleSignal(int)
{
    gStopRequested.store(true);
}

void PrintUsage(const char* program)
{
    std::cout
        << "Usage: " << program << " --network-interface IFACE [options]\n"
        << "Options:\n"
        << "  --speed-level N            Call SportClient::SpeedLevel(N) before monitoring\n"
        << "  --topic NAME               SportModeState topic (default: " << kDefaultSportStateTopic << ")\n"
        << "  --duration SEC             Exit after SEC seconds; 0 means until Ctrl+C (default: 0)\n"
        << "  --print-hz FLOAT           Status print rate in Hz (default: 10)\n"
        << "  --peak-window SEC          Rolling peak window in seconds (default: 3)\n"
        << "  --timeout SEC              SDK client timeout in seconds (default: 10)\n"
        << "  --enable-native-joystick   Call SwitchJoystick(true) before monitoring\n"
        << "  --disable-native-joystick  Call SwitchJoystick(false) before monitoring\n"
        << "  --help                     Show this help\n\n"
        << "Suggested workflow:\n"
        << "  1. Run once with --speed-level 1 --enable-native-joystick\n"
        << "  2. Push the joystick fully forward/sideways/turn for about 2 seconds each\n"
        << "  3. Note peak_vx / peak_vy / peak_wz\n"
        << "  4. Repeat with --speed-level 2 and --speed-level 3\n";
}

std::optional<Config> ParseArgs(int argc, char** argv, std::string& error)
{
    Config config;
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h")
        {
            PrintUsage(argv[0]);
            return std::nullopt;
        }
        if (arg == "--network-interface")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --network-interface";
                return std::nullopt;
            }
            config.networkInterface = argv[++index];
            continue;
        }
        if (arg == "--speed-level")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --speed-level";
                return std::nullopt;
            }
            config.speedLevel = std::stoi(argv[++index]);
            continue;
        }
        if (arg == "--topic")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --topic";
                return std::nullopt;
            }
            config.topic = argv[++index];
            continue;
        }
        if (arg == "--duration")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --duration";
                return std::nullopt;
            }
            config.durationSeconds = std::stod(argv[++index]);
            continue;
        }
        if (arg == "--print-hz")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --print-hz";
                return std::nullopt;
            }
            config.printHz = std::stod(argv[++index]);
            continue;
        }
        if (arg == "--peak-window")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --peak-window";
                return std::nullopt;
            }
            config.peakWindowSeconds = std::stod(argv[++index]);
            continue;
        }
        if (arg == "--timeout")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --timeout";
                return std::nullopt;
            }
            config.timeoutSeconds = std::stod(argv[++index]);
            continue;
        }
        if (arg == "--enable-native-joystick")
        {
            config.enableNativeJoystick = true;
            continue;
        }
        if (arg == "--disable-native-joystick")
        {
            config.disableNativeJoystick = true;
            continue;
        }

        error = "unknown argument: " + arg;
        return std::nullopt;
    }

    if (config.networkInterface.empty())
    {
        error = "--network-interface is required";
        return std::nullopt;
    }
    if (config.durationSeconds < 0.0)
    {
        error = "--duration must be >= 0";
        return std::nullopt;
    }
    if (config.printHz <= 0.0)
    {
        error = "--print-hz must be > 0";
        return std::nullopt;
    }
    if (config.peakWindowSeconds <= 0.0)
    {
        error = "--peak-window must be > 0";
        return std::nullopt;
    }
    if (config.timeoutSeconds <= 0.0)
    {
        error = "--timeout must be > 0";
        return std::nullopt;
    }
    if (config.enableNativeJoystick && config.disableNativeJoystick)
    {
        error = "cannot use --enable-native-joystick and --disable-native-joystick together";
        return std::nullopt;
    }

    return config;
}

double ToSeconds(const std::chrono::steady_clock::duration& duration)
{
    return std::chrono::duration<double>(duration).count();
}

double ToMilliseconds(const std::chrono::steady_clock::duration& duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::signal(SIGINT, HandleSignal);
        std::signal(SIGTERM, HandleSignal);

        std::string error;
        const std::optional<Config> config = ParseArgs(argc, argv, error);
        if (!config.has_value())
        {
            if (!error.empty())
            {
                std::cerr << "go2_speed_level_probe argument error: " << error << std::endl;
                return 2;
            }
            return 0;
        }

        unitree::robot::ChannelFactory::Instance()->Init(0, config->networkInterface);

        unitree::robot::go2::SportClient sportClient;
        sportClient.SetTimeout(static_cast<float>(config->timeoutSeconds));
        sportClient.Init();

        if (config->enableNativeJoystick || config->disableNativeJoystick)
        {
            const bool enabled = config->enableNativeJoystick;
            int32_t ret = -1;
            try
            {
                ret = sportClient.SwitchJoystick(enabled);
            }
            catch (...)
            {
                ret = -1;
            }
            std::cout << "switch_joystick=" << (enabled ? "true" : "false") << " ret=" << ret << std::endl;
        }

        if (config->speedLevel >= 0)
        {
            int32_t ret = -1;
            try
            {
                ret = sportClient.SpeedLevel(config->speedLevel);
            }
            catch (...)
            {
                ret = -1;
            }
            std::cout << "speed_level=" << config->speedLevel << " ret=" << ret << std::endl;
        }

        std::mutex stateMutex;
        StateSnapshot latestState;
        std::deque<Sample> recentSamples;

        unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_> subscriber(config->topic);
        subscriber.InitChannel(
            [&](const void* message)
            {
                const auto* state = static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
                const auto now = std::chrono::steady_clock::now();

                StateSnapshot snapshot;
                snapshot.hasState = true;
                snapshot.mode = state->mode();
                snapshot.gaitType = state->gait_type();
                snapshot.vx = state->velocity()[0];
                snapshot.vy = state->velocity()[1];
                snapshot.vz = state->velocity()[2];
                snapshot.yawSpeed = state->yaw_speed();
                snapshot.monoTime = now;

                std::lock_guard<std::mutex> lock(stateMutex);
                latestState = snapshot;
                recentSamples.push_back(Sample{now, snapshot.vx, snapshot.vy, snapshot.yawSpeed});
            },
            1);

        std::cout << "topic=" << config->topic
                  << " print_hz=" << config->printHz
                  << " peak_window_s=" << config->peakWindowSeconds
                  << " duration_s=" << config->durationSeconds << std::endl;
        std::cout << "mono_s age_ms mode gait vx vy vz wz peak_vx peak_vy peak_wz peak_planar samples" << std::endl;
        std::cout << std::fixed << std::setprecision(3);

        const auto start = std::chrono::steady_clock::now();
        const auto period = std::chrono::duration<double>(1.0 / config->printHz);
        auto nextTick = start;

        while (!gStopRequested.load())
        {
            if (config->durationSeconds > 0.0 &&
                ToSeconds(std::chrono::steady_clock::now() - start) >= config->durationSeconds)
            {
                break;
            }

            nextTick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);

            StateSnapshot snapshot;
            double sampleAgeMs = -1.0;
            double peakVx = 0.0;
            double peakVy = 0.0;
            double peakWz = 0.0;
            size_t sampleCount = 0;
            {
                std::lock_guard<std::mutex> lock(stateMutex);
                const auto pruneBefore =
                    std::chrono::steady_clock::now() - std::chrono::duration<double>(config->peakWindowSeconds);
                while (!recentSamples.empty() && recentSamples.front().monoTime < pruneBefore)
                {
                    recentSamples.pop_front();
                }

                snapshot = latestState;
                if (snapshot.hasState)
                {
                    sampleAgeMs = ToMilliseconds(std::chrono::steady_clock::now() - snapshot.monoTime);
                }
                for (const Sample& sample : recentSamples)
                {
                    peakVx = std::max(peakVx, static_cast<double>(std::abs(sample.vx)));
                    peakVy = std::max(peakVy, static_cast<double>(std::abs(sample.vy)));
                    peakWz = std::max(peakWz, static_cast<double>(std::abs(sample.wz)));
                }
                sampleCount = recentSamples.size();
            }

            const double monoSeconds = ToSeconds(std::chrono::steady_clock::now() - start);
            if (!snapshot.hasState)
            {
                std::cout << monoSeconds << " -1 no_state no_state 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0"
                          << std::endl;
            }
            else
            {
                const double peakPlanar = std::sqrt(peakVx * peakVx + peakVy * peakVy);
                std::cout << monoSeconds
                          << ' ' << sampleAgeMs
                          << ' ' << static_cast<int>(snapshot.mode)
                          << ' ' << static_cast<int>(snapshot.gaitType)
                          << ' ' << snapshot.vx
                          << ' ' << snapshot.vy
                          << ' ' << snapshot.vz
                          << ' ' << snapshot.yawSpeed
                          << ' ' << peakVx
                          << ' ' << peakVy
                          << ' ' << peakWz
                          << ' ' << peakPlanar
                          << ' ' << sampleCount
                          << std::endl;
            }

            std::this_thread::sleep_until(nextTick);
        }

        unitree::robot::ChannelFactory::Instance()->Release();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "go2_speed_level_probe fatal error: " << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "go2_speed_level_probe fatal error: unknown exception" << std::endl;
    }

    return 1;
}
