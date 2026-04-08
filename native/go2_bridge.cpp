#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/video/video_client.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>

namespace
{

constexpr char kSportStateTopic[] = "rt/sportmodestate";

struct Config
{
    std::string networkInterface;
    double controlHz = 50.0;
    double videoPollHz = 10.0;
    bool enableVideo = true;
};

struct LatestState
{
    double timestamp = 0.0;
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    bool valid = false;
};

struct LatestImage
{
    double timestamp = 0.0;
    std::vector<uint8_t> jpegBytes;
    bool valid = false;
};

struct VelocityCommand
{
    float vx = 0.0f;
    float vy = 0.0f;
    float wz = 0.0f;
};

double NowSeconds()
{
    using clock = std::chrono::system_clock;
    const auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
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

std::string Base64Encode(const std::vector<uint8_t>& bytes)
{
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string output;
    output.reserve(((bytes.size() + 2) / 3) * 4);

    for (size_t index = 0; index < bytes.size(); index += 3)
    {
        const uint32_t octetA = bytes[index];
        const uint32_t octetB = (index + 1 < bytes.size()) ? bytes[index + 1] : 0;
        const uint32_t octetC = (index + 2 < bytes.size()) ? bytes[index + 2] : 0;
        const uint32_t triple = (octetA << 16U) | (octetB << 8U) | octetC;

        output.push_back(kAlphabet[(triple >> 18U) & 0x3FU]);
        output.push_back(kAlphabet[(triple >> 12U) & 0x3FU]);
        output.push_back(index + 1 < bytes.size() ? kAlphabet[(triple >> 6U) & 0x3FU] : '=');
        output.push_back(index + 2 < bytes.size() ? kAlphabet[triple & 0x3FU] : '=');
    }

    return output;
}

std::optional<Config> ParseArgs(int argc, char** argv, std::string& error)
{
    Config config;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--network-interface")
        {
            if (i + 1 >= argc)
            {
                error = "missing value for --network-interface";
                return std::nullopt;
            }
            config.networkInterface = argv[++i];
        }
        else if (arg == "--control-hz")
        {
            if (i + 1 >= argc)
            {
                error = "missing value for --control-hz";
                return std::nullopt;
            }
            config.controlHz = std::stod(argv[++i]);
        }
        else if (arg == "--video-poll-hz")
        {
            if (i + 1 >= argc)
            {
                error = "missing value for --video-poll-hz";
                return std::nullopt;
            }
            config.videoPollHz = std::stod(argv[++i]);
        }
        else if (arg == "--disable-video")
        {
            config.enableVideo = false;
        }
        else
        {
            error = "unknown argument: " + arg;
            return std::nullopt;
        }
    }

    if (config.controlHz <= 0.0)
    {
        error = "--control-hz must be positive";
        return std::nullopt;
    }

    if (config.videoPollHz <= 0.0)
    {
        error = "--video-poll-hz must be positive";
        return std::nullopt;
    }

    return config;
}

class Go2Bridge
{
public:
    explicit Go2Bridge(const Config& config) : config_(config)
    {
    }

    ~Go2Bridge()
    {
        Shutdown();
    }

    void Connect()
    {
        if (connected_)
        {
            return;
        }

        if (config_.networkInterface.empty())
        {
            unitree::robot::ChannelFactory::Instance()->Init(0);
        }
        else
        {
            unitree::robot::ChannelFactory::Instance()->Init(0, config_.networkInterface);
        }

        sportClient_ = std::make_unique<unitree::robot::go2::SportClient>();
        sportClient_->SetTimeout(10.0f);
        sportClient_->Init();

        sportStateSubscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(kSportStateTopic);
        sportStateSubscriber_->InitChannel(std::bind(&Go2Bridge::OnSportState, this, std::placeholders::_1), 1);

        if (config_.enableVideo)
        {
            videoClient_ = std::make_unique<unitree::robot::go2::VideoClient>();
            videoClient_->SetTimeout(1.0f);
            videoClient_->Init();
        }

        running_.store(true);
        controlThread_ = std::thread(&Go2Bridge::ControlLoop, this);

        if (config_.enableVideo)
        {
            videoThread_ = std::thread(&Go2Bridge::VideoLoop, this);
        }

        connected_ = true;
    }

    std::string SetVelocity(float vx, float vy, float wz)
    {
        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            latestCommand_ = {vx, vy, wz};
        }
        return AckJson("set_velocity");
    }

    std::string Stop()
    {
        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            latestCommand_ = {};
        }
        const int32_t ret = sportClient_ ? sportClient_->StopMove() : -1;
        return AckJson("stop", ret);
    }

    std::string StandUp()
    {
        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            latestCommand_ = {};
        }
        const int32_t ret = sportClient_ ? sportClient_->StandUp() : -1;
        return AckJson("stand_up", ret);
    }

    std::string StandDown()
    {
        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            latestCommand_ = {};
        }
        const int32_t ret = sportClient_ ? sportClient_->StandDown() : -1;
        return AckJson("stand_down", ret);
    }

    std::string Snapshot() const
    {
        LatestState stateCopy;
        LatestImage imageCopy;

        {
            std::lock_guard<std::mutex> lock(stateMutex_);
            stateCopy = latestState_;
        }
        {
            std::lock_guard<std::mutex> lock(imageMutex_);
            imageCopy = latestImage_;
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << "{\"ok\":true,\"type\":\"snapshot\",\"host_time\":" << NowSeconds() << ",\"state\":";
        if (stateCopy.valid)
        {
            oss << "{\"timestamp\":" << stateCopy.timestamp
                << ",\"roll\":" << stateCopy.roll
                << ",\"pitch\":" << stateCopy.pitch
                << ",\"yaw\":" << stateCopy.yaw << "}";
        }
        else
        {
            oss << "null";
        }

        oss << ",\"image\":";
        if (imageCopy.valid)
        {
            oss << "{\"timestamp\":" << imageCopy.timestamp
                << ",\"encoding\":\"jpeg\",\"jpeg_b64\":" << JsonString(Base64Encode(imageCopy.jpegBytes)) << "}";
        }
        else
        {
            oss << "null";
        }
        oss << "}";
        return oss.str();
    }

    void Shutdown()
    {
        if (!connected_)
        {
            return;
        }

        running_.store(false);

        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            latestCommand_ = {};
        }

        try
        {
            if (sportClient_)
            {
                sportClient_->StopMove();
            }
        }
        catch (...)
        {
            std::cerr << "bridge: StopMove failed during shutdown" << std::endl;
        }

        if (controlThread_.joinable())
        {
            controlThread_.join();
        }

        if (videoThread_.joinable())
        {
            videoThread_.join();
        }

        if (sportStateSubscriber_)
        {
            sportStateSubscriber_->CloseChannel();
            sportStateSubscriber_.reset();
        }

        unitree::robot::ChannelFactory::Instance()->Release();
        connected_ = false;
    }

    std::string ReadyJson() const
    {
        std::ostringstream oss;
        oss << "{\"ok\":true,\"type\":\"ready\",\"state_topic\":" << JsonString(kSportStateTopic)
            << ",\"video_enabled\":" << (config_.enableVideo ? "true" : "false") << "}";
        return oss.str();
    }

private:
    void OnSportState(const void* message)
    {
        const auto* state = static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
        LatestState latest;
        latest.timestamp = NowSeconds();
        latest.roll = state->imu_state().rpy()[0];
        latest.pitch = state->imu_state().rpy()[1];
        latest.yaw = state->imu_state().rpy()[2];
        latest.valid = true;

        std::lock_guard<std::mutex> lock(stateMutex_);
        latestState_ = latest;
    }

    void ControlLoop()
    {
        const auto period = std::chrono::duration<double>(1.0 / config_.controlHz);
        bool stopped = false;
        while (running_.load())
        {
            const auto cycleStart = std::chrono::steady_clock::now();

            VelocityCommand command;
            {
                std::lock_guard<std::mutex> lock(commandMutex_);
                command = latestCommand_;
            }

            try
            {
                if (std::fabs(command.vx) < 1e-4F && std::fabs(command.vy) < 1e-4F && std::fabs(command.wz) < 1e-4F)
                {
                    if (!stopped)
                    {
                        if (sportClient_)
                        {
                            sportClient_->StopMove();
                        }
                        stopped = true;
                    }
                }
                else
                {
                    if (sportClient_)
                    {
                        sportClient_->Move(command.vx, command.vy, command.wz);
                    }
                    stopped = false;
                }
            }
            catch (...)
            {
                std::cerr << "bridge: control command failed" << std::endl;
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
                image.jpegBytes = std::move(jpegBytes);
                image.valid = true;

                std::lock_guard<std::mutex> lock(imageMutex_);
                latestImage_ = std::move(image);
            }

            std::this_thread::sleep_until(cycleStart + period);
        }
    }

    std::string AckJson(const std::string& command, int32_t sdkRet = 0) const
    {
        std::ostringstream oss;
        oss << "{\"ok\":true,\"type\":\"ack\",\"command\":" << JsonString(command) << ",\"sdk_ret\":" << sdkRet << "}";
        return oss.str();
    }

    Config config_;
    mutable std::mutex stateMutex_;
    mutable std::mutex imageMutex_;
    mutable std::mutex commandMutex_;
    LatestState latestState_;
    LatestImage latestImage_;
    VelocityCommand latestCommand_;
    std::atomic<bool> running_{false};
    bool connected_ = false;
    std::unique_ptr<unitree::robot::go2::SportClient> sportClient_;
    std::unique_ptr<unitree::robot::go2::VideoClient> videoClient_;
    std::shared_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>> sportStateSubscriber_;
    std::thread controlThread_;
    std::thread videoThread_;
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
            std::cerr << "bridge argument error: " << error << std::endl;
            return 2;
        }

        Go2Bridge bridge(config.value());
        bridge.Connect();
        std::cout << bridge.ReadyJson() << std::endl;

        std::string line;
        while (std::getline(std::cin, line))
        {
            if (line.empty())
            {
                continue;
            }

            std::istringstream iss(line);
            std::string command;
            iss >> command;

            if (command == "SET_VELOCITY")
            {
                float vx = 0.0f;
                float vy = 0.0f;
                float wz = 0.0f;
                iss >> vx >> vy >> wz;
                std::cout << bridge.SetVelocity(vx, vy, wz) << std::endl;
            }
            else if (command == "STOP")
            {
                std::cout << bridge.Stop() << std::endl;
            }
            else if (command == "STAND_UP")
            {
                std::cout << bridge.StandUp() << std::endl;
            }
            else if (command == "STAND_DOWN")
            {
                std::cout << bridge.StandDown() << std::endl;
            }
            else if (command == "SNAPSHOT")
            {
                std::cout << bridge.Snapshot() << std::endl;
            }
            else if (command == "SHUTDOWN")
            {
                std::cout << "{\"ok\":true,\"type\":\"ack\",\"command\":\"shutdown\",\"sdk_ret\":0}" << std::endl;
                bridge.Shutdown();
                return 0;
            }
            else
            {
                std::cout << "{\"ok\":false,\"error\":\"unknown_command\"}" << std::endl;
            }
        }

        bridge.Shutdown();
        return 0;
    }
    catch (const std::exception& exc)
    {
        std::cerr << "bridge fatal error: " << exc.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "bridge fatal error: unknown exception" << std::endl;
        return 1;
    }
}
