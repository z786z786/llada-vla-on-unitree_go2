#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/video/video_client.hpp>

namespace
{

struct Config
{
    std::string networkInterface;
    double durationSeconds = 20.0;
    double pollHz = 20.0;
    double timeoutSeconds = 1.0;
    size_t hashBytes = 4096;
};

void PrintUsage(const char* program)
{
    std::cout
        << "Usage: " << program << " --network-interface IFACE [options]\n"
        << "Options:\n"
        << "  --duration SEC          Probe duration in seconds (default: 20.0)\n"
        << "  --poll-hz FLOAT         Poll frequency in Hz (default: 20.0)\n"
        << "  --timeout SEC           SDK request timeout in seconds (default: 1.0)\n"
        << "  --hash-bytes N          Bytes used for lightweight frame hash (default: 4096)\n"
        << "  --help                  Show this help\n";
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
        if (arg == "--poll-hz")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --poll-hz";
                return std::nullopt;
            }
            config.pollHz = std::stod(argv[++index]);
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
        if (arg == "--hash-bytes")
        {
            if (index + 1 >= argc)
            {
                error = "missing value for --hash-bytes";
                return std::nullopt;
            }
            config.hashBytes = static_cast<size_t>(std::stoull(argv[++index]));
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
    if (config.durationSeconds <= 0.0)
    {
        error = "--duration must be positive";
        return std::nullopt;
    }
    if (config.pollHz <= 0.0)
    {
        error = "--poll-hz must be positive";
        return std::nullopt;
    }
    if (config.timeoutSeconds <= 0.0)
    {
        error = "--timeout must be positive";
        return std::nullopt;
    }

    return config;
}

uint64_t Fnv1a64(const std::vector<uint8_t>& bytes, size_t limit)
{
    constexpr uint64_t kOffset = 1469598103934665603ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;

    uint64_t hash = kOffset;
    const size_t bytesToHash = std::min(limit, bytes.size());
    for (size_t index = 0; index < bytesToHash; ++index)
    {
        hash ^= static_cast<uint64_t>(bytes[index]);
        hash *= kPrime;
    }
    return hash;
}

double ToMilliseconds(const std::chrono::steady_clock::duration& duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}

double ToSeconds(const std::chrono::steady_clock::time_point& timePoint)
{
    return std::chrono::duration<double>(timePoint.time_since_epoch()).count();
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string error;
        const std::optional<Config> config = ParseArgs(argc, argv, error);
        if (!config.has_value())
        {
            if (!error.empty())
            {
                std::cerr << "go2_video_probe argument error: " << error << std::endl;
                return 2;
            }
            return 0;
        }

        unitree::robot::ChannelFactory::Instance()->Init(0, config->networkInterface);
        unitree::robot::go2::VideoClient videoClient;
        videoClient.SetTimeout(static_cast<float>(config->timeoutSeconds));
        videoClient.Init();

        std::cout << "mono_s ret latency_ms jpeg_size frame_hash duplicate_run success_gap_ms new_frame_gap_ms" << std::endl;
        std::cout << std::fixed << std::setprecision(3);

        const auto period = std::chrono::duration<double>(1.0 / config->pollHz);
        const auto startMono = std::chrono::steady_clock::now();
        auto nextTick = startMono;

        bool haveSuccess = false;
        bool haveNewFrame = false;
        bool havePreviousHash = false;
        auto lastSuccessMono = startMono;
        auto lastNewFrameMono = startMono;
        uint64_t previousHash = 0;
        int duplicateRun = 0;
        size_t totalPolls = 0;
        size_t successCount = 0;
        size_t duplicateSuccessCount = 0;
        size_t newFrameCount = 0;
        double maxLatencyMs = 0.0;
        double maxSuccessGapMs = 0.0;
        double maxNewFrameGapMs = 0.0;
        int currentFailureStreak = 0;
        int maxFailureStreak = 0;
        std::map<int32_t, size_t> retCounts;

        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - startMono).count() < config->durationSeconds)
        {
            nextTick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);

            std::vector<uint8_t> jpegBytes;
            const auto callStart = std::chrono::steady_clock::now();
            int32_t ret = -1;
            try
            {
                ret = videoClient.GetImageSample(jpegBytes);
            }
            catch (...)
            {
                ret = -1;
            }
            const auto callEnd = std::chrono::steady_clock::now();
            ++totalPolls;
            ++retCounts[ret];

            double successGapMs = haveSuccess ? ToMilliseconds(callEnd - lastSuccessMono) : -1.0;
            double newFrameGapMs = haveNewFrame ? ToMilliseconds(callEnd - lastNewFrameMono) : -1.0;
            uint64_t frameHash = 0;
            const double latencyMs = ToMilliseconds(callEnd - callStart);
            maxLatencyMs = std::max(maxLatencyMs, latencyMs);

            if (ret == 0 && !jpegBytes.empty())
            {
                frameHash = Fnv1a64(jpegBytes, config->hashBytes);
                const bool duplicateFrame = havePreviousHash && frameHash == previousHash;
                duplicateRun = duplicateFrame ? (duplicateRun + 1) : 0;
                ++successCount;
                currentFailureStreak = 0;

                lastSuccessMono = callEnd;
                haveSuccess = true;
                successGapMs = 0.0;

                if (!duplicateFrame)
                {
                    lastNewFrameMono = callEnd;
                    haveNewFrame = true;
                    newFrameGapMs = 0.0;
                    ++newFrameCount;
                }
                else
                {
                    ++duplicateSuccessCount;
                }

                previousHash = frameHash;
                havePreviousHash = true;
            }
            else
            {
                ++currentFailureStreak;
                maxFailureStreak = std::max(maxFailureStreak, currentFailureStreak);
                if (successGapMs > 0.0)
                {
                    maxSuccessGapMs = std::max(maxSuccessGapMs, successGapMs);
                }
                if (newFrameGapMs > 0.0)
                {
                    maxNewFrameGapMs = std::max(maxNewFrameGapMs, newFrameGapMs);
                }
            }

            if (successGapMs > 0.0)
            {
                maxSuccessGapMs = std::max(maxSuccessGapMs, successGapMs);
            }
            if (newFrameGapMs > 0.0)
            {
                maxNewFrameGapMs = std::max(maxNewFrameGapMs, newFrameGapMs);
            }

            std::cout << ToSeconds(callEnd)
                      << " " << ret
                      << " " << latencyMs
                      << " " << jpegBytes.size()
                      << " 0x" << std::hex << frameHash << std::dec
                      << " " << duplicateRun
                      << " " << successGapMs
                      << " " << newFrameGapMs
                      << std::endl;

            std::this_thread::sleep_until(nextTick);
        }

        std::cout << "# summary total_polls=" << totalPolls
                  << " success=" << successCount
                  << " new_frames=" << newFrameCount
                  << " duplicate_success=" << duplicateSuccessCount
                  << " max_latency_ms=" << maxLatencyMs
                  << " max_success_gap_ms=" << maxSuccessGapMs
                  << " max_new_frame_gap_ms=" << maxNewFrameGapMs
                  << " max_failure_streak=" << maxFailureStreak
                  << std::endl;
        for (const auto& item : retCounts)
        {
            std::cout << "# ret_count code=" << item.first << " count=" << item.second << std::endl;
        }

        unitree::robot::ChannelFactory::Instance()->Release();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "go2_video_probe fatal error: " << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "go2_video_probe fatal error: unknown exception" << std::endl;
    }

    return 1;
}
