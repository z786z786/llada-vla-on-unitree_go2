#include "web_ui_server.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <netinet/in.h>
#include <sys/socket.h>

namespace
{

namespace fs = std::filesystem;

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

std::string BoolJson(bool value)
{
    return value ? "true" : "false";
}

std::string ReadFileToString(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open())
    {
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

bool SendAll(int fd, const std::string& data)
{
    size_t sent = 0;
    while (sent < data.size())
    {
        const ssize_t result = ::send(fd, data.data() + sent, data.size() - sent, 0);
        if (result <= 0)
        {
            return false;
        }
        sent += static_cast<size_t>(result);
    }
    return true;
}

bool SendAllBytes(int fd, const uint8_t* data, size_t size)
{
    size_t sent = 0;
    while (sent < size)
    {
        const ssize_t result = ::send(fd, data + sent, size - sent, 0);
        if (result <= 0)
        {
            return false;
        }
        sent += static_cast<size_t>(result);
    }
    return true;
}

bool SendAll(int fd, const std::vector<uint8_t>& data)
{
    return SendAllBytes(fd, data.data(), data.size());
}

std::string MakeJsonActionResult(const UiActionResult& result)
{
    std::ostringstream oss;
    oss << "{"
        << "\"ok\":" << BoolJson(result.ok) << ","
        << "\"code\":" << JsonString(result.code) << ","
        << "\"message\":" << JsonString(result.message);
    if (!result.episodeId.empty())
    {
        oss << ",\"episode_id\":" << JsonString(result.episodeId);
    }
    oss << "}";
    return oss.str();
}

std::string MakeJsonStatus(const UiStatusSnapshot& status)
{
    std::ostringstream oss;
    oss << "{"
        << "\"collector\":{"
        << "\"running\":" << BoolJson(status.running) << ","
        << "\"session_dir\":" << JsonString(status.sessionDir) << ","
        << "\"capture_mode\":" << JsonString(status.captureMode) << ","
        << "\"capture_state\":" << JsonString(status.captureState) << ","
        << "\"stop_phase\":" << JsonString(status.stopPhase) << ","
        << "\"safety_state\":" << JsonString(status.safetyState) << ","
        << "\"fault_reason\":" << JsonString(status.faultReason) << ","
        << "\"recording\":" << BoolJson(status.recording) << ","
        << "\"segment_duration_s\":" << status.segmentDurationSeconds << ","
        << "\"buffered_frames\":" << status.bufferedFrames
        << "},"
        << "\"robot\":{"
        << "\"connected\":" << BoolJson(status.robotConnected) << ","
        << "\"state_valid\":" << BoolJson(status.stateValid) << ","
        << "\"image_valid\":" << BoolJson(status.imageValid) << ","
        << "\"state_age_s\":" << status.stateAgeSeconds << ","
        << "\"image_age_s\":" << status.imageAgeSeconds << ","
        << "\"body_height\":" << status.bodyHeight << ","
        << "\"roll\":" << status.roll << ","
        << "\"pitch\":" << status.pitch << ","
        << "\"yaw\":" << status.yaw
        << "},"
        << "\"command\":{"
        << "\"vx\":" << status.commandVx << ","
        << "\"vy\":" << status.commandVy << ","
        << "\"wz\":" << status.commandWz
        << "},"
        << "\"run_context\":{"
        << "\"scene_id\":" << JsonString(status.sceneId) << ","
        << "\"operator_id\":" << JsonString(status.operatorId) << ","
        << "\"instruction\":" << JsonString(status.instruction) << ","
        << "\"capture_mode\":" << JsonString(status.captureMode) << ","
        << "\"task_family\":" << JsonString(status.taskFamily) << ","
        << "\"target_type\":" << JsonString(status.targetType) << ","
        << "\"target_description\":" << JsonString(status.targetDescription) << ","
        << "\"collector_notes\":" << JsonString(status.collectorNotes) << ","
        << "\"cmd_vx_max\":" << status.cmdVxMax << ","
        << "\"cmd_vy_max\":" << status.cmdVyMax << ","
        << "\"cmd_wz_max\":" << status.cmdWzMax
        << "},"
        << "\"process_config\":{"
        << "\"network_interface\":" << JsonString(status.networkInterface) << ","
        << "\"output_dir\":" << JsonString(status.outputDir) << ","
        << "\"loop_hz\":" << status.loopHz << ","
        << "\"video_poll_hz\":" << status.videoPollHz << ","
        << "\"preview_mode\":" << BoolJson(status.previewMode) << ","
        << "\"input_backend\":" << JsonString(status.inputBackend) << ","
        << "\"input_device\":" << JsonString(status.inputDevice) << ","
        << "\"defaults_path\":" << JsonString(status.defaultsPath)
        << "},"
        << "\"pending_label\":{"
        << "\"active\":" << BoolJson(status.pendingLabelActive) << ","
        << "\"episode_id\":" << JsonString(status.pendingEpisodeId) << ","
        << "\"buffered_frames\":" << status.pendingLabelBufferedFrames
        << "},"
        << "\"actions\":{"
        << "\"can_start_recording\":" << BoolJson(status.actions.canStartRecording) << ","
        << "\"can_stop_recording\":" << BoolJson(status.actions.canStopRecording) << ","
        << "\"can_discard_segment\":" << BoolJson(status.actions.canDiscardSegment) << ","
        << "\"can_submit_label\":" << BoolJson(status.actions.canSubmitLabel) << ","
        << "\"can_estop\":" << BoolJson(status.actions.canEstop) << ","
        << "\"can_clear_fault\":" << BoolJson(status.actions.canClearFault)
        << "},"
        << "\"startup_gate\":{"
        << "\"active\":" << BoolJson(status.startupGateActive) << ","
        << "\"prompt\":" << JsonString(status.startupPrompt)
        << "},"
        << "\"web_ui\":{"
        << "\"enabled\":" << BoolJson(status.webUiEnabled) << ","
        << "\"port\":" << status.webPort
        << "}"
        << "}";
    return oss.str();
}

std::optional<std::string> ExtractJsonStringField(const std::string& body, const std::string& key)
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
    size_t valueBegin = body.find('"', colonPos + 1);
    if (valueBegin == std::string::npos)
    {
        return std::nullopt;
    }
    ++valueBegin;
    std::string value;
    bool escaping = false;
    for (size_t index = valueBegin; index < body.size(); ++index)
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

struct HttpRequest
{
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
    std::string body;
};

std::optional<HttpRequest> ReadHttpRequest(int fd)
{
    std::string data;
    char buffer[4096];
    size_t headerEnd = std::string::npos;
    size_t contentLength = 0;

    while (true)
    {
        const ssize_t bytesRead = ::recv(fd, buffer, sizeof(buffer), 0);
        if (bytesRead <= 0)
        {
            return std::nullopt;
        }
        data.append(buffer, buffer + bytesRead);
        headerEnd = data.find("\r\n\r\n");
        if (headerEnd != std::string::npos)
        {
            break;
        }
        if (data.size() > 64 * 1024)
        {
            return std::nullopt;
        }
    }

    HttpRequest request;
    {
        std::istringstream head(data.substr(0, headerEnd));
        std::string requestLine;
        if (!std::getline(head, requestLine))
        {
            return std::nullopt;
        }
        if (!requestLine.empty() && requestLine.back() == '\r')
        {
            requestLine.pop_back();
        }
        std::istringstream requestLineStream(requestLine);
        requestLineStream >> request.method >> request.path;
        std::string headerLine;
        while (std::getline(head, headerLine))
        {
            if (!headerLine.empty() && headerLine.back() == '\r')
            {
                headerLine.pop_back();
            }
            const size_t colonPos = headerLine.find(':');
            if (colonPos == std::string::npos)
            {
                continue;
            }
            std::string key = headerLine.substr(0, colonPos);
            std::string value = headerLine.substr(colonPos + 1);
            while (!value.empty() && value.front() == ' ')
            {
                value.erase(value.begin());
            }
            request.headers[key] = value;
            if (key == "Content-Length")
            {
                try
                {
                    contentLength = static_cast<size_t>(std::stoul(value));
                }
                catch (...)
                {
                    return std::nullopt;
                }
            }
        }
    }

    request.body = data.substr(headerEnd + 4);
    while (request.body.size() < contentLength)
    {
        const ssize_t bytesRead = ::recv(fd, buffer, sizeof(buffer), 0);
        if (bytesRead <= 0)
        {
            return std::nullopt;
        }
        request.body.append(buffer, buffer + bytesRead);
    }
    if (request.body.size() > contentLength)
    {
        request.body.resize(contentLength);
    }
    return request;
}

std::string BuildHttpResponse(
    int statusCode,
    const std::string& statusText,
    const std::string& contentType,
    const std::string& body)
{
    std::ostringstream oss;
    oss << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n"
        << "Content-Type: " << contentType << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Cache-Control: no-store\r\n"
        << "Connection: close\r\n"
        << "Access-Control-Allow-Origin: *\r\n\r\n"
        << body;
    return oss.str();
}

std::string BuildBinaryHttpResponse(
    int statusCode,
    const std::string& statusText,
    const std::string& contentType,
    const std::vector<uint8_t>& body)
{
    std::ostringstream oss;
    oss << "HTTP/1.1 " << statusCode << " " << statusText << "\r\n"
        << "Content-Type: " << contentType << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Cache-Control: no-store\r\n"
        << "Connection: close\r\n"
        << "Access-Control-Allow-Origin: *\r\n\r\n";
    std::string response = oss.str();
    response.append(reinterpret_cast<const char*>(body.data()), body.size());
    return response;
}

} // namespace

struct WebUiServer::Impl
{
    struct ClientThreadEntry
    {
        int clientFd = -1;
        std::shared_ptr<std::atomic<bool>> done;
        std::thread thread;
    };

    explicit Impl(const WebUiServerConfig& cfg)
        : config(cfg)
    {
    }

    ~Impl()
    {
        Stop();
    }

    void Start()
    {
        if (running)
        {
            return;
        }

        serverFd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (serverFd < 0)
        {
            throw std::runtime_error("创建 Web UI socket 失败");
        }

        int reuse = 1;
        ::setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(config.port));
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(serverFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
        {
            const std::string reason = std::strerror(errno);
            ::close(serverFd);
            serverFd = -1;
            throw std::runtime_error("绑定 Web UI 端口失败：" + reason);
        }
        if (::listen(serverFd, 16) != 0)
        {
            const std::string reason = std::strerror(errno);
            ::close(serverFd);
            serverFd = -1;
            throw std::runtime_error("启动 Web UI 监听失败：" + reason);
        }

        running = true;
        thread = std::thread([this]() { ServeLoop(); });
    }

    void Stop()
    {
        if (!running.exchange(false))
        {
            return;
        }
        if (serverFd >= 0)
        {
            ::shutdown(serverFd, SHUT_RDWR);
            ::close(serverFd);
            serverFd = -1;
        }
        ShutdownClientConnections();
        if (thread.joinable())
        {
            thread.join();
        }
        ReapFinishedClientThreads(true);
    }

    void ServeLoop()
    {
        while (running)
        {
            ReapFinishedClientThreads(false);
            pollfd pfd{serverFd, POLLIN, 0};
            const int pollResult = ::poll(&pfd, 1, 100);
            if (pollResult <= 0)
            {
                continue;
            }

            sockaddr_in clientAddr{};
            socklen_t clientLen = sizeof(clientAddr);
            const int clientFd = ::accept(serverFd, reinterpret_cast<sockaddr*>(&clientAddr), &clientLen);
            if (clientFd < 0)
            {
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(clientMutex);
                clientFds.push_back(clientFd);
            }
            std::lock_guard<std::mutex> lock(clientThreadMutex);
            ClientThreadEntry entry;
            entry.clientFd = clientFd;
            entry.done = std::make_shared<std::atomic<bool>>(false);
            entry.thread = std::thread(&Impl::HandleClientLoop, this, clientFd, entry.done);
            clientThreads.emplace_back(std::move(entry));
        }
    }

    void HandleClientLoop(int clientFd, const std::shared_ptr<std::atomic<bool>>& done)
    {
        try
        {
            HandleConnection(clientFd);
        }
        catch (...)
        {
            SendAll(clientFd, BuildHttpResponse(500, "Internal Server Error", "text/plain; charset=utf-8", "internal server error"));
        }

        {
            std::lock_guard<std::mutex> lock(clientMutex);
            clientFds.erase(std::remove(clientFds.begin(), clientFds.end(), clientFd), clientFds.end());
        }
        ::shutdown(clientFd, SHUT_RDWR);
        ::close(clientFd);
        done->store(true);
    }

    void HandleConnection(int clientFd)
    {
        const auto request = ReadHttpRequest(clientFd);
        if (!request.has_value())
        {
            SendAll(clientFd, BuildHttpResponse(400, "Bad Request", "text/plain; charset=utf-8", "bad request"));
            return;
        }

        if (request->method == "GET")
        {
            HandleGet(clientFd, request.value());
            return;
        }
        if (request->method == "POST")
        {
            HandlePost(clientFd, request.value());
            return;
        }
        SendAll(clientFd, BuildHttpResponse(405, "Method Not Allowed", "text/plain; charset=utf-8", "method not allowed"));
    }

    void HandleGet(int clientFd, const HttpRequest& request)
    {
        const std::string path = request.path.substr(0, request.path.find('?'));
        if (path == "/api/status")
        {
            const std::string body = MakeJsonStatus(config.statusProvider());
            SendAll(clientFd, BuildHttpResponse(200, "OK", "application/json; charset=utf-8", body));
            return;
        }
        if (path == "/api/image/latest.jpg")
        {
            const std::vector<uint8_t> jpegBytes = config.latestImageJpegProvider ? config.latestImageJpegProvider() : std::vector<uint8_t>{};
            if (jpegBytes.empty())
            {
                SendAll(clientFd, BuildHttpResponse(404, "Not Found", "text/plain; charset=utf-8", "image unavailable"));
                return;
            }
            SendAll(clientFd, BuildBinaryHttpResponse(200, "OK", "image/jpeg", jpegBytes));
            return;
        }
        if (path == "/api/image/stream.mjpeg")
        {
            HandleMjpegStream(clientFd);
            return;
        }

        std::string assetName;
        std::string contentType;
        if (path == "/" || path == "/index.html")
        {
            assetName = "index.html";
            contentType = "text/html; charset=utf-8";
        }
        else if (path == "/app.js")
        {
            assetName = "app.js";
            contentType = "application/javascript; charset=utf-8";
        }
        else if (path == "/style.css")
        {
            assetName = "style.css";
            contentType = "text/css; charset=utf-8";
        }
        else
        {
            SendAll(clientFd, BuildHttpResponse(404, "Not Found", "text/plain; charset=utf-8", "not found"));
            return;
        }

        const std::string body = ReadFileToString(fs::path(config.assetDir) / assetName);
        if (body.empty())
        {
            SendAll(clientFd, BuildHttpResponse(500, "Internal Server Error", "text/plain; charset=utf-8", "asset missing"));
            return;
        }
        SendAll(clientFd, BuildHttpResponse(200, "OK", contentType, body));
    }

    void HandlePost(int clientFd, const HttpRequest& request)
    {
        UiActionResult result;
        if (request.path == "/api/control/start")
        {
            result = config.startHandler();
        }
        else if (request.path == "/api/control/stop")
        {
            result = config.stopHandler();
        }
        else if (request.path == "/api/control/discard")
        {
            result = config.discardHandler();
        }
        else if (request.path == "/api/control/estop")
        {
            result = config.estopHandler();
        }
        else if (request.path == "/api/control/clear-fault")
        {
            result = config.clearFaultHandler();
        }
        else if (request.path == "/api/control/quit")
        {
            result = config.quitHandler();
        }
        else if (request.path == "/api/label/submit")
        {
            SegmentLabelInput input;
            input.segmentStatus = ExtractJsonStringField(request.body, "segment_status").value_or("");
            input.success = ExtractJsonStringField(request.body, "success").value_or("");
            input.terminationReason = ExtractJsonStringField(request.body, "termination_reason").value_or("");
            result = config.submitLabelHandler(input);
        }
        else
        {
            SendAll(clientFd, BuildHttpResponse(404, "Not Found", "text/plain; charset=utf-8", "not found"));
            return;
        }

        const int statusCode = result.ok ? 200 : 409;
        SendAll(clientFd, BuildHttpResponse(statusCode, result.ok ? "OK" : "Conflict", "application/json; charset=utf-8", MakeJsonActionResult(result)));
    }

    void HandleMjpegStream(int clientFd)
    {
        if (!config.nextImageFrameProvider)
        {
            SendAll(clientFd, BuildHttpResponse(404, "Not Found", "text/plain; charset=utf-8", "stream unavailable"));
            return;
        }

        static const std::string kBoundary = "frame";
        std::ostringstream header;
        header << "HTTP/1.1 200 OK\r\n"
               << "Content-Type: multipart/x-mixed-replace; boundary=" << kBoundary << "\r\n"
               << "Cache-Control: no-store\r\n"
               << "Connection: close\r\n"
               << "Access-Control-Allow-Origin: *\r\n\r\n";
        if (!SendAll(clientFd, header.str()))
        {
            return;
        }

        uint64_t lastSequence = 0;
        while (running.load())
        {
            const UiImageFrame frame = config.nextImageFrameProvider(lastSequence, 1000);
            if (!running.load())
            {
                return;
            }
            if (!frame.valid || frame.jpegBytes.empty() || frame.sequence <= lastSequence)
            {
                continue;
            }

            std::ostringstream partHeader;
            partHeader << "--" << kBoundary << "\r\n"
                       << "Content-Type: image/jpeg\r\n"
                       << "Content-Length: " << frame.jpegBytes.size() << "\r\n"
                       << "X-Sequence: " << frame.sequence << "\r\n\r\n";
            if (!SendAll(clientFd, partHeader.str()) ||
                !SendAll(clientFd, frame.jpegBytes) ||
                !SendAll(clientFd, std::string("\r\n")))
            {
                return;
            }
            lastSequence = frame.sequence;
        }
    }

    void ShutdownClientConnections()
    {
        std::vector<int> clientFdsSnapshot;
        {
            std::lock_guard<std::mutex> lock(clientMutex);
            clientFdsSnapshot = clientFds;
        }
        for (const int clientFd : clientFdsSnapshot)
        {
            ::shutdown(clientFd, SHUT_RDWR);
        }
    }

    void ReapFinishedClientThreads(bool joinAll)
    {
        std::lock_guard<std::mutex> lock(clientThreadMutex);
        for (auto it = clientThreads.begin(); it != clientThreads.end();)
        {
            if (!joinAll && (!it->done || !it->done->load()))
            {
                ++it;
                continue;
            }
            if (it->thread.joinable())
            {
                it->thread.join();
            }
            it = clientThreads.erase(it);
        }
    }

    WebUiServerConfig config;
    std::atomic<bool> running{false};
    int serverFd = -1;
    std::thread thread;
    std::mutex clientMutex;
    std::vector<int> clientFds;
    std::mutex clientThreadMutex;
    std::vector<ClientThreadEntry> clientThreads;
};

WebUiServer::WebUiServer(const WebUiServerConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
}

WebUiServer::~WebUiServer() = default;

void WebUiServer::Start()
{
    impl_->Start();
}

void WebUiServer::Stop()
{
    impl_->Stop();
}
