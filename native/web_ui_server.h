#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

struct SegmentLabelInput
{
    std::string segmentStatus;
    std::string success;
    std::string terminationReason;
};

struct UiActionResult
{
    bool ok = false;
    std::string code;
    std::string message;
    std::string episodeId;
};

struct UiActionAvailability
{
    bool canStartRecording = false;
    bool canStopRecording = false;
    bool canDiscardSegment = false;
    bool canSubmitLabel = false;
    bool canEstop = false;
    bool canClearFault = false;
};

struct UiStatusSnapshot
{
    bool running = false;
    bool webUiEnabled = false;
    int webPort = 0;
    std::string sessionDir;
    std::string captureMode;
    std::string captureState;
    std::string safetyState;
    std::string faultReason;
    bool recording = false;
    double segmentDurationSeconds = 0.0;
    size_t bufferedFrames = 0;
    bool robotConnected = false;
    bool stateValid = false;
    bool imageValid = false;
    double stateAgeSeconds = -1.0;
    double imageAgeSeconds = -1.0;
    double bodyHeight = 0.0;
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    double commandVx = 0.0;
    double commandVy = 0.0;
    double commandWz = 0.0;
    std::string sceneId;
    std::string operatorId;
    std::string instruction;
    std::string taskFamily;
    std::string targetType;
    std::string targetDescription;
    std::string targetInstanceId;
    std::string taskTagsCsv;
    std::string collectorNotes;
    double cmdVxMax = 0.0;
    double cmdVyMax = 0.0;
    double cmdWzMax = 0.0;
    std::string networkInterface;
    std::string outputDir;
    double loopHz = 0.0;
    double videoPollHz = 0.0;
    std::string inputBackend;
    std::string inputDevice;
    std::string defaultsPath;
    bool pendingLabelActive = false;
    std::string pendingEpisodeId;
    size_t pendingLabelBufferedFrames = 0;
    UiActionAvailability actions;
};

struct UiConfigUpdateInput
{
    std::string sceneId;
    std::string operatorId;
    std::string instruction;
    std::string captureMode;
    std::string taskFamily;
    std::string targetType;
    std::string targetDescription;
    std::string targetInstanceId;
    std::string taskTagsCsv;
    std::string collectorNotes;
    double cmdVxMax = 0.0;
    double cmdVyMax = 0.0;
    double cmdWzMax = 0.0;
};

struct WebUiServerConfig
{
    int port = 8080;
    std::string assetDir;
    std::function<UiStatusSnapshot()> statusProvider;
    std::function<UiActionResult()> startHandler;
    std::function<UiActionResult()> stopHandler;
    std::function<UiActionResult()> discardHandler;
    std::function<UiActionResult()> estopHandler;
    std::function<UiActionResult()> clearFaultHandler;
    std::function<UiActionResult(const SegmentLabelInput&)> submitLabelHandler;
    std::function<UiActionResult(const UiConfigUpdateInput&)> updateConfigHandler;
    std::function<UiActionResult()> saveDefaultsHandler;
};

class WebUiServer
{
public:
    explicit WebUiServer(const WebUiServerConfig& config);
    ~WebUiServer();

    void Start();
    void Stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
