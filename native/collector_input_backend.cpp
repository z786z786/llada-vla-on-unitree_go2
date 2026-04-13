#include "collector_input_backend.h"

#include <cctype>
#include <optional>

namespace collector::input
{

namespace
{

std::optional<TeleopEvent> EventFromChar(char ch)
{
    const char lowered = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    switch (lowered)
    {
    case 'r':
        return TeleopEvent{TeleopEventType::StartCapture};
    case 't':
        return TeleopEvent{TeleopEventType::StopCapture};
    case 'c':
        return TeleopEvent{TeleopEventType::ClearFault};
    case 'v':
        return TeleopEvent{TeleopEventType::ToggleStand};
    case 'p':
        return TeleopEvent{TeleopEventType::PrintStatus};
    case 'h':
        return TeleopEvent{TeleopEventType::PrintHelp};
    case 'x':
        return TeleopEvent{TeleopEventType::Quit};
    case '1':
    case '2':
    case '3':
    case '4':
        return TeleopEvent{TeleopEventType::SubmitLabelShortcut, lowered};
    default:
        break;
    }

    if (ch == 27)
    {
        return TeleopEvent{TeleopEventType::CancelSegment};
    }
    if (ch == ' ')
    {
        return TeleopEvent{TeleopEventType::EmergencyStop};
    }
    return std::nullopt;
}

}  // namespace

std::optional<TeleopEvent> EventFromTerminalChar(char ch)
{
    return EventFromChar(ch);
}

}  // namespace collector::input
