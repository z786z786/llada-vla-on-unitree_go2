#pragma once

#include <stdexcept>

#include <termios.h>
#include <unistd.h>

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
