#include <iostream>
#include <string>
#include <vector>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>

namespace
{

std::string ServiceAlias(const std::string& form, const std::string& name)
{
    if (form == "0")
    {
        if (name == "normal")
        {
            return "sport_mode";
        }
        if (name == "ai")
        {
            return "ai_sport";
        }
        if (name == "advanced")
        {
            return "advanced_sport";
        }
    }
    else
    {
        if (name == "ai-w")
        {
            return "wheeled_sport(go2W)";
        }
        if (name == "normal-w")
        {
            return "wheeled_sport(b2W)";
        }
    }
    return "";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " networkInterface" << std::endl;
        return 2;
    }

    try
    {
        const std::string networkInterface = argv[1];
        unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);

        unitree::robot::b2::MotionSwitcherClient motionSwitcher;
        motionSwitcher.SetTimeout(10.0f);
        motionSwitcher.Init();

        std::string form;
        std::string modeName;
        const int32_t modeRet = motionSwitcher.CheckMode(form, modeName);

        std::cout << "CheckMode ret: " << modeRet << std::endl;
        if (modeRet != 0)
        {
            std::cout << "Motion mode: unknown (CheckMode failed or timed out)" << std::endl;
        }
        else if (modeName.empty())
        {
            std::cout << "Motion mode: release/low-level or inactive" << std::endl;
        }
        else
        {
            std::cout << "Motion mode: form=" << form << " name=" << modeName;
            const std::string alias = ServiceAlias(form, modeName);
            if (!alias.empty())
            {
                std::cout << " alias=" << alias;
            }
            std::cout << std::endl;
        }

        unitree::robot::go2::RobotStateClient robotState;
        robotState.SetTimeout(10.0f);
        robotState.Init();

        std::vector<unitree::robot::go2::ServiceState> services;
        const int32_t listRet = robotState.ServiceList(services);
        std::cout << "ServiceList ret: " << listRet << std::endl;
        std::cout << "serviceStateList size: " << services.size() << std::endl;
        for (const auto& service : services)
        {
            std::cout << "name=" << service.name
                      << " status=" << service.status
                      << " protect=" << service.protect << std::endl;
        }

        unitree::robot::ChannelFactory::Instance()->Release();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "go2_mode_probe fatal error: " << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "go2_mode_probe fatal error: unknown exception" << std::endl;
    }

    return 1;
}
