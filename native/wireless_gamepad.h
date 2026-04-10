#pragma once

#include <cmath>
#include <cstdint>

#include <unitree/idl/go2/WirelessController_.hpp>

namespace collector::input
{

inline float ApplyAxisDeadZone(float value, float deadZone)
{
    return std::fabs(value) < deadZone ? 0.0f : value;
}

// Copied from the Unitree SDK wireless controller example so collector input
// follows the same button-edge and stick filtering behavior.
union KeySwitch
{
    struct
    {
        uint8_t R1 : 1;
        uint8_t L1 : 1;
        uint8_t start : 1;
        uint8_t select : 1;
        uint8_t R2 : 1;
        uint8_t L2 : 1;
        uint8_t F1 : 1;
        uint8_t F2 : 1;
        uint8_t A : 1;
        uint8_t B : 1;
        uint8_t X : 1;
        uint8_t Y : 1;
        uint8_t up : 1;
        uint8_t right : 1;
        uint8_t down : 1;
        uint8_t left : 1;
    } components;
    uint16_t value = 0;
};

class Button
{
public:
    void Update(bool state)
    {
        onPress = state ? state != pressed : false;
        onRelease = state ? false : state != pressed;
        pressed = state;
    }

    bool pressed = false;
    bool onPress = false;
    bool onRelease = false;
};

class Gamepad
{
public:
    void Update(const unitree_go::msg::dds_::WirelessController_& message)
    {
        rawLx = ApplyAxisDeadZone(message.lx(), deadZone);
        rawRx = ApplyAxisDeadZone(message.rx(), deadZone);
        rawRy = ApplyAxisDeadZone(message.ry(), deadZone);
        rawLy = ApplyAxisDeadZone(message.ly(), deadZone);

        lx = lx * (1.0f - smooth) + rawLx * smooth;
        rx = rx * (1.0f - smooth) + rawRx * smooth;
        ry = ry * (1.0f - smooth) + rawRy * smooth;
        ly = ly * (1.0f - smooth) + rawLy * smooth;

        key_.value = message.keys();
        R1.Update(key_.components.R1);
        L1.Update(key_.components.L1);
        start.Update(key_.components.start);
        select.Update(key_.components.select);
        R2.Update(key_.components.R2);
        L2.Update(key_.components.L2);
        F1.Update(key_.components.F1);
        F2.Update(key_.components.F2);
        A.Update(key_.components.A);
        B.Update(key_.components.B);
        X.Update(key_.components.X);
        Y.Update(key_.components.Y);
        up.Update(key_.components.up);
        right.Update(key_.components.right);
        down.Update(key_.components.down);
        left.Update(key_.components.left);
    }

    float smooth = 0.03f;
    float deadZone = 0.01f;

    float lx = 0.0f;
    float rx = 0.0f;
    float ry = 0.0f;
    float ly = 0.0f;
    float rawLx = 0.0f;
    float rawRx = 0.0f;
    float rawRy = 0.0f;
    float rawLy = 0.0f;

    Button R1;
    Button L1;
    Button start;
    Button select;
    Button R2;
    Button L2;
    Button F1;
    Button F2;
    Button A;
    Button B;
    Button X;
    Button Y;
    Button up;
    Button right;
    Button down;
    Button left;

private:
    KeySwitch key_{};
};

} // namespace collector::input
