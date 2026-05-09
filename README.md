# SafeStride

SafeStride is an ESP32-CAM mobility-assist firmware prototype focused on real-time obstacle awareness for assistive navigation.

It combines:
- vision-based scene understanding
- TOF proximity sensing
- haptic feedback scaling based on risk level

## Features

- ESP32-CAM capture and preprocessing pipeline
- VL53L0X time-of-flight distance sensing over I2C
- Sensor-fusion alert logic (vision confidence + distance threshold)
- PWM haptic output with intensity tied to obstacle distance
- Non-blocking loop timing using `millis()` for responsive behavior
- PlatformIO project layout for fast iteration

## Hardware

- ESP32-CAM (AI Thinker mapping in current config)
- VL53L0X TOF sensor
- Vibration motor with proper driver stage
- Stable 3.3V power path (ESP32-CAM peak current capable)

## Current Pin Mapping

- TOF SDA -> `GPIO13`
- TOF SCL -> `GPIO14`
- Haptic PWM -> `GPIO16`

Update constants in `src/main.cpp` if your board wiring differs.

## Project Layout

- `src/main.cpp` - main control loop, camera/TOF init, haptic control, perception path
- `include/model_data.h` - model wiring include
- `include/person_detect_model_data.h` - model symbol declarations
- `src/person_detect_model_data.cpp` - model data translation unit
- `platformio.ini` - board config, build flags, library dependencies

## Build and Flash

1. Install [PlatformIO](https://platformio.org/).
2. Open this repo in Cursor or VS Code.
3. Build:
   - `pio run`
4. Upload:
   - `pio run -t upload`
5. Monitor:
   - `pio device monitor -b 115200`

## Runtime Logic

- TOF is sampled continuously and compared to a critical range threshold.
- Camera frames are resized/preprocessed for the perception path.
- Alert is triggered when either proximity or visual confidence crosses threshold.
- Haptic frequency/duty increases as distance decreases.

## Key Tunables

Tune these in `src/main.cpp`:

- `kTofCriticalMm`
- `kMlConfidenceTrip`
- `kFramePeriodMs`
- `kTofPeriodMs`
- `kHapticRefreshMs`

## Dependencies

Declared in `platformio.ini`:

- `espressif/esp32-camera`
- `pololu/VL53L0X`
- `tanakamasayuki/TensorFlowLite_ESP32`

## Roadmap

- Add environment-specific tuning profiles
- Improve haptic patterns for directional cues
- Expand evaluation tooling and logging
- Harden packaging and power subsystem for daily use

## Note

SafeStride is a development prototype and is not a certified medical device.
