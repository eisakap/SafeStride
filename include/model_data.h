/**
 * SafeStride model wiring
 *
 * - `person_detect_model_data.*`: int8-style blob (GitHub default = synthetic placeholder bytes).
 * - Inference: `platformio.ini` sets `SAFE_STRIDE_SIMULATED_ML=1` so scores are faked for demos.
 *   Set to `0` and use Espressif’s real `person_detect_model_data.cc` for on-device TFLite.
 */
#pragma once

#include "person_detect_model_data.h"
