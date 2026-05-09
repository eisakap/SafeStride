/**
 * Placeholder model blob for GitHub / portfolio (not a valid TFLite graph).
 * Starts with TFL3-style magic bytes + synthetic payload so the file “looks the part”.
 *
 * Firmware defaults to SAFE_STRIDE_SIMULATED_ML (see platformio.ini): no real inference.
 * For real person_detection weights, set that flag to 0 and replace this file with
 * Espressif’s person_detect_model_data.cc or run scripts/fetch_person_detect_model.ps1.
 */
#include "person_detect_model_data.h"

#include <cstddef>

alignas(8) const unsigned char g_person_detect_model_data[] = {
#include "person_detect_model_data_generated.inc"
};

const int g_person_detect_model_data_len = static_cast<int>(sizeof(g_person_detect_model_data));
