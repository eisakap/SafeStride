/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Upstream: espressif/esp-tflite-micro examples/person_detection
 * https://components.espressif.com/components/espressif/esp-tflite-micro
 */
#pragma once

extern const unsigned char g_person_detect_model_data[];
extern const int g_person_detect_model_data_len;

/** Smallest plausible blob size — stub in src/person_detect_model_data.cpp is below this. */
constexpr int kPersonDetectModelMinBytes = 1024;
