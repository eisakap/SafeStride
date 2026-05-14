#include <Arduino.h>
#include <math.h>
#include <Wire.h>
#include <VL53L0X.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>

#include "model_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

static constexpr int kI2cSdaPin = 13;
static constexpr int kI2cSclPin = 14;
static constexpr int kHapticPin = 16;
static constexpr ledc_channel_t kHapticLedcChannel = LEDC_CHANNEL_1;

static constexpr int kModelW = 96;
static constexpr int kModelH = 96;

static constexpr uint16_t kTofCriticalMm = 1000;
static constexpr uint16_t kTofValidMaxMm = 8190;
static constexpr float kMlConfidenceTrip = 0.60f;
static constexpr uint32_t kFramePeriodMs = 160;
static constexpr uint32_t kTofPeriodMs = 45;
static constexpr uint32_t kHapticRefreshMs = 12;

static constexpr size_t kTensorArenaBytes = 150 * 1024;

#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM (-1)
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

static VL53L0X g_tof;
static bool g_cam_ok = false;
static bool g_tof_ok = false;
static bool g_ml_ok = false;
static bool g_ml_simulated = false;
static tflite::MicroErrorReporter g_error_reporter;
static const tflite::Model* g_model = nullptr;
static tflite::AllOpsResolver g_resolver;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input = nullptr;
static TfLiteTensor* g_output = nullptr;
static uint8_t* g_tensor_arena = nullptr;

static uint8_t g_gray96[kModelW * kModelH];

static uint16_t g_last_tof_mm = 8191;
static float g_last_ml_score = 0.0f;

static uint32_t g_next_frame_ms = 0;
static uint32_t g_next_tof_ms = 0;
static uint32_t g_next_haptic_ms = 0;

static uint32_t g_cam_fail = 0;
static uint32_t g_ml_invoke_fail = 0;

static bool initCamera();
static bool initTof();
static bool initMl();
static bool initHaptic();
static void shutdownMl();

static inline uint32_t nowMs() { return static_cast<uint32_t>(millis()); }

static void logLine(const __FlashStringHelper* msg) { Serial.println(msg); }

static void applyHapticFromDistance(uint16_t distance_mm, bool alert_active) {
  if (!alert_active) {
    ledcWrite(kHapticLedcChannel, 0);
    return;
  }

  const uint16_t near_clip = 150;
  uint16_t d = distance_mm;
  if (d < near_clip) {
    d = near_clip;
  }
  if (d > kTofCriticalMm) {
    d = kTofCriticalMm;
  }

  const float span = static_cast<float>(kTofCriticalMm - near_clip);
  float closeness = 1.0f - (static_cast<float>(d - near_clip) / span);
  if (closeness < 0.0f) {
    closeness = 0.0f;
  }
  if (closeness > 1.0f) {
    closeness = 1.0f;
  }

  uint32_t freq_hz = 90;
  if (closeness > 0.85f) {
    freq_hz = 340;
  } else if (closeness > 0.55f) {
    freq_hz = 240;
  } else if (closeness > 0.25f) {
    freq_hz = 160;
  }

  ledcSetup(static_cast<uint8_t>(kHapticLedcChannel), freq_hz, 10);
  ledcAttachPin(kHapticPin, static_cast<uint8_t>(kHapticLedcChannel));

  const uint32_t duty_max = (1u << 10) - 1u;
  uint32_t duty = static_cast<uint32_t>(duty_max * closeness);
  if (duty < 48 && alert_active) {
    duty = 48;
  }
  ledcWrite(kHapticLedcChannel, duty);
}

static bool rgb565ToGrayBuffer(const uint8_t* buf, int w, int h, uint8_t* out_gray) {
  const auto* row = reinterpret_cast<const uint16_t*>(buf);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const uint16_t px = row[y * w + x];
      const uint8_t r5 = (px >> 11) & 0x1F;
      const uint8_t g6 = (px >> 5) & 0x3F;
      const uint8_t b5 = px & 0x1F;
      const uint8_t r = (r5 << 3) | (r5 >> 2);
      const uint8_t g = (g6 << 2) | (g6 >> 4);
      const uint8_t b = (b5 << 3) | (b5 >> 2);
      const uint16_t yv = static_cast<uint16_t>((r * 38 + g * 75 + b * 15) >> 7);
      out_gray[y * w + x] = static_cast<uint8_t>(yv > 255 ? 255 : yv);
    }
  }
  return true;
}

static void resizeGrayNearest(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
  for (int y = 0; y < dh; ++y) {
    const int sy = (y * sh) / dh;
    for (int x = 0; x < dw; ++x) {
      const int sx = (x * sw) / dw;
      dst[y * dw + x] = src[sy * sw + sx];
    }
  }
}

static bool buildModelInputGrayscale(camera_fb_t* fb) {
  const int w = fb->width;
  const int h = fb->height;

  if (fb->format == PIXFORMAT_GRAYSCALE) {
    resizeGrayNearest(fb->buf, w, h, g_gray96, kModelW, kModelH);
    return true;
  }
  if (fb->format == PIXFORMAT_RGB565) {
    const size_t need = static_cast<size_t>(w) * static_cast<size_t>(h);
    uint8_t* tmp =
        static_cast<uint8_t*>(heap_caps_malloc(need, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!tmp) {
      tmp = static_cast<uint8_t*>(malloc(need));
    }
    if (!tmp) {
      return false;
    }
    rgb565ToGrayBuffer(fb->buf, w, h, tmp);
    resizeGrayNearest(tmp, w, h, g_gray96, kModelW, kModelH);
    free(tmp);
    return true;
  }
  logLine(F("[cam] Unsupported pixel format for ML preprocessing"));
  return false;
}

static void fillInputTensor(const uint8_t* gray96) {
  if (!g_input) {
    return;
  }
  const int n = kModelW * kModelH;
  switch (g_input->type) {
    case kTfLiteUInt8: {
      memcpy(g_input->data.uint8, gray96, static_cast<size_t>(n));
      break;
    }
    case kTfLiteInt8: {
      for (int i = 0; i < n; ++i) {
        g_input->data.int8[i] =
            static_cast<int8_t>(static_cast<uint8_t>(gray96[i]) ^ 0x80u);
      }
      break;
    }
    case kTfLiteFloat32: {
      for (int i = 0; i < n; ++i) {
        g_input->data.f[i] = static_cast<float>(gray96[i]) * (1.0f / 255.0f);
      }
      break;
    }
    default:
      logLine(F("[ml] Unsupported input tensor type"));
      break;
  }
}

static float dequantizeOutputScalar(const TfLiteTensor* out, int idx) {
  switch (out->type) {
    case kTfLiteFloat32:
      return out->data.f[idx];
    case kTfLiteUInt8:
      return (static_cast<float>(out->data.uint8[idx]) - static_cast<float>(out->params.zero_point)) *
             out->params.scale;
    case kTfLiteInt8:
      return (static_cast<float>(out->data.int8[idx]) - static_cast<float>(out->params.zero_point)) *
             out->params.scale;
    default:
      return 0.0f;
  }
}

static float interpretPersonObstacleScore() {
  if (!g_output) {
    return 0.0f;
  }

  int flat = 1;
  for (int d = 0; d < g_output->dims->size; ++d) {
    flat *= g_output->dims->data[d];
  }
  if (flat <= 0) {
    return 0.0f;
  }

  constexpr int kPersonClassIndex = 1;
  float p = (flat >= 2) ? dequantizeOutputScalar(g_output, kPersonClassIndex)
                        : dequantizeOutputScalar(g_output, 0);
  if (p < 0.0f) {
    p = 0.0f;
  }
  if (p > 1.0f) {
    p = 1.0f;
  }
  return p;
}

static bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

#if defined(CONFIG_IDF_TARGET_ESP32S3)
  config.sccb_i2c_port = 0;
#endif

  auto try_init = [&](camera_config_t& c) -> esp_err_t {
    esp_camera_deinit();
    return esp_camera_init(&c);
  };

  esp_err_t err = try_init(config);
  if (err == ESP_OK) {
    return true;
  }

  Serial.printf_P(PSTR("[cam] GRAY/QVGA+PSRAM failed (0x%x), retry GRAY/QQVGA\n"), static_cast<unsigned>(err));
  config.frame_size = FRAMESIZE_QQVGA;
  err = try_init(config);
  if (err == ESP_OK) {
    return true;
  }

  Serial.printf_P(PSTR("[cam] GRAY/QQVGA failed (0x%x), retry DRAM buffer\n"), static_cast<unsigned>(err));
  config.frame_size = FRAMESIZE_QQVGA;
  config.fb_location = CAMERA_FB_IN_DRAM;
  err = try_init(config);
  if (err == ESP_OK) {
    return true;
  }

  Serial.printf_P(PSTR("[cam] DRAM GRAY failed (0x%x), retry RGB565/QQVGA+DRAM\n"), static_cast<unsigned>(err));
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QQVGA;
  config.fb_location = CAMERA_FB_IN_DRAM;
  err = try_init(config);
  if (err != ESP_OK) {
    Serial.printf_P(PSTR("[cam] Camera init failed: 0x%x\n"), static_cast<unsigned>(err));
    return false;
  }
  return true;
}

static bool initTof() {
  Wire.begin(kI2cSdaPin, kI2cSclPin);
  Wire.setClock(400000);

  g_tof.setTimeout(35);
  if (!g_tof.init()) {
    logLine(F("[tof] VL53L0X init failed (wiring / address / pull-ups)"));
    return false;
  }
  g_tof.setSignalRateLimit(0.25f);
  g_tof.setMeasurementTimingBudget(22000);
  g_tof.startContinuous(33);
  return true;
}

static bool initMl() {
#if SAFE_STRIDE_SIMULATED_ML
  g_ml_simulated = true;
  logLine(F("[ml] SIMULATED — placeholder person score (set SAFE_STRIDE_SIMULATED_ML=0 + real weights to infer)"));
  return true;
#else
  if (g_person_detect_model_data_len < kPersonDetectModelMinBytes) {
    logLine(F("[ml] Person model missing — run scripts/fetch_person_detect_model.ps1 (or paste weights)."));
    return false;
  }

  g_model = tflite::GetModel(g_person_detect_model_data);
  if (!g_model) {
    logLine(F("[ml] GetModel() returned null"));
    return false;
  }
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf_P(PSTR("[ml] Schema mismatch (model %lu, runtime %d)\n"),
                    static_cast<unsigned long>(g_model->version()), TFLITE_SCHEMA_VERSION);
    return false;
  }

  g_tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kTensorArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!g_tensor_arena) {
    g_tensor_arena = static_cast<uint8_t*>(malloc(kTensorArenaBytes));
  }
  if (!g_tensor_arena) {
    logLine(F("[ml] Tensor arena allocation failed"));
    return false;
  }

  static tflite::MicroInterpreter static_interpreter(g_model, g_resolver, g_tensor_arena,
                                                     kTensorArenaBytes, &g_error_reporter);
  g_interpreter = &static_interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    logLine(F("[ml] AllocateTensors() failed (arena too small or unsupported ops?)"));
    shutdownMl();
    return false;
  }

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  if (!g_input || !g_output) {
    logLine(F("[ml] Missing input/output tensors"));
    shutdownMl();
    return false;
  }

  if (g_input->dims->size < 3) {
    logLine(F("[ml] Expected CHW/HWC rank-3 input — adjust preprocessing"));
  }

  Serial.printf_P(PSTR("[ml] Initialized (arena=%u bytes, input type=%d)\n"),
                  static_cast<unsigned>(kTensorArenaBytes), static_cast<int>(g_input->type));
  return true;
#endif
}

static void shutdownMl() {
  if (g_tensor_arena) {
    free(g_tensor_arena);
    g_tensor_arena = nullptr;
  }
  g_interpreter = nullptr;
  g_model = nullptr;
  g_input = nullptr;
  g_output = nullptr;
}

static bool initHaptic() {
  ledcSetup(static_cast<uint8_t>(kHapticLedcChannel), 140, 10);
  ledcAttachPin(kHapticPin, static_cast<uint8_t>(kHapticLedcChannel));
  ledcWrite(kHapticLedcChannel, 0);
  return true;
}

void setup() {
  Serial.begin(115200);
  const uint32_t boot_t = nowMs();
  while (!Serial && (nowMs() - boot_t) < 1500) {
  }

  logLine(F("\n=== SafeStride boot ==="));
  g_cam_ok = initCamera();
  g_tof_ok = initTof();
  g_ml_ok = initMl();
#if !SAFE_STRIDE_SIMULATED_ML
  g_ml_ok = g_ml_ok && g_cam_ok;
#endif

  initHaptic();

  const uint32_t t0 = nowMs();
  g_next_frame_ms = t0;
  g_next_tof_ms = t0;
  g_next_haptic_ms = t0;

  if (g_cam_ok) {
    logLine(F("[cam] OK"));
  }
  if (g_tof_ok) {
    logLine(F("[tof] OK"));
  }
  if (g_ml_ok) {
    logLine(F("[ml] OK"));
  } else {
    logLine(F("[ml] Disabled or failed — proximity alerts still available via TOF if present"));
  }
}

void loop() {
  const uint32_t t = nowMs();

  if (g_tof_ok && t >= g_next_tof_ms) {
    g_next_tof_ms = t + kTofPeriodMs;
    const uint16_t mm = g_tof.readRangeContinuousMillimeters();
    if (!g_tof.timeoutOccurred() && mm < kTofValidMaxMm) {
      g_last_tof_mm = mm;
    }
  }

  if (g_ml_ok && t >= g_next_frame_ms) {
    g_next_frame_ms = t + kFramePeriodMs;

    if (g_ml_simulated) {
      if (g_cam_ok) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          if (buildModelInputGrayscale(fb)) {
            uint32_t sum = 0;
            for (int i = 0; i < kModelW * kModelH; ++i) {
              sum += g_gray96[i];
            }
            const float mean = static_cast<float>(sum) / static_cast<float>(kModelW * kModelH * 255);
            g_last_ml_score = 0.22f + 0.68f * mean;
          }
          esp_camera_fb_return(fb);
        }
      } else {
        g_last_ml_score = 0.5f + 0.48f * sinf(static_cast<float>(t) * 0.0009f);
      }
      if (g_last_ml_score < 0.0f) {
        g_last_ml_score = 0.0f;
      }
      if (g_last_ml_score > 1.0f) {
        g_last_ml_score = 1.0f;
      }
    } else if (g_cam_ok) {
      camera_fb_t* fb = esp_camera_fb_get();
      if (!fb) {
        g_cam_fail++;
        if ((g_cam_fail % 25u) == 1u) {
          logLine(F("[cam] Frame capture failed"));
        }
      } else {
        if (buildModelInputGrayscale(fb)) {
          fillInputTensor(g_gray96);
          if (g_interpreter->Invoke() != kTfLiteOk) {
            g_ml_invoke_fail++;
            if ((g_ml_invoke_fail % 10u) == 1u) {
              logLine(F("[ml] Invoke failed"));
            }
          } else {
            g_last_ml_score = interpretPersonObstacleScore();
          }
        }
        esp_camera_fb_return(fb);
      }
    }
  }

  const bool tof_trip = g_tof_ok && (g_last_tof_mm < kTofCriticalMm);
  const bool ml_trip = g_ml_ok && (g_last_ml_score >= kMlConfidenceTrip);
  const bool alert = tof_trip || ml_trip;

  if (t >= g_next_haptic_ms) {
    g_next_haptic_ms = t + kHapticRefreshMs;
    uint16_t haptic_distance = g_last_tof_mm;
    if (!g_tof_ok || g_last_tof_mm >= kTofValidMaxMm) {
      haptic_distance = kTofCriticalMm;
    }
    applyHapticFromDistance(haptic_distance, alert);
  }

#if defined(CORE_DEBUG_LEVEL) && (CORE_DEBUG_LEVEL > 0)
  static uint32_t last_report = 0;
  if (t - last_report > 750) {
    last_report = t;
    Serial.printf_P(PSTR("[dbg] mm=%u ml=%.3f trip=%u tof=%u ml_t=%u\n"), g_last_tof_mm, g_last_ml_score,
                    static_cast<unsigned>(alert ? 1u : 0u), static_cast<unsigned>(tof_trip ? 1u : 0u),
                    static_cast<unsigned>(ml_trip ? 1u : 0u));
  }
#endif

  yield();
}
