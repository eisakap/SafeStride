#include "person_detect_model_data.h"

#include <cstddef>

alignas(8) const unsigned char g_person_detect_model_data[] = {
#include "person_detect_model_data_generated.inc"
};

const int g_person_detect_model_data_len = static_cast<int>(sizeof(g_person_detect_model_data));
