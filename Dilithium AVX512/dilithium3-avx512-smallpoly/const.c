#include <stdint.h>
#include "params.h"

#define MASKS11 0x201008
#define MASKS12 0x1008

const uint32_t _16xeta[16]  __attribute__((aligned(32))) = {ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA, ETA};
const uint32_t _16xmasks11[16]  __attribute__((aligned(32))) = {MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11, MASKS11};
const uint32_t _16xmasks12[16]  __attribute__((aligned(32))) = {MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12, MASKS12};
