/*
 * HIPU MMAB: Local memory of HIPU core
 */

#ifndef HW_HIPU_MMAB_H
#define HW_HIPU_MMAB_H

#define TYPE_HIPU_MMAB "riscv.hipu.mmab"
#define HIPU_MMA_RANGE 0x40000
#define HIPU_MMB_RANGE 0x80000
#define HIPU_MMAB(obj) \
    OBJECT_CHECK(HiPUMMABState, (obj), TYPE_HIPU_MMAB)


typedef struct HiPUMMABState {
    /*< private >*/
    SysBusDevice parent_obj;

    /*< public >*/
    MemoryRegion mmio;
    uint32_t aperture_size;
} HiPUMMABState;

DeviceState *hipu_mmab_create(hwaddr addr, hwaddr size);

#endif