/*
 * HIPU MMAB: Local memory of HIPU core
 */

#include "qemu/osdep.h"
#include "qemu/error-report.h"
#include "hw/sysbus.h"
#include "target/riscv/cpu.h"
#include "hw/riscv/hipu_mmab.h"
#include "qemu/timer.h"

#ifdef CONFIG_HIPUISA
/* CPU wants to read MMAB address */
static uint64_t hipu_mmab_read(void *opaque, hwaddr addr, unsigned size)
{
    // HiPUMMABState *clint = opaque;
    size_t hartid = 0;
    CPUState *cpu = qemu_get_cpu(hartid);
    CPURISCVState *env = cpu ? cpu->env_ptr : NULL;
    uint32_t *mmab_ptr = (uint32_t *)&env->mmab.mblks[0];
    if(addr >= HIPU_MMB_RANGE || size != 4){
        error_report("hipu mmab: invalid read: %08x size=%d", (uint32_t)addr, size);
        return 0;
    }
    else{
        return mmab_ptr[addr>>2];
    }
}

/* CPU wrote to MMAB address */
static void hipu_mmab_write(void *opaque, hwaddr addr, uint64_t value,
        unsigned size)
{
    // HiPUMMABState *clint = opaque;
    size_t hartid = 0;
    CPUState *cpu = qemu_get_cpu(hartid);
    CPURISCVState *env = cpu ? cpu->env_ptr : NULL;
    uint32_t *mmab_ptr = (uint32_t *)&env->mmab.mblks[0];
    // printf("hipu_mmab_write: %lx\n\r", addr);
    if(addr >= HIPU_MMA_RANGE || size != 4)
        error_report("hipu mmab: invalid write: %08x size=%d", (uint32_t)addr, size);
    else
        mmab_ptr[addr>>2] = (uint32_t)value;
    return;
}

static const MemoryRegionOps hipu_mmab_ops = {
    .read = hipu_mmab_read,
    .write = hipu_mmab_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4
    }
};

static Property hipu_mmab_properties[] = {
    DEFINE_PROP_UINT32("aperture-size", HiPUMMABState, aperture_size, 0),
    DEFINE_PROP_END_OF_LIST(),
};

static void hipu_mmab_realize(DeviceState *dev, Error **errp)
{
    HiPUMMABState *s = HIPU_MMAB(dev);
    memory_region_init_io(&s->mmio, OBJECT(dev), &hipu_mmab_ops, s,
                          TYPE_HIPU_MMAB, s->aperture_size);
    sysbus_init_mmio(SYS_BUS_DEVICE(dev), &s->mmio);
}

static void hipu_mmab_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    dc->realize = hipu_mmab_realize;
    dc->props = hipu_mmab_properties;
}

static const TypeInfo hipu_mmab_info = {
    .name          = TYPE_HIPU_MMAB,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(HiPUMMABState),
    .class_init    = hipu_mmab_class_init,
};

static void hipu_mmab_register_types(void)
{
    type_register_static(&hipu_mmab_info);
}

type_init(hipu_mmab_register_types)


/*
 * Create CLINT device.
 */
DeviceState *hipu_mmab_create(hwaddr addr, hwaddr size)
{
    DeviceState *dev = qdev_create(NULL, TYPE_HIPU_MMAB);
    qdev_prop_set_uint32(dev, "aperture-size", size);
    qdev_init_nofail(dev);
    // printf("qdev_init_nofail OVER\n\r");
    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, addr);
    return dev;
}
#endif