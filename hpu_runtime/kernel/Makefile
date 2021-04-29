CROSS_PREFIX = riscv32-unknown-elf
CC = $(CROSS_PREFIX)-gcc
G++ = $(CROSS_PREFIX)-g++
AS = $(CROSS_PREFIX)-as
LD = $(CROSS_PREFIX)-ld
NM = $(CROSS_PREFIX)-nm
OBJCOPY = $(CROSS_PREFIX)-objcopy
OBJDUMP = $(CROSS_PREFIX)-objdump
STRIP = $(CROSS_PREFIX)-strip

output_name = kernel_all
config = ./link.lds

elf = $(output_name).elf
bin = $(output_name).bin
data = $(output_name).dat
hex = $(output_name).hex
dump = $(output_name).asm

OUT_DIR			= output
RT_INC_PATH		= ../include
INC_PATH		= krnl_inc
KER_PATH		= operators
LIB_PATH		= krnl_lib
QEMU_PATH		= qemu
FMT_PATH		= qemu/libfemto/include
FMT_BIN			= qemu/libfemto/libfemto.a

# ifdef QEMU
# PARAM = -DQEMU_ENV
# all: dir qemu_core1 qemu_core2
# else
# PARAM =
# all: dir veri_core1 veri_core2
# endif

OPTIMIZE		= -Os
INCLUDES 		= -I$(INC_PATH) -I$(KER_PATH) -I$(LIB_PATH) -I$(RT_INC_PATH) -I$(QEMU_PATH) -I$(FMT_PATH)
CFLAGS			= -march=rv32ima -mabi=ilp32 -mno-div -mcmodel=medany -ffunction-sections -fdata-sections -g $(INCLUDES) $(PARAM)
# LDFLAGS			= -nostartfiles -nostdlib -static -lgcc -Wl,--nmagic -Wl,--gc-sections
LDFLAGS			= -nostartfiles -static -lgcc -Wl,--nmagic -Wl,--gc-sections

ifdef QEMU
PARAM = -DQEMU_ENV
# config = ./link_qemu.lds
# OBJ = $(OUT_DIR)/start.o $(OUT_DIR)/hpu_api.o $(OUT_DIR)/dma.co $(OUT_DIR)/lock.co $(OUT_DIR)/libconv.co $(OUT_DIR)/int.co $(OUT_DIR)/krnl_log.co $(OUT_DIR)/hpu_util.co $(OUT_DIR)/conv2d.co $(OUT_DIR)/vadd.co $(OUT_DIR)/kernel_main.co $(OUT_DIR)/qemu.co  $(FMT_BIN)
OTHER_OBJ = $(OUT_DIR)/qemu.co 
else
PARAM =
# config = ./link.lds
OTHER_OBJ = $(OUT_DIR)/printf.co 
endif
OBJ = $(OTHER_OBJ) \
		$(OUT_DIR)/start.o \
		$(OUT_DIR)/hpu_api.o \
		$(OUT_DIR)/dma.co \
		$(OUT_DIR)/lock.co \
		$(OUT_DIR)/libconv.co \
		$(OUT_DIR)/int.co \
		$(OUT_DIR)/krnl_log.co \
		$(OUT_DIR)/hpu_util.co \
		$(OUT_DIR)/conv2d.co \
		$(OUT_DIR)/vadd.co \
		$(OUT_DIR)/conv3s2_dwc3s1_conv1s1.co \
		$(OUT_DIR)/conv1s1_dwc3s2_conv1s1.co \
		$(OUT_DIR)/conv1s1_dwc3s1_conv1s1_add.co \
		$(OUT_DIR)/conv1s1_conv3s1_conv3s1.co \
		$(OUT_DIR)/conv1s1_upsmp2x_add.co \
		$(OUT_DIR)/network_lane_efficientnet.co \
		$(OUT_DIR)/network_obstacle.co \
		$(OUT_DIR)/network_trafficlight.co \
		$(OUT_DIR)/kernel_main.co \
		$(OUT_DIR)/fc.co

all: dir $(bin)

dir:
	mkdir -p $(OUT_DIR)

$(bin): $(OBJ) 
	# $(CC) $(CFLAGS) $(LDFLAGS) $(OBJ) $(DEFINES) -o $(OUT_DIR)/$(elf) -T $(config) -nostartfiles -nostdlib
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $(OUT_DIR)/$(elf) -T $(config) $(FMT_BIN)
	$(OBJDUMP) -DSxtd $(OUT_DIR)/$(elf) > $(OUT_DIR)/$(dump)
	$(OBJCOPY) -O binary $(OUT_DIR)/$(elf) $(OUT_DIR)/$(bin)

	#$(OBJCOPY) -O verilog $(OUT_DIR)/$(elf) $(OUT_DIR)/$(data)
	#$(OBJCOPY) -O ihex $(OUT_DIR)/$(elf) $(OUT_DIR)/$(hex)

# $(OUT_DIR)/%.o: krnl_lib/%.s
# 	$(AS) -c $< -o $@ -march=rv32ima -mabi=ilp32
$(OUT_DIR)/%.o: krnl_lib/%.s
	$(AS) -c $< -o $@ -march=rv32ima -mabi=ilp32

$(OUT_DIR)/%.co: krnl_lib/%.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -c $< -o $@ 
	# -mcmodel=medany -ffunction-sections -fdata-sections

$(OUT_DIR)/%.co: operators/%.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -c $< -o $@ 

$(OUT_DIR)/%.co: qemu/%.c
	$(CC) $(CFLAGS) $(OPTIMIZE) -c $< -o $@ 

clean:
	rm -rf $(OBJ) $(OUT_DIR)/*

