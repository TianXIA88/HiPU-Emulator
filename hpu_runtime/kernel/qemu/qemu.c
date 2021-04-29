#include "hihw.h"

#ifdef QEMU_ENV
#include "qemu.h"
#include "dma.h"
#include "femto.h"

void qemu_echo(){
    asm(".word 0x2eadbeef":::);
}

void qemu_fprint(int type, int addr, int len){
    csrw(CSR_FPRINT_ADDR, addr);
    csrw(CSR_FPRINT_LEN, len);
    if(type == QEMU_LOG_MEM)
        asm(".word 0xbeadbeef":::);
    else if(type == QEMU_LOG_MMAB)
        asm(".word 0x8eadbeef":::);
    return;
}

auxval_t __auxv[] = {
    { UART0_CLOCK_FREQ,         32000000   },
    { UART0_BAUD_RATE,          115200     },
    { SIFIVE_UART0_CTRL_ADDR,   0x10013000 },
    { SIFIVE_TEST_CTRL_ADDR,    0x100000   },
    { 0, 0 }
};

void qemu_arch_setup()
{
    register_console(&console_sifive_uart);
    register_poweroff(&poweroff_sifive_test);
    printf("QEMU ARCH setup over\n\r");
}

#endif