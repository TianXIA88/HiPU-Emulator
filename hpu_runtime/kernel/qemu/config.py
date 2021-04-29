class global_var:
    QEMU_PATH = "/home/jingwen/Work/qemu-install/qemu4_install/bin/qemu-system-riscv32"
    ifm_qemu_addr = "0xc0000000"
    wt_qemu_addr = "0xc0100000"

def get_QEMU_PATH():
    return global_var.QEMU_PATH

def get_ifm_qemu_addr():
    return global_var.ifm_qemu_addr

def get_wt_qemu_addr():
    return global_var.wt_qemu_addr