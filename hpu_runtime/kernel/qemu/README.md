## Configure

To set the configuration, first you should enter /kernel/qemu, then edit the file config.py. This file includes variables as following:

1. QEMU_PATH, which is the path qemu-system-riscv32 is located(including qemu-system-riscv32).
2. ifm_qemu_addr, which is the address of input feature map loaded in memory.
3. wt_qemu_addr, which is the address of weight loaded in memory.

## Compile

First you should enter /kernel, then run the command:

```
./make_QEMU.sh
```

## How to run

First you should enter /kernel/qemu, then run the command:

```
python launch.py
```

Then you can see the ofm in /kernel/output/mem1.txt

