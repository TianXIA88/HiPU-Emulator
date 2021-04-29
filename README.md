# QEMU-HiPU

## Introduction
QENU-HiPU is developed based on the official QEMU-4.0 version software, which is aimed to provide system-level fucntional simulation for HiPU instruction extension.
HiPU instruction set is extended on standard RISC-V IA(Atmoic) architecture. 
Current QEMU-HiPU simulator is based on the sifive-e development board, which means we follow the sifive-e's hardware resources, address space, peripharals, etc.
You can check the spec of board in the <SiFive-E20-Manual-v1p0.pdf> file.

## Current State
From the respective of CPU simulation, HiPU instruction extension involves the following changes:
- New vector instruction set.
- New matrix instruction set.
- New CSR registers and instructions.
- New local memory (MMA, MMB) which is addressable.
- New DMA controller to move data between main memory and local memory.

## Compile
First install the dependency packages of QEMU. For Ubuntu systems, run:
>sudo apt-get install git libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev

For other Linux distributions, please refer to [HERE](http://https://wiki.qemu.org/Hosts/Linux).

To compile and generate QEMU-RSICV binary with HiPU extension, set the configuration using the following command:
>./configure --target-list=riscv32-softmmu --prefix=$YOUR_QEMU_PATH --enable-hipuisa

Then start the compilation:
>make -j16 && make install

After the compilation, you should find the generated QEMU toolchain at the installation path $YOUR_QEMU_PATH.

To clear the compilation, run:
>make clear

## How to Use
To use QEMU-HiPU simulation for bare-metal riscv programs, run command:
>$YOUR_QEMU_PQTH/bin/qemu-system-riscv32 -machine sifive_e -nographic -kernel test.elf

## How to Debug
You can also launch the GDB to debug your program. To do so, run cammand:
>$YOUR_QEMU_PQTH/bin/qemu-system-riscv32 -machine sifive_e -nographic -s -S -kernel test.elf

Then you can start a RISC-V GDB toolchain which attaches and debugs your program, via QEMU external ports. 
Check a tutorial [HERE](http://doppioandante.github.io/2015/07/10/Simple-ARM-programming-on-linux.html).

Note that, currently GDB is not able to show the extensional registers (VR, VPR).

## DEMO
The kernel is a three-layer DNN conv-depthconv-conv implementation on HiPU. 
The actual source code to run is HPU-Runtime/kernel/operators/conv1s1\_dwc3s2\_conv1s1.c
The DEMO is to compile and run the kernel on QEMU and check the results.

### How to build
To build the kernel, go to /HPU-Runtime/kernel, and execute:
>make QEMU=1

After the compilation, the generated kernel.elf and assembly file can be found in output.

### How to run
Make sure the required parameter table and binary data files are availabe in HPU-Runtime/kernel/qemu/qemu\_bin
The command line to run demo is recored in HPU-Runtime/kernel/qemu/cmd

### Check result
The final result ofm in QEMU DDR is saved as txt file in HPU-Runtime/kernel/output/mem1.txt

### To print more information
You can tweak the prinf functions in source files to get what you want. Have fun.
