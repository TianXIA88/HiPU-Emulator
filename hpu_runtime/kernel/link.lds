OUTPUT_ARCH( "riscv" )

ENTRY( _start )

MEMORY
{
    itcm  : ORIGIN = 0x84000000, LENGTH = 128K
    dtcm  : ORIGIN = 0x88000000, LENGTH = 32K
}

SECTIONS
{
    __STACK_AREA__ = 2K;
    __IRQ_CONTEXT_SIZE__ = 256;
    __STACE_SIZE__ = __STACK_AREA__ - __IRQ_CONTEXT_SIZE__;

    .text : ALIGN(0x10)
    { 
        *start.o(.text);
        *(.text) 
    	*(.rodata .rodata.*)
    } >itcm

	.sdata : ALIGN(0x40)
	{
	  __sdata_load_start__ = LOADADDR(.sdata);
	  __sdata_start__ = .; 
	  *(.srodata*)
	  *(.sdata .sdata.*)
	  . = ALIGN(0x10);
	  __sdata_end__ = .;
	} >dtcm AT>itcm

    .data : ALIGN(0x40)
    { 
	    __data_load_start__ = LOADADDR(.data);
	    __data_start__ = .; 
	    *(.data .data.*)
	    . = ALIGN(0x10);
	    __data_end__ = .;
    } >dtcm AT>itcm

	/* sbss section */
	.sbss : ALIGN(0x40)
	{
		__sbss_start__ = .;
		*(.sbss .sbss.*)
		*(.scommon)
		. = ALIGN(0x40);
		__sbss_end__ = .;
	} >dtcm

	/* bss section */
	.bss : ALIGN(0x40)
	{ 
		__bss_start__ = .;
		*(.bss .bss.*)
		*(.common .comm)
		. = ALIGN(0x10);
		__bss_end__ = .;
	} >dtcm

    .stack ORIGIN(dtcm) + LENGTH(dtcm)  - __STACK_AREA__ * 2 :
    {
        __stack_bottom__ = .;
        . += __STACE_SIZE__;
        __stack_top__ = .;
        . += __IRQ_CONTEXT_SIZE__;
    } >dtcm

	.interrupt_stack ORIGIN(dtcm) + LENGTH(dtcm) - __STACK_AREA__ :
    {
        __interrupt_stack_bottom__ = .;
        . += __STACE_SIZE__;
        __stack_top__ = .;
        . += __IRQ_CONTEXT_SIZE__;
    } >dtcm
}

