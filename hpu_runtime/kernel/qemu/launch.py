import sys
import subprocess
import re
import os
import config





if __name__ == '__main__':

    # print(type(sys.argv))
    path = 'qemu_bin/'
    files = os.listdir(path)
    command = ''
    # QEMU_PATH = '/home/jingwen/Work/qemu-install/qemu4_install/bin/qemu-system-riscv32'
    kernel_name = '../output/kernel_all.elf'
    ifm_name = ''
    wt_name = ''
    paramTable_name = ''
    # ifm_qemu_addr = '0xc0000000'
    # wt_qemu_addr = '0xc0100000'
    paramTable_qemu_addr = ''
    for file in files:
        ifm_Obj = re.findall(r'ifm.+$', file, re.M|re.I)
        wt_Obj = re.findall(r'wt.+$', file, re.M|re.I)
        paramTable_Obj = re.findall(r'ParamTable.+$', file, re.M|re.I)
        paramTable_qemu_addr_Obj = re.findall(r'ParamTable[\S]*local_(.+?)_', file, re.M|re.I)

        if ifm_Obj:
            ifm_name = ifm_Obj[0]
            # print(ifm_name)
            
        if wt_Obj:
            wt_name = wt_Obj[0]
            # print(wt_name)
        
        if paramTable_Obj:
            paramTable_name = paramTable_Obj[0]
            # print(paramTable_name)

        if paramTable_qemu_addr_Obj:
            paramTable_qemu_addr = paramTable_qemu_addr_Obj[0]
            # print(paramTable_qemu_addr)

    command = config.get_QEMU_PATH()    + ' -machine sifive_e -nographic -kernel ' + kernel_name \
                                        + ' -device loader,file=' +  path + ifm_name + ',addr=' + config.get_ifm_qemu_addr() \
                                        + ' -device loader,file=' +  path + wt_name + ',addr=' + config.get_wt_qemu_addr() \
                                        + ' -device loader,file=' + path + paramTable_name + ',addr=' + paramTable_qemu_addr

    #print(command)
    p=subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line)
    # retval = p.wait()
