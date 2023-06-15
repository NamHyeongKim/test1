#!/bin/sh

#ctors.cpp, gasal_init_streams and buffer at bandedSWA.cpp
clear; make GPU_SM_ARCH=sm_86 MAX_QUERY_LEN=35200 N_CODE=0x4E N_PENALTY=1

#cp ./lib/libgasal.a ../bwamemcuda/libgasal.a
#cp ./lib/libgasal.a ../bwamemmulti/libgasal.a
#cp ./lib/libgasal.a ../bwamemGPU/libgasal.a
