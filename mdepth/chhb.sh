#!/bin/bash
# shellcheck disable=SC2086
set -eE
set -o pipefail
set -x

#==============================================================================
HHB_OUT_DIR="npu_model"
# Compile the input file
RISCV_INCLUDES="-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/x86/include/ \
-I /usr/local/lib/python3.8/dist-packages/tvm/dlpack/include/ \
-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/x86/include/shl_public/ \
-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/x86/include/csinn/ \
-I /usr/local/lib/python3.8/dist-packages/tvm/include/ \
-I /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/runtime/cmd_parse \
-I ${HHB_OUT_DIR}"

riscv64-unknown-linux-gnu-gcc -O2 -g -mabi=lp64d  \
${RISCV_INCLUDES} \
${HHB_OUT_DIR}/main.c  -c -o  ${HHB_OUT_DIR}/main.o 

riscv64-unknown-linux-gnu-gcc -O2 -g -mabi=lp64d  \
${RISCV_INCLUDES} \
${HHB_OUT_DIR}/model.c  -c -o  ${HHB_OUT_DIR}/model.o 

riscv64-unknown-linux-gnu-gcc -O2 -g -mabi=lp64d  \
${RISCV_INCLUDES} \
${HHB_OUT_DIR}/jit.c  -c -o  ${HHB_OUT_DIR}/jit.o 

#==============================================================================
# Link the compiled files
RISCV_LIBS="-L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
-L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/decode/install/lib/rv \
-L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/runtime/riscv_linux \
-lshl -lprebuilt_runtime -ljpeg -lpng -lz -lstdc++ -lm"

riscv64-unknown-linux-gnu-gcc ${HHB_OUT_DIR}/model.o  ${HHB_OUT_DIR}/main.o  -o  ${HHB_OUT_DIR}/hhb_runtime  \
-Wl,--gc-sections  \
-O2 -g -mabi=lp64d  \
-Wl,-unresolved-symbols=ignore-in-shared-libs  \
${RISCV_LIBS}

riscv64-unknown-linux-gnu-gcc ${HHB_OUT_DIR}/model.o  ${HHB_OUT_DIR}/jit.o  -o  ${HHB_OUT_DIR}/hhb_jit  \
-Wl,--gc-sections  \
-O2 -g -mabi=lp64d  \
-Wl,-unresolved-symbols=ignore-in-shared-libs  \
${RISCV_LIBS}

# HHB base dir: /usr/local/lib/python3.8/dist-packages/hhb/

#==============================================================================
# x86 compilation
X86_INCLUDES="-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/x86/include/ \
-I /usr/local/lib/python3.8/dist-packages/tvm/dlpack/include/ \
-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/x86/include/shl_public/ \
-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/x86/include/csinn/ \
-I /usr/local/lib/python3.8/dist-packages/tvm/include/ \
-I /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/runtime/cmd_parse \
-I ${HHB_OUT_DIR}"

gcc -O2 -g  \
${X86_INCLUDES} \
${HHB_OUT_DIR}/main.c  -c -o  ${HHB_OUT_DIR}/main.o 

gcc -O2 -g  \
${X86_INCLUDES} \
${HHB_OUT_DIR}/model.c  -c -o  ${HHB_OUT_DIR}/model.o 

gcc -O2 -g  \
${X86_INCLUDES} \
${HHB_OUT_DIR}/jit.c  -c -o  ${HHB_OUT_DIR}/jit.o 

#==============================================================================
# Link the compiled files
X86_LIBS="-L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/pnna_x86/lib/ \
-L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/decode/install/lib/x86 \
-L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/runtime/x86_linux \
-lshl_pnna_x86 -lprebuilt_runtime -ljpeg -lpng -lz -lstdc++ -lm"

gcc ${HHB_OUT_DIR}/model.o  ${HHB_OUT_DIR}/main.o  -o  ${HHB_OUT_DIR}/hhb_th1520_x86_runtime  \
-Wl,--gc-sections  \
-O2 -g  \
-Wl,-unresolved-symbols=ignore-in-shared-libs  \
${X86_LIBS}

gcc ${HHB_OUT_DIR}/model.o  ${HHB_OUT_DIR}/jit.o  -o  ${HHB_OUT_DIR}/hhb_th1520_x86_jit  \
-Wl,--gc-sections  \
-O2 -g  \
-Wl,-unresolved-symbols=ignore-in-shared-libs  \
${X86_LIBS}

./compile.sh depth_inference.c
