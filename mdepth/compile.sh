#!/bin/bash
set -eE

InputFile=$1
OutputFile=${InputFile%.*}

echo -e "\nCompiling ${InputFile} to ${OutputFile} ...\n"

# Установка библиотеки libjpeg-dev, если её нет
# apt-get install libjpeg-dev

# Compile the input file
riscv64-unknown-linux-gnu-gcc "${InputFile}" -o "${OutputFile}" \
npu_model/io.c npu_model/model.c \
-Wl,--gc-sections -O2 -g \
-march=rv64gcv0p7_zfh_xtheadc \
-mabi=lp64d \
-I. \
-I npu_model/ \
-I ../stb/ \
-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/ \
-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/shl_public \
-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/csinn \
-L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
-L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/decode/install/lib/rv \
-L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/runtime/riscv_linux \
-lshl -lm \
-Wl,-unresolved-symbols=ignore-in-shared-libs

exit 0


riscv64-unknown-linux-gnu-gcc depth_stb.c -o depth_stb \
npu_model/model.c npu_model/io.c \
-Wl,--gc-sections -O2 -g \
-march=rv64gcv0p7_zfh_xtheadc \
-mabi=lp64d \
-I. -I./stb -I npu_model/ \
-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/ \
-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/shl_public \
-I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/csinn \
-L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
-L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/decode/install/lib/rv \
-L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/runtime/riscv_linux \
-lshl -lm \
-Wl,-unresolved-symbols=ignore-in-shared-libs


-I /usr/local/lib/python3.8/dist-packages/tvm/dlpack/include/ \
-I /usr/local/lib/python3.8/dist-packages/tvm/include/  \
-lprebuilt_runtime -ljpeg -lpng -lz -lstdc++ -lm -lshl \
