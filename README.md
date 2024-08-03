## Test Environment

The test environment is as follows:

- Vivado, Vitis, PetaLinux: 2023.1
- Vitis AI: 3.5

The evaluation board used is KV260.

## Creating a PetaLinux Image for KV260

The BSP file for building is available at the link below. Download it and run PetaLinux:

[Download BSP File](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/1641152513/Kria+K26+SOM#PetaLinux-Board-Support-Packages)

```bash
petalinux-create -t project -s xilinx-kv260-starterkit-v2023.1-05080224.bsp 
cd xilinx-kv260-starterkit-2023.1/
petalinux-build
petalinux-package --boot --u-boot --force
petalinux-package --wic --images-dir images/linux/ --bootfiles "ramdisk.cpio.gz.u-boot,boot.scr,Image,system.dtb,system-zynqmp-sck-kv-g-revB.dtb" --disk-name "mmcblk1"
```

Write the created SD image (`petalinux-sdimage.wic`) using balenaEtcher or a similar tool. It can be found at:

`/xilinx-kv260-starterkit-2023.1/images/linux`

## Initial Boot on KV260

The initial boot on KV260 is the same as when tested with C++. The Vitis AI runtime was similarly installed on KV260:

```bash
sudo dnf install xrt packagegroup-petalinux-opencv
wget https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-3.5.0.tar.gz -O vitis-ai-runtime-3.5.0.tar.gz
tar -xzvf vitis-ai-runtime-3.5.0.tar.gz
cd vitis-ai-runtime-3.5.0/2023.1/aarch64/centos/
sudo bash ./setup.sh
```

## DPU Configuration

The DPU is initially configured to be loaded with `xmutil`.

An application named `b4096_300m` was created. The method for creating the necessary files.

```bash
cd sd_aiedge_4096/
sudo mkdir /lib/firmware/xilinx/b4096_300m
sudo cp pl.dtbo shell.json /lib/firmware/xilinx/b4096_300m/
sudo cp dpu.xclbin /lib/firmware/xilinx/b4096_300m/binary_container_1.bin
ls /lib/firmware/xilinx/b4096_300m/
```

Also, replace with the newly created `vart.conf`. It is recommended to reboot:

```bash
sudo mv /etc/vart.conf /etc/old_vart.conf
sudo cp vart.conf /etc/
sudo reboot
```

## Running the Demo

From here, the steps follow the demo video introduced at the beginning:

```bash
sudo xmutil listapps
sudo xmutil unloadapp
sudo xmutil loadapp b4096_300m
cd pytorch_test/
python pt-yolox.py
```

## Object Detection with PyTorch

```plaintext
bboxes of detected objects: [[ 458.11553955  125.8078537   821.88452148  489.57681274]
 [  40.24644089    0.         1239.75366211  720.        ]]
scores of detected objects: [0.56179011 0.11786249]
Details of detected objects: [49. 60.]
Pre-processing time: 0.0087 seconds
DPU execution time: 0.0115 seconds
Post-process time: 0.0330 seconds
Total run time: 0.0532 seconds
Performance: 18.780998267115038 FPS
```







## Installation and Setup for ONNX Runtime with Vitis AI

This guide will help you set up the environment necessary to run ONNX Runtime using Vitis AI.

### Step 1: Download and Install Required Packages

First, download and extract the Vitis AI package:

```bash
wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2023.1-r3.5.0.tar.gz
sudo tar -xzvf openDownload\?filename\=vitis_ai_2023.1-r3.5.0.tar.gz -C /
```

Next, download and install the VOE and ONNX Runtime Vitis AI Python wheels:

```bash
wget https://www.xilinx.com/bin/public/openDownload?filename=voe-0.1.0-py3-none-any.whl -O voe-0.1.0-py3-none-any.whl
pip3 install voe*.whl
wget https://www.xilinx.com/bin/public/openDownload?filename=onnxruntime_vitisai-1.16.0-py3-none-any.whl -O onnxruntime_vitisai-1.16.0-py3-none-any.whl
pip3 install onnxruntime_vitisai*.whl
```

### Step 2: Reboot and Run the Program

After installation, reboot your system and start the DPU using `xmutil` before running the program.

```bash
sudo reboot
sudo xmutil listapps
sudo xmutil unloadapp
sudo xmutil loadapp b4096_300m
cd onnx-test/
python onnx-yolox.py
```

### Issue: Running in CPU Mode

Initially, the program may run in CPU mode, leading to slower inference times for YOLOX (approximately 300ms). This issue was due to the Vitis AI library overwriting the default settings in `/etc/vart.conf`.

### Solution: Restore `vart.conf`

Restore the `vart.conf` file to its prepared state, then reboot the system.

```bash
cd ..
cd sd_aiedge_4096/
sudo mv /etc/vart.conf /etc/old_vart.conf
sudo cp vart.conf /etc/
sudo reboot
```

### Step 3: Run the Program in DPU Mode

After rebooting, start the DPU again and run the program to achieve faster inference times (~10ms).

```bash
sudo xmutil listapps
sudo xmutil unloadapp
sudo xmutil loadapp b4096_300m
cd onnx-test/
python onnx-yolox.py
```



