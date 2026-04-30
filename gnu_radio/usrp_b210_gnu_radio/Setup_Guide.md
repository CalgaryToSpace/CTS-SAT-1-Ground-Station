# USRP B210 GNU Radio Setup Guide

## Part 1: USRP B210 UHD Drivers

To install the USRP B200 driver on Ubuntu, add the Ettus Research PPA, update your package list, and install libuhd-dev and uhd-host, then download device firmware using
uhd_images_downloader.py, and finally configure USB permissions with udev rules for non-root access, all before connecting your B200 device and verifying with uhd_find_devices.


```bash
# Add the UHD PPA (by Ettus Research).
sudo add-apt-repository ppa:ettusresearch/uhd
sudo apt-get update

# Install.
sudo apt-get install libuhd-dev uhd-host

# Download USRP Firmware Images (FPGA bitstreams).
# The B200 loads its firmware from files, which you can download with this script.
sudo /usr/lib/uhd/utils/uhd_images_downloader.py
# -- or --
sudo uhd_images_downloader


# Configure USB Permissions (udev rules)
# This step allows users to access the USRP without sudo.
cd <install-path>/lib/uhd/utils # Often /usr/lib/uhd/utils
sudo cp uhd-usrp.rules /etc/udev/rules.d/
# -- or --
sudo cp /usr/lib/udev/rules.d/60-uhd-host.rules /etc/udev/rules.d/
# Activate rules:
sudo udevadm control --reload-rules
sudo udevadm trigger

# Note: You might need to log out and back in or reboot for udev changes to fully apply, though often a trigger is enough.
```

## Part 2: Setup `UHD_IMAGES_DIR` Environment Variable

**Disclaimer:** This section isn't fully reviewed yet.

Configure your bash/cli to be able to use the firmware images
add the following to the end of your bashrc file (depending if you used default install or custom):
```
        export UHD_IMAGES_DIR="/usr/share/uhd/4.9.0/images"
    OR
        export UHD_IMAGES_DIR="/usr/local/share/uhd/images"
```

`nano ~/.bashrc`

Add the following to the end of your `~/.bashrc` file:

```
        export UHD_IMAGES_DIR="/usr/share/uhd/4.9.0/images"
    OR
        export UHD_IMAGES_DIR="/usr/local/share/uhd/images"
```

Then close and re-open your terminal to apply the changes.

## Part 3: Connect Hardware & Verify

1. Connect your B200 to a USB 3.0 port on your computer and verify with uhd_find_devices command.
2. Run the `uhd_find_devices` command.

You should get something like this in response:
```

[INFO] [UHD] linux; GNU C++ version 13.3.0; Boost_108300; UHD_4.9.0.0-0ubuntu1~noble3
--------------------------------------------------
-- UHD Device 0
--------------------------------------------------
Device Address:
    serial: 315F298
    name: MyB210
    product: B210
    type: b200

```

## Part 4: Python Setup

Python dependencies:

```
pip install xtea
pip install crc
pip install reed_solomon_ccsds
```

Copy files from git repo into this folder and then follow the instructions for getting the transciever up and running
