# PiNexusOS

PiNexusOS is an advanced operating system for Raspberry Pi, designed to be lightweight, fast, and flexible. It is built from scratch using a combination of assembly, C, and C++ code.

# Features

1. Custom kernel and file system
2. Support for device drivers
3. Interrupt handling and process management
4. Memory management and system calls
5. Math and string libraries
6. Testing framework for kernel and file system

# Getting Started

To get started with PiNexusOS, follow these steps:

1. Install the Raspberry Pi OS on your Raspberry Pi.
2. Clone the PiNexusOS repository:

```bash

1. git clone https://github.com/KOSASIH/pi-nexus-autonomous-banking-network.git
```

3. Build the PiNexusOS kernel and file system:

```bash

1. cd pi-nexus-autonomous-banking-network/PiNexusOS/
2. make
```

4. Copy the PiNexusOS kernel and file system to your Raspberry Pi:

```bash

1. sudo cp kernel.img /boot/
2. sudo cp fs.img /boot/
```

5. Reboot your Raspberry Pi to load the PiNexusOS kernel and file system.

# License

PiNexusOS is released under the MIT License. See the LICENSE file for more information.

# Acknowledgments

PiNexusOS was created by [KOSASIH](https://www.linkedin.com/in/kosasih-81b46b5a). Thanks to the Raspberry Pi community for their support and inspiration.
