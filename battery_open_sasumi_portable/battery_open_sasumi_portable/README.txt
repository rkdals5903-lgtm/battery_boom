Battery Open Sasumi portable bundle
====================================

Requirements
------------
- Linux PC with an NVIDIA GPU and compatible NVIDIA driver
- NVIDIA Isaac Sim 5.1

Run
---
From this extracted directory, run:

  <ISAAC_SIM>/python.sh batteryfactory/battery_open_sasumi.py

Example:

  /path/to/isaacsim/python.sh batteryfactory/battery_open_sasumi.py

Notes
-----
- Keep the batteryfactory and M0609 directories together.
- The Python code uses paths relative to this extracted directory.
- The collected USD directory contains the referenced scene assets.
- Generated RMPFlow files are written inside batteryfactory when the script starts.
- Set SASUMI_KEEP_GUI_OPEN=0 to close Isaac Sim automatically after completion.

