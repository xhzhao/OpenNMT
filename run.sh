#!/bin/bash
source ~/.bashrc
th train.lua -data data/demo-train.t7 -save_model model -profiler true

#~/intel/vtune_amplifier_xe/bin64/amplxe-cl -collect hotspots -knob analyze-openmp=true -knob sampling-interval=30 --resume-after 8 -d 50  th train.lua -data data/demo-train.t7 -save_model model -profiler true

