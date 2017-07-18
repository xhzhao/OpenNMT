#!/bin/sh
logName=$1

grep "ori_time" $logName  | awk '{print $1,$2}' &> all_type.log
sort all_type.log | uniq > type.log
