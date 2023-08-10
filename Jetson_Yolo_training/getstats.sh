#!/bin/bash

export S_TIME_FORMAT=ISO
date_str=$(date)
s_timestamp=$(date -d "${date_str}" +"%s")

while true ; do
  stat=$(iostat -xcdt)
  date_str=$(echo "$stat" | grep -B 1 'avg-cpu' | head -n 1)
  timestamp=$(date -d "${date_str}" +"%s")
  short_timestamp=$(calc $timestamp - $s_timestamp)

  mpstat_line=$(mpstat 1 1 | tail -n 1)
  cpu_idle=$(echo $mpstat_line | cut -d' ' -f12)
  cpu_iowait=$(echo $mpstat_line | cut -d' ' -f6)
  cpu_usage=$(calc 100.0-$cpu_idle)
  gpu_usage=$(tegrastats | head -n 1 |xargs | cut -d' ' -f16)

  stats="$timestamp,$short_timestamp,$cpu_idle,$cpu_usage,$cpu_iowait,$gpu_usage"
  echo $stats
done
