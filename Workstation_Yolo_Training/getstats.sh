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

  stats="$timestamp,$short_timestamp,$cpu_idle,$cpu_usage,$cpu_iowait"

  gpu_stat=$(nvidia-smi dmon -c 1 | grep -v '\#')
  num_gpus=$(echo "$gpu_stat" | wc -l)
  for n in $(seq 1 "$num_gpus") ; do
    line=$(echo "$gpu_stat" | sed "${n}q;d")
    gpu_sm=$(echo $line | xargs | cut -d' ' -f5)
    gpu_enc=$(echo $line | xargs | cut -d' ' -f6)
    gpu_mem_stats=$(nvidia-smi -i $((n-1)) --query-gpu=memory.total,memory.free,memory.used --format=csv | tail -n1)
    gpu_mem_total=$(echo "$gpu_mem_stats" | xargs | cut -d',' -f1 | xargs | cut -d' ' -f1)
    gpu_mem_free=$(echo "$gpu_mem_stats" | xargs | cut -d',' -f2 | xargs | cut -d' ' -f1)
    gpu_mem_used=$(echo "$gpu_mem_stats" | xargs | cut -d',' -f3 | xargs | cut -d' ' -f1)
    stats="$stats,$gpu_sm,$gpu_enc,$gpu_mem_free,$gpu_mem_used,$gpu_mem_total"
  done

  echo $stats
done