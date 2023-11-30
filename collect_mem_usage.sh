watch -t -n 1 "echo \$(date '+%Y-%m-%d %H:%M:%S') \$(sudo xpu-smi stats -d 0 | grep 'GPU Memory Used') | tee -a memory.log"
