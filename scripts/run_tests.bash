print_red () {
	echo -e "\e[1;31m$1\e[0m"
}

print_green () {
	echo -e "\e[1;34m$1\e[0m"
}


set -e
set -x

cd ~/WorkSpace/

print_red "1 Mbps speed"
sudo wondershaper eth0 1000 1000
~/WorkSpace/scripts/run_all_tests_subset.bash 1_mbps

print_red "3 Mbps speed"
sudo wondershaper eth0 3000 3000
~/WorkSpace/scripts/run_all_tests_subset.bash 3_mbps

print_red "5 Mbps speed"
sudo wondershaper eth0 5000 5000
~/WorkSpace/scripts/run_all_tests_subset.bash 5_mbps

print_red "7 Mbps speed"
sudo wondershaper eth0 7000 7000
~/WorkSpace/scripts/run_all_tests_subset.bash 7_mbps

print_red "10 Mbps speed"
sudo wondershaper eth0 10000 10000
~/WorkSpace/scripts/run_all_tests_subset.bash 10_mbps

print_red "15 Mbps speed"
sudo wondershaper eth0 15000 15000
~/WorkSpace/scripts/run_all_tests_subset.bash 15_mbps

print_red "20 Mbps speed"
sudo wondershaper eth0 20000 20000
~/WorkSpace/scripts/run_all_tests_subset.bash 1_mbps

print_red "30 Mbps speed"
sudo wondershaper eth0 30000 30000
~/WorkSpace/scripts/run_all_tests_subset.bash 30_mbps

print_red "50 Mbps speed"
sudo wondershaper eth0 50000 1000
~/WorkSpace/scripts/run_all_tests_subset.bash 50_mbps

