set -e
set -x

if [ -z "$1" ]
then
  echo "no server ip"
  SERVER=" "
else
  #IP=35.200.232.85
  IP=$1
  SERVER=" -s $IP"
fi

if [ -z "$2" ]
then
  echo "no result folder"
  RESULT_FOLDER="temp"
else
  RESULT_FOLDER=$2
fi

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
TESTS=" -m 100"
VERBOSE=" -v 0 -r $DATE_WITH_TIME "

cd ~/WorkSpace/
rm -rf ./temp/results/results.csv 

python3 ./client.py $SERVER -t 2 $VERBOSE $TESTS
python3 ./client.py $SERVER -t 3 $VERBOSE $TESTS
python3 ./client.py $SERVER -t 4 $VERBOSE $TESTS

python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 3
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 40
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 63
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 86
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 100
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 132
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 164
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 196
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 228
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 248
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 100 --split_layer 279

python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 3
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 40
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 63
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 86
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 100
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 132
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 164
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 196
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 228
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 248
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 150 --split_layer 279

python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 3
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 40
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 63
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 86
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 100
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 132
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 164
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 196
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 228
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 248
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 200 --split_layer 279

python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 3
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 40
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 63
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 86
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 100
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 132
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 164
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 196
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 228
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 248
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 250 --split_layer 279

python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 3
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 40
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 63
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 86
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 100
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 132
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 164
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 196
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 228
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 248
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 300 --split_layer 279

python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 3
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 40
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 63
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 86
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 100
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 132
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 164
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 196
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 228
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 248
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 350 --split_layer 279

python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 3
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 40
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 63
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 86
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 100
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 132
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 164
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 196
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 228
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 248
python3 ./client.py $SERVER -t 5 $VERBOSE $TESTS --image_size 400 --split_layer 279

python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 3
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 40
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 63
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 86
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 100
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 132
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 164
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 196
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 228
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 248
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 100 --split_layer 279

python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 3
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 40
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 63
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 86
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 100
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 132
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 164
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 196
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 228
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 248
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 150 --split_layer 279

python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 3
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 40
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 63
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 86
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 100
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 132
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 164
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 196
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 228
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 248
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 200 --split_layer 279

python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 3
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 40
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 63
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 86
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 100
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 132
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 164
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 196
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 228
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 248
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 250 --split_layer 279

python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 3
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 40
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 63
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 86
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 100
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 132
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 164
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 196
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 228
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 248
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 300 --split_layer 279

python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 3
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 40
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 63
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 86
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 100
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 132
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 164
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 196
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 228
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 248
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 350 --split_layer 279

python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 3
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 40
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 63
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 86
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 100
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 132
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 164
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 196
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 228
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 248
python3 ./client.py $SERVER -t 6 $VERBOSE $TESTS --image_size 400 --split_layer 279

python3 ./client.py $SERVER -t 1 $VERBOSE $TESTS --image_size 100
python3 ./client.py $SERVER -t 1 $VERBOSE $TESTS --image_size 150
python3 ./client.py $SERVER -t 1 $VERBOSE $TESTS --image_size 200
python3 ./client.py $SERVER -t 1 $VERBOSE $TESTS --image_size 250
python3 ./client.py $SERVER -t 1 $VERBOSE $TESTS --image_size 300
python3 ./client.py $SERVER -t 1 $VERBOSE $TESTS --image_size 350
python3 ./client.py $SERVER -t 1 $VERBOSE $TESTS --image_size 400
