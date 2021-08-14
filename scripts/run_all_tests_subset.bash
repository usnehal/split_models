set -e
set -x

SERVER=" -s 34.90.41.208"

if [ -z "$1" ]
then
  echo "no result folder"
  RESULT_FOLDER="temp"
else
  RESULT_FOLDER=$1
fi

TESTS=" -m 100"
VERBOSE=" -v 0 -r $RESULT_FOLDER "

cd ~/WorkSpace/

python3 ./client.py $SERVER -t 2 $VERBOSE $TESTS --image_size 250 
python3 ./client.py $SERVER -t 3 $VERBOSE $TESTS --image_size 250 
python3 ./client.py $SERVER -t 4 $VERBOSE $TESTS --image_size 250 

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


python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 3
python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 40
python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 63
python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 86
python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 100
python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 132
python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 164
python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 196
python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 228
python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 248
python3 ./client.py $SERVER -t 7 $VERBOSE $TESTS --image_size 250 --split_layer 279


python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 3
python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 40
python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 63
python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 86
python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 100
python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 132
python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 164
python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 196
python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 228
python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 248
python3 ./client.py $SERVER -t 8 $VERBOSE $TESTS --image_size 250 --split_layer 279

