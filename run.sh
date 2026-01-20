
mkdir $1
cd $1
cp ../train.py .
cp -r ../lib .
cp -r ../misc .
python train.py
