set -e

download_all=0
while getopts "a" opt; do
    case "$opt" in
    a)  download_all=1
    esac
done

echo "Finalizing Build!"
echo "#################"

echo "Cloning repo..."
git clone https://github.com/dfilan/clusterability_in_neural_networks.git

cd clusterability_in_neural_networks

git checkout main

echo "Installing Python dependencies..."
# PIP_USER=yes  # install as --user
pipenv install --system

echo "Patching scipy and sklearn..."
python -m site | grep /usr/local/lib/python3.7/dist-packages || { echo "Cannot find dist-packages directory for patching of scipy and sklearn"; exit 1; }
cp -r devops/patches/* /usr/local/lib/python3.7/dist-packages


echo "Downloading results folder"
aws s3 cp --recursive s3://nn-clustering/results results

if [ $download_all -gt 0 ]
then
    echo "Downloading models folder, including checkpoints. Hold tight, this might take a while"
    aws s3 cp --recursive s3://nn-clustering/models models
else
    echo "Downloading models folder, excluding checkpoints"
    aws s3 cp --recursive --exclude "*.ckpt" s3://nn-clustering/models models

fi


echo "Downloading datasets folder"
aws s3 cp --recursive s3://nn-clustering/datasets datasets

cd ..

echo "Deleting this script - you can find it in the repo!"
rm -- "$0"
