set -e

download_all=0
while getopts "a" opt; do
    case "$opt" in
    a)  download_all=1
    esac
done

echo "Finalizing Build!"
echo "#################"

echo "Generate personal access token at https://github.com/settings/tokens"
echo "In the website, select repo for the scope."
echo "Paste your personal access token, followed by [ENTER]:"
read token
git config --global url."https://$token:@github.com/".insteadOf "https://github.com/"

echo "Type your GitHub username, followed by [ENTER]":
read github_username
git config --global user.name "$github_username"

echo "Type your GitHub email, followed by [ENTER]":
read github_email
git config --global user.email "$github_email"

echo "Paste your AWS access key (for your IAM user on Daniel's account), followed by [ENTER]"
read aws_access

echo "Paste your AWS secret key (for your IAM user on Daniel's account), followed by [ENTER]"
read aws_secret

mkdir ~/.aws

printf "[default]\nregion = us-west-1" > ~/.aws/config

printf "[default]\naws_access_key_id = %s\naws_secret_access_key = %s" $aws_access $aws_secret > ~/.aws/credentials

echo "Cloning repo..."
git clone https://github.com/HumanCompatibleAI/nn_clustering.git

cd nn_clustering

git checkout master

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
