/usr/bin/env

set -x

if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi


cp -r patches/* $1