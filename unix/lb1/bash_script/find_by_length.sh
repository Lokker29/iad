#!/bin/bash

if [ ! -d "$1" ]
then
  echo "$1 directory doesn't exist"
  exit 1
fi

if [[ ! "$2" =~ ^[0-9]+$ ]] || [[ "$2" -le 0 ]]
then
  echo "$2 is an invalid integer"
  exit 1
fi

find "$(realpath "$1")" -type f -regextype posix-extended -regex ".*/[^/]{$2,}" | sort | tee result.txt
