#!/bin/bash

echo executing core tests ...
python3 core_tests.py
exit_code=$?
if [ $exit_code -eq 0 ]; then
echo core_tests.py successfully executed
else
echo core_tests.py failed
exit 1
fi

for example_test in example_tests/*
do
  echo executing $example_test ...
  python3 $example_test
  exit_code=$?
  if [ $exit_code -eq 0 ]; then
  echo $example_test successfully executed
  else
  echo $example_test failed
  exit 1
  fi
done

