#!/bin/bash

echo executing core tests ...
python3 tests/core_tests.py
exit_code=$?
if [ $exit_code -eq 0 ]; then
echo core tests successfully executed
else
echo core tests failed
exit 1
fi

for example_test in tests/example_tests/*
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

