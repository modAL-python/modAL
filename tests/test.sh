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

