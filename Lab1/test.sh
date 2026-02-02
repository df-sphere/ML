#!/bin/sh

conda activate cs7643-a1
python3 -m unittest tests.test_loading tests.test_activation tests.test_loss tests.test_network tests.test_training

