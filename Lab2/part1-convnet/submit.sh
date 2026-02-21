#!/bin/sh

conda activate cs7643-a2
python3 -c "from cs7643.submit import make_a2_1_submission; make_a2_1_submission('.')"
rm -rf submit
mkdir submit
cd submit
cp ../assignment_2_1_submission.zip .
unzip assignment_2_1_submission.zip 
