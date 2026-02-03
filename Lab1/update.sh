#!/bin/sh

mkdir assignment_original
cp assignment1.zip assignment_original/
cd assignment_original
unzip assignment1.zip
cd ..
rm -rf configs
mv assignment_original/configs .
rm -rf assignment_original
echo "Configs folder has been updated."
