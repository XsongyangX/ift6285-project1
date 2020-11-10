#!/bin/bash

# Removes training and testing temp files
rm data/models -r
rm data/predictions -r

if [ "$1" == "--vectorizers" ]; then
    rm data/vectorizers -r
fi