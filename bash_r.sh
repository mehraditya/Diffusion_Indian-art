#!/bin/bash

source = 
extract = 
destination = 

7z  x "$source" -o "$extract"
if [ $? -eq 0 ]; then
	cp -R "$extract" "destination"

	echo "DONZO"
else
	echo "error"