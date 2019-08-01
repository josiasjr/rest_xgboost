#!/usr/bin/env bash

file="test.json"
while true; do
	while IFS= read -r line; do
		curl -d $line -H "Content-Type: application/json" -X POST http://localhost:8080/calculator/predict
#		curl -d $line -H "Content-Type: application/json" -X POST https://{appid}.appspot.com/calculator/predict
		echo
		sleep 1
	done <"$file"
done

#file="test.json"
#(
#	while true; do
#		while IFS= read -r line; do
#			printf '%s\n' "$line"
#			sleep 0.1
#		done <"$file"
#	done
#) | curl -d '{"key1":"value1", "key2":"value2"}' -H "Content-Type: application/json" -X POST http://localhost:8080/calculator/predict
#curl -d '{"key1":"value1", "key2":"value2"}' -H "Content-Type: application/json" -X POST http://localhost:3000/data

