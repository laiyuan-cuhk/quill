#!/bin/sh

mkdir ../data/stdlib-good ../data/stdlib-bad

for i in ../data/stdlib-json/*.json;do
 mv "$i" ../data/stdlib/
 file=$(realpath "../data/stdlib/$(basename "$i")")
 python3 preprocess.py
 if [ $? -eq 0 ];then
  mv "$file" ../data/stdlib-good/
 else
  mv "$file" ../data/stdlib-bad/
 fi
done
