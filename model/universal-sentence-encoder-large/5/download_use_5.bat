@echo off
set url=https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed
set file=universal-sentence-encoder-large_5.tar.gz
set folder=uncompressed_folder

curl -L -o %file% %url%
tar -xzf %file%
del %file%