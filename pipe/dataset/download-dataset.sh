#!/bin/bash
curl -L -o ./toxic_comments.zip https://www.kaggle.com/api/v1/datasets/download/rounak02/imported-data

unzip ./toxic_comments.zip -d ./
mv ./train.csv ./toxic_comment.csv
rm ./toxic_comments.zip ./wiki-news-300d-1M.vec 

#If doesn't work try the Kaggle Instructions in the dataset link