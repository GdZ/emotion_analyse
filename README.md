# emotion_analyse

this is fork from [Twitter Emotion](https://github.com/cnnjqzr/NLP_Lab)

and have been restructure and developed


## How to run this programm

`make`

## How to clean created files

`make clean`


# Twitter Emotion:
## First Step: Evaluation
- We use F-score and accuracy to evaluate the predicted data and the gold data
- dev-predicted.csv stores predicted data(predicted emotion)
- dev.txt stores gold data(right emotion)
- Evaluation.py is the code of evaluation tool, we can get accuracy and f1-score of each emotion after runnning it

## We cleaned the data set that was provided by the tutor
- we added an x to empty columns because blank spaces were not recognised as columns, meaning that different information was in different columns on different rows.
- we removed columns that we thought were unnecessary such as the phone number of the sender. We left the emotion tag in column 1, the language (en, de, es etc) and the tweet content.
- we filtered the tweets so that we left only the English language ones, based on the language indicator. It is possible that foreign words may be contained in some tweets but the main language is English.
- We removed the language column, so we are left with the emotion hashtag and remaining tweet content.

## we created a stop word list
- we found one from the internet and made a new file.
