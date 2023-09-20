from itertools import islice
import pandas as pd
from youtube_comment_downloader import *
downloader = YoutubeCommentDownloader()
comments = downloader.get_comments_from_url('https://www.youtube.com/watch?v=TcE14c60BtI', sort_by=SORT_BY_POPULAR)

dataframe = pd.DataFrame(comments)
df = pd.DataFrame(dataframe['text'])


from textblob import TextBlob

df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment)
df['polarity'] = df['sentiment'].apply(lambda x: x[0])
df['subjectivity'] = df['sentiment'].apply(lambda x: x[1])

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score==0:
        return 'Neutral'
    else:
        return 'Positive'
    
df['exact'] = df['polarity'].apply(getAnalysis)
output = df.groupby('exact').count()
output.drop('text', axis=1, inplace=True)
output.drop('subjectivity', axis=1, inplace=True)
output.drop('polarity',axis=1, inplace=True)

print(output)


