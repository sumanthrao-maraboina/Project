#!/usr/bin/env python
# coding: utf-8

# In[21]:


import requests
from IPython.display import JSON
import pandas as pd
import numpy as np 
import time

# Visualization
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px

# Panel/hvplot (holoviz)
import panel as pn
pn.extension()
import param
import hvplot.pandas

# Others
import pickle
from io import StringIO


# In[22]:


# API secret
API_KEY = "ed08cae3b2ed48d18fdcf0857bf5ec96"


# ## Get audio transcription using AssemblyAI

# In[23]:


# Submitting Files for Transcription
import requests
endpoint = "https://api.assemblyai.com/v2/transcript"
json = {
    "audio_url": "https://github.com/thu-vu92/audio_analyzer_assemblyai/blob/main/How_I_Would_Learn_to_Code.mp3?raw=true",
    "auto_highlights": True,
    "sentiment_analysis": True,
    "auto_chapters": True,
    "iab_categories": True,
}
headers = {
    "authorization": API_KEY,
    "content-type": "application/json"
}
response = requests.post(endpoint, json=json, headers=headers)
print(response.json())


# In[24]:


# Getting the Transcription Result
result_endpoint = endpoint + "/" + response.json()["id"]
headers_auth = {
    "authorization": API_KEY,
}
transcript_response = requests.get(result_endpoint, headers=headers_auth)
print(transcript_response.json())

# While loop for requesting transcription
while response.json()['status'] != "completed":
    response = requests.get(result_endpoint, headers=headers_auth)
    time.sleep(3)


# In[25]:


JSON(transcript_response.json())


# In[26]:


# Save pickle
with open('speech_data.pkl', 'wb') as f:
    pickle.dump(transcript_response.json().copy(), f)


# ## Dashboard components

# In[27]:


# Load data pickle
with open('speech_data.pkl', 'rb') as f:
    data = pickle.load(f)


# In[28]:


data["text"]


# ### 0) Download transcript widget

# In[29]:


buffer = StringIO()
buffer.write(data["text"])
buffer.seek(0)


# In[13]:


buffer = StringIO()
if data["text"] is not None:
    buffer.write(str(data["text"]))
buffer.seek(0)


# In[14]:


transcript_download = pn.widgets.FileDownload(file=buffer, 
                                              filename="transcript.txt", 
                                              button_type='success')
transcript_download


# ### 00) Audio play

# In[15]:


audio_url = "https://github.com/thu-vu92/audio_analyzer_assemblyai/blob/main/How_I_Would_Learn_to_Code.mp3?raw=true"
audio_play = pn.pane.Audio(audio_url, name='Audio', time = 360)
audio_play


# ### 1) Sentiment plot

# In[16]:


sentiment = data["sentiment_analysis_results"]


# In[17]:


sentiment_df = pd.DataFrame(sentiment)
sentiment_df


# In[18]:


sentiment_df_grouped = sentiment_df['sentiment'].value_counts()
sentiment_df_grouped


# In[95]:


# Bar plot
sentiment_plot = sentiment_df_grouped.hvplot(title = "Sentences by Sentiment Category", kind="bar")
pn.Row(sentiment_plot)


# In[96]:


positive_df = sentiment_df[sentiment_df["sentiment"] == "POSITIVE"][["text", "sentiment"]]
negative_df = sentiment_df[sentiment_df["sentiment"] == "NEGATIVE"][["text", "sentiment"]]
neutral_df = sentiment_df[sentiment_df["sentiment"] == "NEUTRAL"][["text", "sentiment"]]

sentiment_tabs = pn.Tabs(('Sentiment overview', sentiment_plot), 
                       ('Positive', pn.widgets.DataFrame(positive_df, autosize_mode='fit_columns', width=700, height=300)),
                       ('Negative', pn.widgets.DataFrame(negative_df, autosize_mode='fit_columns', width=700, height=300)),
                       ('Neutral', pn.widgets.DataFrame(neutral_df, autosize_mode='fit_columns', width=700, height=300))
                        )
sentiment_tabs


# ### 2) Word cloud

# In[97]:


stopwords = set(STOPWORDS)


# In[98]:


transcript = data["text"]


# In[99]:


transcript_lower = [item.lower() for item in str(transcript).split()]
transcript_lower


# In[100]:


all_words = ' '.join(transcript_lower) 
all_words


# In[101]:


# Word cloud plot
wordcloud = WordCloud(background_color='black', stopwords = stopwords, max_words = 20,
                      colormap='viridis', collocations=False).generate(all_words)

wordcloud_plot = px.imshow(wordcloud) 
# Remove labels on axes
wordcloud_plot.update_xaxes(showticklabels=False)
wordcloud_plot.update_yaxes(showticklabels=False)
wordcloud_plot


# In[102]:


# Create interactive slider
class Controller(param.Parameterized):
    word_slider = param.Integer(30, bounds=(5, 50), step=5)

controller = Controller()

@pn.depends(controller.param.word_slider, watch=True)
def update_wordcloud(num_words):
    # Word cloud plot
    wordcloud = WordCloud(background_color='black', stopwords = stopwords, max_words = num_words,
                          colormap='viridis', collocations=False).generate(all_words)

    wordcloud_plot = px.imshow(wordcloud) 
    # Remove labels on axes
    wordcloud_plot.update_xaxes(showticklabels=False)
    wordcloud_plot.update_yaxes(showticklabels=False)
    return wordcloud_plot


# ### 3) Auto chapter summary

# In[103]:


chapters = data["chapters"]
chapters


# In[104]:


chapter_summary = pn.widgets.StaticText(value=chapters[0]["summary"], 
                                        width=1000, 
                                        height_policy = "fit")
chapter_summary


# In[105]:


button = pn.widgets.Button(name=str(int(chapters[0]["start"]/1000)), button_type='primary')
button


# In[106]:


chapter_audio = pn.pane.Audio(audio_url, name='Audio', time = round(chapters[0]["start"]/1000))
chapter_audio


# In[107]:


# Create chapter summary layout
chapters_layout = pn.Column(pn.pane.Markdown("### Auto Chapter Summary"))

class ButtonAudio():
    def __init__(self, start_time):
        self.start_time = start_time
        self.button = pn.widgets.Button(name=str(int(self.start_time/1000)), button_type='primary', width=60)
        self.chapter_audio = pn.pane.Audio(audio_url, name='Audio', time = round(self.start_time/1000))
        self.button.on_click(self.move_audio_head)

    def move_audio_head(self, event):
        self.chapter_audio.time = self.start_time/1000
        
for chapter in chapters:
    chapter_summary = pn.widgets.StaticText(value=chapter["summary"], width=1000, height_policy = "fit")
    button_audio = ButtonAudio(chapter["start"])
    button = button_audio.button
    chapter_audio = button_audio.chapter_audio
    chapters_layout.append(pn.Row(pn.Column(button), pn.Column(chapter_audio), pn.Column(chapter_summary)))
    
chapters_layout


# ### 4) Auto highlights

# In[108]:


highlights = data["auto_highlights_result"]["results"]
highlights_df = pd.DataFrame(highlights)
highlights_df


# ## Dashboard

# In[109]:


# Dashboard template
template = pn.template.FastListTemplate(
    title='Audio Content Explorer', 
    sidebar=[pn.pane.Markdown("# Explore audio content"), 
             pn.pane.Markdown("#### This app analyzes the content of your audio file, including sentiment, wordcloud, automatic content summary and highlights using AssemblyAI API."),
             pn.pane.Markdown("#### This example is based on the audio content of Ken Jee's Youtube video on how to learn to code."),
             pn.pane.PNG("kenjee_thumbnail.png", sizing_mode="scale_both"),
             pn.pane.Markdown("### [Link to video!](https://www.youtube.com/watch?v=EBjYqC3aNTA&t=311s)"),
             pn.pane.Markdown("### Download transcript:"),
             transcript_download
             ],
    main=[pn.Row(pn.Column(sentiment_tabs), pn.Column(pn.Row(controller.param.word_slider), 
                                                      pn.Row(update_wordcloud, title = "WordCloud of Speech Content"))
                ),
          pn.Row(chapters_layout),
          pn.Row(highlights_plot, title = "Automatic Highlights")],
    accent_base_color="#88d8b0",
    header_background="#c0b9dd",
)

template.show()


# In[111]:


# Get requirements
get_ipython().system('pip3 freeze > requirements.txt')

