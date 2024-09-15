import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
from textblob import TextBlob
import networkx as nx
import nltk
from nltk import word_tokenize, bigrams
nltk.download('punkt_tab')
from collections import Counter

# Load Files
alfie_data = pd.read_csv('alfie_data_processed.csv')
alfie_mask = np.array(Image.open('alfie.png'))

st.set_page_config(layout="centered",page_title='Alfie-Unscripted')
st.title('💬Alfie-Unscripted')

all_dialogues = ' '.join(alfie_data['Dialog'])
wordcloud = WordCloud(width=1400, height=700, background_color='white', mask=alfie_mask, colormap='bone').generate(all_dialogues)

st.write('I recently stumbled upon a dialogue analysis for Rick and Morty on Kaggle, and it sparked an idea. Inspired by this, I decided to dive into the words of one of my all-time favorite characters from Peaky Blinders — Alfie Solomons. Alfie, with his sharp wit and unpredictable nature, has always fascinated me.')
st.write('So, I thought, why not break down his dialogue to see what makes him so compelling? This website explores the language of Alfie Solomons, revealing the layers behind his iconic lines.')

# Visual 1
st.subheader('(1) Count of Alfie\'s Dialogues Across Episodes')
plt.figure(figsize=(14, 7))
sns.countplot(x='Season', data=alfie_data, color='#76b5c5')
plt.tick_params(axis='x', labelsize=10)
plt.ylabel('Lines of Dialogues')
plt.title('Countplot of Lines of Dialogue Across Season.Episode Number')
st.pyplot(plt)
st.write('Alfie has the most lines of dialogue in Season 3 Episode 5, with over 100 lines. This suggests that his role was important in this particular episode, possibly a pivotal moment in the storyline.')
st.write('Alfie made a significant impact right from his first appearance, which featured his third-highest number of lines of dialogue in the series. However, after Season 4, he appears in only three more episodes across Seasons 5 and 6, underscoring a narrative structure that reserves his character for specific, impactful moments rather than maintaining a constant presence throughout the series.')


# Visual 2
st.subheader('(2) Word Cloud of Alfie Solomons\' Most Used Words')
fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)
st.write('The word cloud of Alfie Solomons’ most used words describes his unique speech pattern and personality. Key terms like “f*cking,” “right,” “mate,” and “yeah” stand out prominently, reflecting Alfie’s blunt and direct nature.')
st.write('The frequent use of words like “jew” and “business” also points to the significance of his identity and relationships in the storyline. Moreover, filler words like “hmm” and “mmm” are often present, adding to his contemplative demeanor.')


# Visual 3
st.subheader('(3) Alfie\'s Dialogue Mood Distribution')
alfie_data['Sentiment'] = alfie_data['Dialog'].apply(lambda x: TextBlob(x).sentiment.polarity)

def classify_mood(sentiment):
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

alfie_data['Mood'] = alfie_data['Sentiment'].apply(classify_mood)

mood_counts = alfie_data['Mood'].value_counts()
pve_nve_nrl = mood_counts[['Positive', 'Negative', 'Neutral']]

fig, ax = plt.subplots(figsize=(14, 7))
ax.pie(pve_nve_nrl, labels=pve_nve_nrl.index, autopct='%1.1f%%',
       colors=['#76b5c5', '#ff6f61', '#cccccc'], startangle=140)
ax.set_title('Alfie Solomons: Positive vs Negative vs Neutral Mood Distribution')
st.pyplot(fig)
st.write('With 53% of his lines being neutral, it suggests that much of Alfie’s speech is delivered in a straightforward, matter-of-fact manner, without strong emotional inflection.')
st.write('However, the balance between positive (23.1%) and negative (23.9%) moods highlights the complexity of his character. Alfie often shifts between moments of aggression and friendliness, reflecting the dual nature of his persona.')


# Visual 4
st.subheader('(4) Alfie\'s Dialogue Mood Distribution Over Seasons')
season_mood_counts = alfie_data.groupby(['Season', 'Mood']).size().unstack(fill_value=0)
season_mood_counts = season_mood_counts[['Positive', 'Negative', 'Neutral']]
season_mood_percentages = season_mood_counts.div(season_mood_counts.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(season_mood_percentages.index, season_mood_percentages['Positive'], marker='o', label='Positive', color='#76b5c5')
ax.plot(season_mood_percentages.index, season_mood_percentages['Negative'], marker='o', label='Negative', color='#ff6f61')
ax.plot(season_mood_percentages.index, season_mood_percentages['Neutral'], marker='o', label='Neutral', color='#cccccc')
ax.set_xlabel('Season')
ax.set_ylabel('Percentage')
ax.set_title('Percentage of Positive, Negative, and Neutral Moods Over Seasons')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)
st.write('Alfie\'s neutral tone remains dominant throughout the seasons, but there are notable shifts in his positive and negative sentiments. For instance, in Season 3, Episode 6, there’s a significant spike in negativity, likely reflecting key plot developments where Alfie’s darker side becomes more pronounced.')
st.write('Conversely, across most episodes in Seasons 4 and 5, Alfie exhibits a more positive tone, indicating a shift in his character’s demeanor during that period. Alfie’s dialogue consistently adds depth to the narrative, making him a pivotal and unpredictable figure in the series.')


# Visual 5 
st.subheader('(5) Hmms…, Mmms… and Ums…')
filler_words = ['hmm', 'mmm', 'uh', 'um', 'ah', 'oh', 'yeah', 'like', 'huh']
filler_counts = {word: 0 for word in filler_words}

for dialog in alfie_data['Dialog']:
    for word in filler_words:
        filler_counts[word] += dialog.lower().split().count(word)

filler_df = pd.DataFrame(list(filler_counts.items()), columns=['Filler Word', 'Count'])

fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(filler_df['Filler Word'], filler_df['Count'], color='#76b5c5')
ax.set_xlabel('Filler Word')
ax.set_ylabel('Count')
ax.set_title('Count of Filler Words in Alfie Solomons\' Dialogues')
st.pyplot(fig)
st.write('The filler word “yeah” stands out significantly, with nearly 80 occurrences, making it his most frequently used filler word by a large margin, possibly used to assert agreement or as a pause to gather his thoughts before delivering his next line.')
st.write('Other fillers like “hmm,” “mmm,” “like,” and “oh” also appear with moderate frequency, adding to the character’s distinct manner of speaking and contributing to the portrayal of Alfie as calculating.')

# Visual 6 
st.subheader('(6) Bigram network of Alfie\'s Dialogue (Min. 3 Appearances)')
alfie_data['Tokens'] = alfie_data['Dialog'].apply(word_tokenize)
all_tokens = [token for tokens in alfie_data['Tokens'] for token in tokens]
bigram_list = list(bigrams(all_tokens))
bi_grams_count = Counter(bigram_list)

# Plot bigram network
def plot_network(grams_count, title, node_size=2000, edge_width=1.0):
    G = nx.Graph()
    node_sizes = {}
    for gram, count in grams_count.items():
        if count > 2:  # Filter out less frequent bigrams
            G.add_edge(gram[0], gram[1], weight=count)
            for term in gram:
                node_sizes[term] = node_sizes.get(term, 0) + count

    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(G, k=0.8, seed=69)
    nx.draw_networkx_nodes(G, pos, node_size=[v * 200 for v in node_sizes.values()], node_color='#76b5c5')
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    ax.set_title(title)
    st.pyplot(fig)

plot_network(bi_grams_count, 'Bigram Network of Alfie Solomons\' Dialogues (Min. 3)')
st.write('The bigram network of Alfie Solomons\' dialogues offers an understanding of his speech patterns and the words he frequently pairs together. The most prominent bigram, centered around the word “f*cking,” highlights its frequent use, often in combination with words like “right” and “hell” to assert dominance or emphasize a point.')
st.write('The word “yeah,” which was previously identified as his most common filler, also appears prominently in the network, frequently paired with words like “mate” and “oh.” This reinforces the idea that “yeah” serves as a conversational tool for Alfie, allowing him to maintain control of the conversation.')
st.write('Other notable bigrams include “tommy shelby,” indicating that much of Alfie\'s dialogue involves direct interactions or references to the show’s protagonist.')

st.text("Last Updated: 15/9/2024 (Never Updates)")
st.text("By: Hong Kai")



