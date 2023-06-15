import streamlit as st
import subprocess

st.set_page_config(layout='wide', page_title='Fake News Detection | Inspirit AI Weekday 2 All-Hands 3', page_icon=':newspaper:')

# load the model and stuff
@st.cache_resource
def load():
  # system
  import os

  # mathematics and operations
  import math
  import numpy as np

  # capture data
  import pickle
  import requests
  import urllib.request
  import io
  import zipfile

  # html
  from bs4 import BeautifulSoup as bs
  
  # download and unzip resources
  basepath = '.'
  if not os.path.exists(os.path.join(basepath, 'train_val_data.pkl')):
    urllib.request.urlretrieve(
      'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Fake%20News%20Detection/inspirit_fake_news_resources%20(1).zip', 'data.zip'
    )
    
    import zipfile
    with zipfile.ZipFile('data.zip') as zip:
       zip.extractall()
    
    with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
      train_data, val_data = pickle.load(f)

  # model
  from sklearn.feature_extraction.text import CountVectorizer
  from torchtext.vocab import GloVe
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import precision_recall_fscore_support
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import confusion_matrix

  # natural language and vocab
  import nltk
  nltk.download('words')
  from nltk.corpus import words
  vocab = words.words()

  # for keywords, later
  import spacy

  subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_md'])
  text_to_nlp = spacy.load('en_core_web_md')

  y_train = [label for url, html, label in train_data]
  y_val = [label for url, html, label in val_data]

  # prepare data
  def prepare_data(data, featurizer, is_train):
    X = []
    for index, datapoint in enumerate(data):
      url, html, label = datapoint
      html = html.lower()

      features = featurizer(url, html, index, is_train, None)

      # Gets the keys of the dictionary as descriptions, gets the values as the numerical features.
      feature_descriptions, feature_values = zip(*features.items())

      X.append(feature_values)

    return X, feature_descriptions

  # train model
  def train_model(X_train):
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    return model

  # wrapper function for everything above
  def instantiate_model(compiled_featurizer):
    X_train, feature_descriptions = prepare_data(train_data, compiled_featurizer, True)
    X_val, feature_descriptions = prepare_data(val_data, compiled_featurizer, False)

    model = train_model(X_train)

    return model, X_train, X_val, feature_descriptions

  # a wrapper function that takes in named a list of keyword argument functions
  # each of those functions are given the URL and HTML, and expected to return a list or dictionary with the appropriate features
  def create_featurizer(**featurizers):
    def featurizer(url, html, index, is_train, description):
      features = {}

      for group_name, featurizer in featurizers.items():
        group_features = featurizer(url, html, index, is_train, description)

        if type(group_features) == type([]):
          for feature_name, feature_value in zip(range(len(group_features))):
            features[group_name + ' [' + feature_name + ']'] = feature_value
        elif type(group_features) == type({}):
          for feature_name, feature_value in group_features.items():
            features[group_name + ' [' + feature_name + ']'] = feature_value
        else:
          features[group_name] = feature_value

      return features
    return featurizer

  # evaluate model
  def evaluate_model(model, X_val):
    y_val_pred = model.predict(X_val)

    print_metrics(y_val, y_val_pred)
    confusion_matrix(y_val, y_val_pred)

    return y_val_pred

  # confusion matrices
  from sklearn import metrics
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  def confusion_matrix(y_val, y_val_pred):
    # Create the Confusion Matrix
    cnf_matrix = metrics.confusion_matrix(y_val, y_val_pred)

    # Visualizing the Confusion Matrix
    class_names = [0,1] # Our diagnosis categories

    fig, ax = plt.subplots()
    # Setting up and visualizing the plot (do not worry about the code below!)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g') # Creating heatmap
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y = 1.1)
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')

  # other metrics
  def print_metrics(y_val, y_val_pred):
    prf = precision_recall_fscore_support(y_val, y_val_pred)

    print('Accuracy (validation):', accuracy_score(y_val, y_val_pred))
    print('Precision:', prf[0][1])
    print('Recall:', prf[1][1])
    print('F-Score:', prf[2][1])

  # show weights (coefficients for each feature)
  def show_weights(model, feature_descriptions):
    print("\n\n".join(map(lambda feature: f"The feature '{feature[0]}' has a value of {feature[1]}.\nBecause of its sign, the presence of this feature indicates {'fake' if feature[1] > 0 else 'real'} news.", sorted(zip(feature_descriptions, baseline_model.coef_[0].tolist()), key=lambda x: abs(x[1])))))

  # gets the log count of a phrase/keyword in HTML (transforming the phrase/keyword to lowercase).
  def get_normalized_keyword_count(html, soup, keyword):
    # only concern words inside the body, to speed things up
    try:
      necessary_html = soup.body.get_text() # already parsed, contains a body
    except:
      try:
        necessary_html = html.split("<body")[1].split("</body>")[0] # soup could not find a body, but there does exist a body
      except:
        necessary_html = html # if it doesn't have a body...

    return math.log(1 + necessary_html.count(keyword.lower())) # log is a good normalizer

  # count the number of words in a URL
  def count_words_in_url(url):
    for i in range(1, len(url)): # don't count the first letter, because sometimes that might be a word by itself (like why bother counting 'l' a word?)
      if url[:i].lower() in vocab: # if it's a word
        return 1 + count_words_in_url(url[i:]) # get more words, and keep counting
    return 0 # no words in URL (or at least, it doesn't start with a word, such as NYTimes)

  # get the description (usually a meta tag) from raw HTML
  def get_description_from_html(soup):
    description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
    if description_tag:
      description = description_tag.get('content') or ''
    else: # If there is no description, return an empty string.
      description = ''
    return description

  train_bs = [bs(html) for url, html, label in train_data]
  val_bs = [bs(html) for url, html, label in val_data]

  # get raw descriptions
  def get_descriptions_from_data(data, is_train):
    # A dictionary mapping from url to description for the websites in the train_data.
    descriptions = []
    if is_train:
      for soup in train_bs:
        descriptions.append(get_description_from_html(soup))
    else:
      for soup in val_bs:
        descriptions.append(get_description_from_html(soup))

    return descriptions

  train_descriptions = get_descriptions_from_data(train_data, True)
  val_descriptions = get_descriptions_from_data(val_data, False)

  # bag of words (bow)
  vectorizer = CountVectorizer(max_features=300) # create a new vectorizer
  vectorizer.fit(train_descriptions)

  def vectorize_data_descriptions(descriptions): # convert a description into a vector
    return vectorizer.transform(descriptions).todense() # .todense() fills in blank values in the vector, so we can work with it

  train_bow_features = np.array(vectorize_data_descriptions(train_descriptions))
  val_bow_features = np.array(vectorize_data_descriptions(val_descriptions))

  # GloVe
  VEC_SIZE = 300
  glove = GloVe(name='6B', dim=VEC_SIZE)

  # Returns word vector for word if it exists, else return None.
  def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

  # just like above, transform our data
  def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
      found_words = 0.0
      description = description.strip()
      for word in description.split():
        vec = get_word_vector(word)
        if vec is not None:
          # Increment found_words and add vec to X[i].
          found_words += 1
          X[i] += vec
      # We divide the sum by the number of words added, so we have the
      # average word vector.
      if found_words > 0:
        X[i] /= found_words

    return X

  train_glove_features = glove_transform_data_descriptions(train_descriptions)
  val_glove_features = glove_transform_data_descriptions(val_descriptions)

  def url_extension_featurizer(url, html, index, is_train, description):
    features = {}

    features['.com'] = url.endswith('.com')
    features['.org'] = url.endswith('.org')
    features['.edu'] = url.endswith('.edu')
    features['.net'] = url.endswith('.net')
    features['.co'] = url.endswith('.co')
    features['.nz'] = url.endswith('.nz')
    features['.media'] = url.endswith('.media')
    features['.za'] = url.endswith('.za')
    features['.fr'] = url.endswith('.fr')
    features['.is'] = url.endswith('.is')
    features['.tv'] = url.endswith('.tv')
    features['.press'] = url.endswith('.press')
    features['.news'] = url.endswith('.news')
    features['.uk'] = url.endswith('.uk')
    features['.info'] = url.endswith('.info')
    features['.ca'] = url.endswith('.ca')
    features['.agency'] = url.endswith('.agency')
    features['.us'] = url.endswith('.us')
    features['.ru'] = url.endswith('.ru')
    features['.su'] = url.endswith('.su')
    features['.biz'] = url.endswith('.biz')
    features['.ir'] = url.endswith('.ir')

    return features

  def keyword_featurizer(url, html, index, is_train, description):
    features = {}

    if is_train:
      soup = train_bs[index]
    else:
      soup = val_bs[index]

    keywords = ['vertical', 'news', 'section', 't', 'light', 'data', 'eq', 'medium', 'large', 'ad', 'header', 'text', 'js', 'nav', 'analytics', 'article', 'menu', 'tv', 'cnn', 'button', 'icon', 'edition', 'span', 'item', 'label', 'link', 'world', 'politics', 'president', 'donald', 'business', 'food', 'tech', 'style', 'amp', 'vr', 'watch', 'search', 'list', 'media', 'wrapper', 'div', 'zn', 'l', 'card', 'm', 'z', 'var', 'prod', 'true', 'window', 'u', 'n', 'new', 's', 'color', 'width', 'container', 'mobile', 'fixed', 'flex', 'aria', 'tablet', 'desktop', 'type', 'size', 'tracking', 'heading', 'logo', 'svg', 'path', 'fill', 'content', 'ul', 'li', 'shop', 'home', 'static', 'wrap', 'main', 'img', 'celebrity', 'lazy', 'image', 'high', 'noscript', 'inner', 'margin', 'headline', 'child', 'interest', 'john', 'movies', 'music', 'parents', 'real', 'warren', 'opens', 'share', 'people', 'max', 'min', 'state', 'event', 'story', 'click', 'time', 'trump', 'elizabeth', 'year', 'visit', 'post', 'public', 'module', 'latest', 'star', 'skip', 'imagesvc', 'p', 'posted', 'ltc', 'summer', 'square', 'solid', 'default', 'g', 'super', 'house', 'pride', 'week', 'america', 'man', 'day', 'wp', 'york', 'id', 'gallery', 'inside', 'calls', 'big', 'daughter', 'photo', 'joe', 'deal', 'app', 'special', 'j', 'source', 'red', 'table', 'money', 'family', 'featured', 'makes', 'pete', 'michael', 'video', 'case', 'says', 'popup', 'carousel', 'category', 'script', 'helvetica', 'feature', 'dark', 'extra', 'small', 'horizontal', 'bg', 'hierarchical', 'paginated', 'siblings', 'grid', 'active', 'demand', 'background', 'height', 'cn', 'cd', 'src', 'cnnnext', 'dam', 'report', 'trade', 'images', 'file', 'huawei', 'mueller', 'impeachment', 'retirement', 'tealium', 'col', 'immigration', 'china', 'flag', 'track', 'tariffs', 'sanders', 'staff', 'fn', 'srcset', 'green', 'orient', 'iran', 'morning', 'jun', 'debate', 'ocasio', 'cortez', 'voters', 'pelosi', 'barr', 'buttigieg', 'american', 'object', 'javascript', 'r', 'h', 'uppercase', 'omtr', 'chris', 'dn', 'hfs', 'rachel', 'maddow', 'lh', 'teasepicture', 'db', 'xl', 'articletitlesection', 'founders', 'mono', 'ttu', 'biden', 'boston', 'bold', 'anglerfish', 'jeffrey', 'radius']
    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_keyword_count(html, soup, keyword)

    return features

  def url_word_count_featurizer(url, html, index, is_train, description):
    return count_words_in_url(url.split(".")[-2])
    # for example, www.google.com will return google and nytimes.com will return nytimes

  def bag_of_words_featurizer(url, html, index, is_train, description):
    if index == -1:
      return vectorize_data_descriptions([description])[0]

    if is_train:
      return train_bow_features[index]
    else:
      return val_bow_features[index]

  def glove_featurizer(url, html, index, is_train, description):
    if index == -1:
      return glove_transform_data_descriptions([description])[0]
      
    if is_train:
      return train_glove_features[index]
    else:
      return val_glove_features[index]

  return url_extension_featurizer, keyword_featurizer, url_word_count_featurizer, bag_of_words_featurizer, glove_featurizer, create_featurizer, instantiate_model, evaluate_model, show_weights

# load everything
url_extension_featurizer, keyword_featurizer, url_word_count_featurizer, bag_of_words_featurizer, glove_featurizer, create_featurizer, instantiate_model, evaluate_model, show_weights = load()

compiled_featurizer = create_featurizer(
    url_extension=url_extension_featurizer,
    keyword=keyword_featurizer,
    url_word_count=url_word_count_featurizer,
    bag_of_words=bag_of_words_featurizer,
    glove=glove_featurizer)

model, X_train, X_val, feature_descriptions = instantiate_model(compiled_featurizer)

# columns
left, right = st.columns(2)

# on the left, do the overview
left.title('Fake News Detection')
left.header('Inspirit AI')
left.subheader('Weekday 2 All-Hands 3')
left.text('**Instructor:** Paul')
left.text('**Group Members:** Daanish, Daniel, Dheeraj, Justin, Pranil, Timothy')

left.divider()

left.text('See the [Google Colab](https://colab.research.google.com/drive/1NutMv5iJ2DAbU_YHPRonSrurvHQ2Al9v?usp=sharing).')

left.divider()

left.text('Here are some metrics! ... metrics')

# on the right side, allow users to submit a URL
right.header('Try it out!')

with right.form(key='try_it_out'):
  url = st.text_input(label='Enter a news article or site URL to predict validity', key='url')
  submit_button = st.form_submit_button(label='Submit', type='primary')

  advice = st.text('*Make sure your URL is a valid news site.*')

  if st.form_submit_button(label='Submit', type='primary'):
    try:
      response = requests.get(url)
      html = response.text.lower()
      soup = bs(html)

      features = compiled_featurizer(url, html, -1, False, soup)
      _, feature_values = zip(*features.items())

      prediction = model.predict([feature_values])[0]

      advice = st.text('*We predict that your news is ' + ('FAKE' if prediction else 'REAL') + ' news!')

      # put some scores, maybe? idk
      # also put the weights and the features
    except:
      advice = st.text('*I don\'t think your URL worked. Please check your spelling or try another.*')
