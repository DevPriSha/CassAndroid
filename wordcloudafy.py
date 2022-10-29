import matplotlib.pyplot as pPlot
from wordcloud import WordCloud, STOPWORDS, get_single_color_func
import numpy as np
from PIL import Image

def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl(0,100%, 1%)")

dataset = open("/Users/devprisha/Documents/GitHub/CassAndroid/Text files/Sequentia.txt", "r").read()
def create_word_cloud(string):
   alice_mask = np.array(Image.open("/Users/devprisha/Downloads/kisspng-star-computer-icons-symbol-eid-icon-5b14e303bde298.2180247415280954917778.png"))

   stopwords = set(STOPWORDS)
   stopwords.add("said")
   stopwords.add("one")
   # stopwords = set("")

   wc = WordCloud(background_color="white", max_words=1000, mask=alice_mask, stopwords=stopwords, contour_width=3, contour_color='steelblue')
   wc.generate(string)
   wc.recolor(color_func=black_color_func)
   wc.to_file("./Images/Vent_wordCloud.png")

dataset = dataset.lower()
create_word_cloud(dataset)