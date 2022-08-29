import jieba
from wordcloud import WordCloud
txt = '弱小的人,才习惯,嘲讽和否定，而内心,强大的人,从不吝啬赞美和鼓励！我们就是后浪，奔涌吧！后浪，奔涌吧！本篇先来解释一个名词“词云”，“词云”就是对网络文本中出现频率较高的“关键词”予以视觉上的突出，形成“关键词云层”或“关键词渲染”，从而过滤掉大量的文本信息，使浏览网页者只要一眼扫过文本就可以领略文本的主旨。 '
words = jieba.lcut(txt)     #精确分词
newtxt = ''.join(words)    #空格拼接
wordcloud = WordCloud(background_color='white',colormap='Blues',font_path =  "STKAITI.TTF").generate(newtxt)
wordcloud.to_file('中文词云图.jpg')
