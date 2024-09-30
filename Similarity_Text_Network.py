#Package 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
!wget -O TaipeiSansTCBeta-Regular.ttf https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download
import matplotlib
matplotlib.font_manager.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
matplotlib.rc('font', family='Taipei Sans TC Beta')
import jieba
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pprint
jieba.add_word("小小兵卡")
jieba.add_word("上海商銀")
jieba.add_word("忍者卡")

#生成文本
text_data = [
        "我喜歡上海商銀的產品",
        "上海商銀掰掰囉",
        "上海商銀的忍者卡我有5張",
        "上海商銀服務有夠爛",
        "我有上海商銀小小兵卡，可愛",
        "上海商銀的服務很頂",
        "上海商銀的小小兵卡沒有回饋",
        "我沒有上海商銀的小小兵卡",
        "優惠來說上海商銀很棒666",
        "討厭上海商銀555",
        "忍者"
]
#斷詞斷句
def remove_english_and_numbers(text):
    # 删除英文和数字
    text = re.sub(r'\d','', text)
    # 刪除標點符號
    text = re.sub(r'[^\w\s]','', text)
    # 删除空格
    text = re.sub(r'\s+','', text)
    text = text.lower()
    return text

# 將文本進行斷詞
# tokenized_texts = [jieba.lcut(text) for text in text_data]

cleaned_text = [remove_english_and_numbers(text) for text in text_data]
# tokenized_texts = [
#     [word for word in jieba.cut(cleaned_text) if len(word) > 1]
# ]
tokenized_texts = [
    [word for word in jieba.cut(text) if len(word) > 1]
    for text in cleaned_text
]
word_freq = Counter()
# 查看斷詞結果
for tokens in tokenized_texts:
   word_freq.update(tokens)
   print(tokens)
   print(word_freq)
filtered_words = {word for word, freq in word_freq.items() if freq > 0}
#print(filtered_words)

#計算詞性相似度
#BERT
model_name = "bert-base-chinese"  # 可以換成其他支持中文的 BERT 模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


# 定義一個函數來取得詞語的 BERT 嵌入向量
def get_word_embedding(word, model, tokenizer):
    inputs = tokenizer(word, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 取 [CLS] 的嵌入向量（第0個詞的嵌入向量），shape: (1, hidden_size)
    return outputs.last_hidden_state[0, 0, :].numpy()

# 計算所有詞語的 BERT 嵌入向量
word_embeddings = {}
for word in filtered_words:
    word_embeddings[word] = get_word_embedding(word, model, tokenizer)

# 計算相似性矩陣
words = list(word_embeddings.keys())
embedding_matrix = np.array([word_embeddings[word] for word in words])
similarity_matrix = cosine_similarity(embedding_matrix)

# 將結果轉為字典格式
similarity = {}
for i, word in enumerate(words):
    similarity[word] = {}
    for j, other_word in enumerate(words):
        if i != j:
            similarity[word][other_word] = similarity_matrix[i, j]

# 查看相似性字典
pprint.pprint(similarity)

# 設置相似度閾值
threshold = 0
# 創建網路圖
G = nx.Graph()
# 添加節點和邊，並篩選相似度高於閾值的邊
for word, neighbors in similarity.items():
    for neighbor, sim in neighbors.items():
        if sim >= threshold:
            G.add_edge(word, neighbor, weight=sim)
# 節點大小與詞頻相關
node_size = [word_freq[word] * 300 for word in G.nodes()]
# 繪製網路圖
pos = nx.spring_layout(G, seed=42)  # 節點布局並且固定住形狀
# pos = nx.circular_layout(G)  # 節點布局
edges = G.edges(data=True)
weights = [edge[2]['weight'] * 10 for edge in edges]  # 調整邊的粗細
colors = [edge[2]['weight'] for edge in edges]  # 根據相似性設置顏色
#
plt.figure(figsize=(8, 6))
nx.draw(G, pos, node_color='skyblue', node_size=node_size, with_labels=True, font_family='Taipei Sans TC Beta')  # 根據詞頻調整節點大小
nx.draw_networkx_edges(G, pos, edge_color=colors, edge_cmap=plt.cm.Blues, width=2)  # 深色表示高相似度
plt.title('關鍵詞與其他詞的相似性網路圖', fontfamily='Taipei Sans TC Beta')  # 設置標題字體
plt.show()

