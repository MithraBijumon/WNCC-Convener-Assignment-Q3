# WNCC-Convener-Assignment-Q3
An AI-based system that can automatically categorize queries into a category

#Pre - Processors
1. I used a function called clean. I basically wnted to make the sentences lower cased and remove the "?" at the end of each sentence since I didn't want the vocabulary size grow unnecessarily. If I did't do this suppose a question is "What is Machine Learning?" and another is "Explain what machine learning is" then "What", "what", "Machine", "machine", "Learning?", "learning" will be added to my vocab. This is basically a lot of noise and my model will not learn fast and will require huge amounts of data to fine tune its predictions. Another function of my clean function is the stopword removal. So to be frank I didn't add this to my model until I read the assignment questions. Which is when I realized what it was and how effective it could be. So it means "getting rid of the super common words in a sentence that don’t add much meaning.". So I added that feature as well. As of now I didn't get the time to add stemming or lemmatization because I didn't get the time. But I did find out what it was and have thoughts on it. "Stemming is a technique in Natural Language Processing (NLP) that reduces a word to its base or root form—called the stem—by chopping off suffixes and prefixes. To treat different forms of a word (like connect, connected, connection, connecting) as the same base word, so models don't treat them as different concepts." "Lemmatization is like smart stemming—it reduces words to their dictionary base form, called a lemma, but does so by understanding the context and grammar of the word." So in this model I feel Stemming would be suffficient since its a basic classifier model and stemming would be way quicker and easier to implement. Since the grammar of words don't really matter much here. 
2.  Next I also used tokenizers. So basically I created a vocabulary of words from the training set. I gave each word an index. Then later when a sentence is passed through my tokenize function, it retrieves the index for each word and creates a list with each word replaced by its index now called "token". So what if a word was not found in my vocabulary? It will return the token for "Unknown" ie 1 in this case. My vocabulary also uses 0 as the index for padding. So why did I add padding here? Basically later when I concatenated all tokenized sentences from the training set together it gives a tensor of shape [no of sentences, length of sentence] right. But of course u can't have shape varying at every row. So I had to fix the size of each tokenized sentence. So I added a padding "0" at the end. So now in my training set suppose the length of the longest sentence is l, the shape of my tensor is [no of sentences, l]. Now initially to be on the safe side I fixed this l to 100. But then training took a hell lot of time. So I thought of finding the maximum length of the sentence by iterating through my training set. Now one issue that can arise is if I give a sentence longer than this in evaluation. I plan to work on this later. So the question is does padding affect my training? The answer is no because "if mask is not None: attn_scores = attn_scores.masked_fill(mask == 0, -1e9)" this statement asks my model to ignore all 0 indexed posiitions.
3. Now one thing I wish to highight is that I used these pre-processors for my training data only: i.e., for my input only. I directly used the categgory names for the output data. Now why? Beacuse its a basic classifier model and the output is a maximum of 2 words and does not require embedding... (note that embedding is used when the position of the word in the sentence is important). Why should I embed a six class output in huge dimensions and increase training time 😂. So while my source data is of shape [no of sentences, length of longest sentence], my target data is of shape [no of sentences]

#Text Representation Method   (Note : From now on I use "l" for no of words in the longest sentence, and N for no of sentences, and d for dimension=512)
I used the transformer method for training my AI. What it does? So now I have a source data tensor right. Each word is represented by a token. So right now the word knows what it is. But does it know it's meaning? Does it know what its relation with the other words in the sentence is? Does it know what its position is? 
1. So the transformer model first embeds each word in an n dimensional space, I used 512 dimensions to embed each word. So each token is given a vector of length 512. So now our data is a tensor of shape[N,l,d]. 
2. Next I pass it through a positional encoder. Why is this needed? To let the words understand were it lies in the sentence. So how does it work? First it initializes a tensor "pe" with 0s of shape [l,d]. Next it initializes a tensor [0,1,2,3,4....l] of shape [l-1,1] and another tensor [0,2,4,....512]*(-(math.log(10000.0) / d_model))) (basically this tensor helps create smooth but unique positional signals across dimensions by scaling down the input to sin and cos. Without it, all positional encodings would look the same and lose their ability to differentiate between token positions) of shape [1, d/2] and matrix multiplies them to give a tensor of shape [l,d/2]. Each value is then either signed(and added to even columns of pe) or cosined(and added to odd columns of pe). pe now holds the positional encodings information. Then pe is registered as a "non-trainable" tensor (from what I understood it means grad is not calculated for it when passed through register_buffer) and now its shape is set [1,l,d]. Later when training dataset is passed through positional encoder it returns a tensor of embeddings that also has positional knowledge now.
3. Now for the final processed dataset this encoded one is passed through a dropout function. Whhat does it do? Suppose dropout is 0.1(as is here), then during training my model "drops" or randomly sets 10% of my elements to 0 (i.e., padded) to prevent overfitting. It now does not overly rely on one specific sentence for training.
4. So now what does my transformer do? It passes my dataset through an encoder block. Traditionally a transformer has an encoder and decoder block. I used the very traditional model but made changes so that now my transformer has only an encoder block. This is because we don't need a decoder block since my model is only classifying things.
This encoder block has a multi head attention. This is the deal breaker. It allows each word in a sentence to know how it relates to every other word including itself. First it splits each word embedding into a specific number of heads. And then gives an attention score for each head and then combines it again. I'll explain. Suppose the source has shape [N,l,d] and you want 8 heads. First the data is made into shape [N, no_of_heads = 8, l, d/8]. Now for each head which is of shape [1,d/8] is matrix multiplied by the transpose of the head of every word in the sentence in all sentences. This is then normalized by dividing with math.sqrt(self.d_k). Suppose you call this the energy of each word(i.e., how much attention it deserves). I'll explain what this energy is in detail. Suppose you have a dataset - A of shape [5, 2, 10, 4], where:
- 5 = number of sentences (batch size)
- 2 = number of attention heads
- 10 = number of words
- 4 = dimension of each head
- So for each sentence and each head, you're encoding 10 words into 4-dimensional vectors.
- A @ A.transpose(-2, -1)
- This is a matrix multiplication between:
- A: shape [5, 2, 10, 4]            #Query
- A.transpose(-2, -1): shape [5, 2, 4, 10]               #Key
- So the resulting matrix now has a shape [5,2,10,10] which basically gives the energy. What does this represent? I'll show how each head looks like.

| Queries ↓ / Keys → | Word 1 | Word 2 | Word 3 | Word 4 | Word 5 | Word 6 | Word 7 | Word 8 | Word 9 | Word 10 |
|--------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|
| Word 1             |        |        |        |        |        |        |        |        |        |          |
| Word 2             |        |        |        |        |        |        |        |        |        |          |
| Word 3             |        |        |        |        |        |        |        |        |        |          |
| Word 4             |        |        |        |        |        |        |        |        |        |          |
| Word 5             |        |        |        |        |        |        |        |        |        |          |
| Word 6             |        |        |        |        |        |        |        |        |        |          |
| Word 7             |        |        |        |        |        |        |        |        |        |          |
| Word 8             |        |        |        |        |        |        |        |        |        |          |
| Word 9             |        |        |        |        |        |        |        |        |        |          |
| Word 10            |        |        |        |        |        |        |        |        |        |          |

So each sentence has 2 such tables representing each head. Why did we divide it into heads becomes clear now... to make it into batches so that computation becomes faster. This is also the reason we get increasing and decreasing losses because the heads vary. 
So basically every this tensor shows how much attention each word must give to the other words in the sentence to realize what it means. This is then softmaxed to give the probabilities. Now these probabilities are vector mulitplied with the original A again. So now suppose A is a head with the following structure.
| Word ↓ / Dim → | dim1 | dim2 | dim3 | dim4 |
|----------------|------|------|------|------|
| w1             |      |      |      |      |
| w2             |      |      |      |      |
| w3             |      |      |      |      |
| w4             |      |      |      |      |
| w5             |      |      |      |      |
| w6             |      |      |      |      |
| w7             |      |      |      |      |
| w8             |      |      |      |      |
| w9             |      |      |      |      |
| w10            |      |      |      |      |

Now you have the attention score table of A above. So multiplying that with A gives
| Word ↓ / Dim → | dim1 | dim2 | dim3 | dim4 |
|----------------|------|------|------|------|
| w1             |      |      |      |      |
| w2             |      |      |      |      |
| w3             |      |      |      |      |
| w4             |      |      |      |      |
| w5             |      |      |      |      |
| w6             |      |      |      |      |
| w7             |      |      |      |      |
| w8             |      |      |      |      |
| w9             |      |      |      |      |
| w10            |      |      |      |      |

were each cell represent how much attention a word gives its sentence per dimension. 
Now multi head attention recombine everything to give a tensor of shape [5,10,8] were now every sentence has 10 words and each word has 8 elements representing how much attention each dimension gives it sentence.
