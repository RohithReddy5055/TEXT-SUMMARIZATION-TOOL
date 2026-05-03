
# text_summarizer.py

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

def summarize_text(text, per=0.1):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = [token.text for token in doc]
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    # Ensure we select at least one sentence if the text is not empty
    select_length = max(1, int(len(sentence_tokens) * per))
    
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [sent.text.strip() for sent in summary]
    summary = ' '.join(final_summary)
    return summary

if __name__ == "__main__":
    # Example usage of the text summarizer.
    input_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. NLP is used in various applications such as machine translation, spam detection, sentiment analysis, and text summarization. Text summarization is the process of shortening a text document with software, in order to create a coherent and fluent summary having only the most important points outlined in the document. Automatic text summarization is a common problem in natural language processing and has been the subject of research for many years. The two main approaches to text summarization are extractive and abstractive summarization. Extractive summarization involves selecting a subset of existing words, phrases, or sentences in the original text to form the summary. In contrast, abstractive summarization involves generating new phrases and sentences that may not appear in the original text, in order to convey the meaning of the source document in a more concise way.
    """
    summary = summarize_text(input_text, per=0.2)
    print("Input Text:")
    print(input_text)
    print("\nSummary:")
    print(summary)
