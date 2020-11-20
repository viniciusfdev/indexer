from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
from pathlib import Path
import string
import nltk
import os

# nltk.download('punkt')

class Cleaner:
    def __init__(self, stop_words_file: str, language: str,
                 perform_stop_words_removal: bool, perform_accents_removal: bool,
                 perform_stemming: bool):
        self.set_stop_words = self.read_stop_words(stop_words_file)

        self.stemmer = SnowballStemmer(language)
        in_table = "áéíóúâêôçãẽõü"
        out_table = "aeiouaeocaeou"

        self.accents_translation_table = str.maketrans(in_table, out_table)
        self.set_punctuation = set(string.punctuation)

        # flags
        self.perform_stop_words_removal = perform_stop_words_removal
        self.perform_accents_removal = perform_accents_removal
        self.perform_stemming = perform_stemming

    def html_to_plain_text(self, html_doc: str) -> str:
        try:
            return BeautifulSoup(html_doc, features="lxml").get_text()
        except:
            return None

    def read_stop_words(self, str_file):
        set_stop_words = set()
        with open(str_file, "r") as stop_words_file:
            for line in stop_words_file:
                arr_words = line.split("\n")
                [set_stop_words.add(word) for word in arr_words]
        return set_stop_words

    def is_stop_word(self, term: str):
        return True if term in self.set_stop_words else False

    def word_stem(self, term: str):
        return self.stemmer.stem(term)

    def remove_accents(self, term: str) -> str:
        return term.translate(self.accents_translation_table)

    def preprocess_word(self, term: str) -> str:
        term = str.lower(term)

        if self.perform_stop_words_removal and self.is_stop_word(term):
            return None

        if self.perform_accents_removal:
            term = self.remove_accents(term)

        if self.perform_stemming:
            term = self.word_stem(term)

        return term


class HTMLIndexer:
    cleaner = Cleaner(stop_words_file="stopwords.txt",
                      language="portuguese",
                      perform_stop_words_removal=True,
                      perform_accents_removal=True,
                      perform_stemming=True)

    def __init__(self, index: "HashIndex"):
        self.index = index

    def text_word_count(self, plain_text: str):
        dic_word_count = {}

        words = [self.cleaner.preprocess_word(w) for w in nltk.word_tokenize(plain_text)]

        for w in words:
            if w in dic_word_count:
                dic_word_count[w] = dic_word_count[w] + 1
            elif w:
                dic_word_count[w] = 1
        return dic_word_count

    def index_text(self, doc_id: int, text_html: str):
        plain_text = self.cleaner.html_to_plain_text(text_html)  # (1)
        dict_text = self.text_word_count(plain_text)  # (2)

        for w, freq in dict_text.items():
            self.index.index(w, doc_id, freq)

    def index_text_dir(self, path: str):
        # Exercício 14
        for str_sub_dir in os.listdir(path):
            path_sub_dir = f"{path}/{str_sub_dir}"
            for str_sub_sub in os.listdir(path_sub_dir):
                file_path = f"{path_sub_dir}/{str_sub_sub}"
                with open(file_path, "r") as idx_file:
                    idx_file.read
                    ID = int(str_sub_sub.split(".")[0])
                    self.index_text(ID, idx_file)

    # Wikipedia
    def index_all_text_recursively(self, path: str):
        for i, path in enumerate(Path(path).rglob('*.html')):
            # delta = datetime.now() - self.time
            # self.print_status(delta)
            with open(path, "r") as idx_file:
                idx_file.read
                try:
                    self.index_text(i, idx_file)
                except:
                    pass
        self.index.finish_indexing()
