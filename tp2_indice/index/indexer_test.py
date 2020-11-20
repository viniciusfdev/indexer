import math
import tracemalloc
from datetime import datetime

from IPython.core.display import clear_output
from index.indexer import *
from index.structure import *
import unittest

class IndexerTest(unittest.TestCase):
    def test_indexer_wiki_short(self):
        print("running...")

        self.time = datetime.now()
        tracemalloc.start()

        obj_index = FileIndex()
        html_indexer = HTMLIndexer(obj_index)
        html_indexer.index_all_text_recursively("wiki/100")

        delta = datetime.now() - self.time
        self.print_status(delta)
        tracemalloc.stop()
        obj_index.calc_avg_time_mem(delta.total_seconds())

    def test_indexer_wiki_hashindex(self):
        print("running...")

        self.time = datetime.now()
        tracemalloc.start()

        obj_index = HashIndex()
        html_indexer = HTMLIndexer(obj_index)
        html_indexer.index_all_text_recursively("wiki/100")

        delta = datetime.now() - self.time
        self.print_status(delta)
        tracemalloc.stop()
        obj_index.calc_avg_time_mem()

    def test_indexer_wiki_full(self):
        print("running...")

        self.time = datetime.now()
        tracemalloc.start()

        obj_index = FileIndex()
        html_indexer = HTMLIndexer(obj_index)
        html_indexer.index_all_text_recursively("wiki")

        delta = datetime.now() - self.time
        self.print_status(delta)
        tracemalloc.stop()
        obj_index.calc_avg_time_mem(delta.total_seconds())

    def print_status(self, delta):
        current, peak = tracemalloc.get_traced_memory()

        clear_output(wait=True)
        print((f"Memoria usada: {current / 10 ** 6:,} MB; Máximo {peak / 10 ** 6:,} MB"), flush=True)
        print((f"Tempo gasto: {delta.total_seconds()}s"), flush=True)

    def test_indexer(self):
        obj_index = HashIndex()
        html_indexer = HTMLIndexer(obj_index)
        html_indexer.index_text_dir("index/docs_test")
        set_vocab = set(obj_index.vocabulary)
        set_expected_vocab = set(['a', 'cas', 'ser', 'verd', 'ou', 'nao', 'eis', 'questa'])

        sobra_expected = set_expected_vocab - set_vocab
        sobra_vocab = set_vocab - set_expected_vocab

        self.assertTrue(len(sobra_expected) == 0 and len(sobra_vocab) == 0,
                        f"O Vocabulário indexado não é o esperado!\nVocabulario indexado: {set_vocab}\nVocabulário esperado: {set_expected_vocab}")
        lst_occur = obj_index.get_occurrence_list("cas")
        dic_expected = {111: TermOccurrence(111, 2, 1),
                        100102: TermOccurrence(100102, 2, 2)}
        for occur in lst_occur:
            self.assertTrue(type(occur.doc_id) == int, f"O tipo do documento deveria ser inteiro")
            self.assertTrue(occur.doc_id in dic_expected,
                            f"O docid número {occur.doc_id} não deveria existir ou não deveria indexar o termo 'cas'")
            self.assertEqual(dic_expected[occur.doc_id].term_freq, occur.term_freq,
                             f"A frequencia do termo 'cas' no documento {occur.doc_id} deveria ser {occur.term_freq}")


if __name__ == "__main__":
    unittest.main()
