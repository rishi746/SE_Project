from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sms_ac.retrieval import CorpusIndex, retrieve_context


class RetrievalTests(unittest.TestCase):
    def test_topic_relevance_and_document_diversity(self):
        rows = [
            {'id': 'a', 'document_id': 'one', 'text': 'DNS cache resolver negative caching TTL', 'split': 'train'},
            {'id': 'b', 'document_id': 'one', 'text': 'DNS cache resolver forwarding', 'split': 'train'},
            {'id': 'c', 'document_id': 'two', 'text': 'DNS recursive resolution timeout', 'split': 'train'},
            {'id': 'd', 'document_id': 'three', 'text': 'VPN tunnel IPsec session access', 'split': 'test'},
        ]
        index = CorpusIndex(rows)
        hits = index.search('DNS resolver cache', 3)
        self.assertEqual(hits[0]['document_id'], 'one')
        self.assertEqual(len(hits), 2)
        self.assertEqual(index.search('VPN tunnel'), [])
        self.assertEqual(index.search('VPN tunnel', split=None)[0]['id'], 'd')

    def test_missing_corpus_does_not_fabricate_citations(self):
        self.assertIn('Do not invent', retrieve_context('DNS', Path('missing-corpus-test.jsonl')))


if __name__ == '__main__':
    unittest.main()
