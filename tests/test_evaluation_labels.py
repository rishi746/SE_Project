import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_dual_models import lifecycle_label


class RecommendationLabels(unittest.TestCase):
    def test_recommendation_after_title_and_blank_lines(self):
        self.assertEqual(lifecycle_label('# Project\n\n## Recommendation and confidence\nUse **Risk-driven spiral** for this design.\n\n## Alternatives\nWaterfall and agile.'), 'spiral')

    def test_alternatives_do_not_supply_primary_label(self):
        self.assertIsNone(lifecycle_label('# Project\n\n## Recommendation and confidence\nInsufficient evidence to select a lifecycle.\n\n## Alternatives\nWaterfall and spiral.'))

    def test_existing_explicit_field(self):
        self.assertEqual(lifecycle_label('Recommended model: V-model\nAlternatives: waterfall'), 'v-model')

    def test_detailed_dataset_prototype_heading(self):
        self.assertEqual(lifecycle_label('# Most Appropriate Life Cycle Model: Prototype\n\n**Decision:** The Prototype model is best.'), 'prototyping')
