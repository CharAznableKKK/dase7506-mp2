"""Behavioral tests for student selection and training, using synthetic fixtures."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import selection
    import training
except ModuleNotFoundError:
    selection = training = None


def question(answers, name='synthetic'):
    return {'id': name, 'question': 'How many?', 'candidates': [
        {'index': i, 'text': text} for i, text in enumerate(answers)
    ]}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(selection, 'Student selection module is not implemented')

    def test_equivalent_numbers_share_votes_and_invalid_is_not_a_vote(self):
        bank = [question(['#### 1/2', '#### 0.5', 'no marked answer', '#### 3'])]
        self.assertEqual(selection.answer_groups(bank), [['1/2', '1/2', None, '3']])
        self.assertEqual(selection.choose(bank, np.zeros((1, 4)), 'majority'), [0])

    def test_invalid_candidate_never_beats_valid_and_ties_are_stable(self):
        bank = [question(['unparseable', '#### 2', '#### 3', '#### 4'])]
        self.assertEqual(selection.choose(bank, np.array([[99, 1, 1, 0]]), 'none'), [1])
        self.assertEqual(selection.choose([question(['?', '?', '?', '?'])], np.ones((1, 4)), 'none'), [0])

    def test_answer_pooling_changes_choice_using_only_same_answer_evidence(self):
        bank = [question(['#### 1', '#### 1', '#### 2', '#### 3'])]
        scores = np.array([[4., -4., 2., 1.]])
        self.assertEqual(selection.choose(bank, scores, 'none'), [0])
        self.assertEqual(selection.choose(bank, scores, 'answer_mean'), [2])

    def test_predictions_do_not_depend_on_ids_question_count_or_order(self):
        bank = [question(['#### 1', '#### 1', '#### 2', '?'], 'a'),
                question(['#### 2', '#### 3', '#### 3', '#### 4'], 'b')]
        arrays = {'embeddings': np.zeros((2, 4, 1024), dtype=np.float32)}
        choices = selection.predict(bank, arrays, None)
        self.assertEqual(choices, [0, 1])
        self.assertTrue(all(type(x) is int for x in choices))
        reordered = [dict(bank[1], id='new-id'), dict(bank[0], id='another')]
        self.assertEqual(selection.predict(reordered, {'embeddings': arrays['embeddings'][::-1]}, None), [1, 0])
        self.assertEqual(selection.predict(bank[1:], {'embeddings': arrays['embeddings'][1:]}, None), [1])

    def test_feature_contract_rejects_misaligned_embeddings(self):
        with self.assertRaises(ValueError):
            selection.features([question(['#### 1'] * 4)], np.zeros((2, 4, 1024)), 'embedding')

    def test_surface_features_handle_fractions_negative_values_and_missing_answers(self):
        bank = [question(['#### 1/2', '#### -3', 'no marker', '#### 100000000000000000000'])]
        result = selection.features(bank, np.zeros((1, 4, 1024)), 'full')
        self.assertEqual(result.shape, (1, 4, 1039))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result[0, 0, 1034], 0)
        self.assertLess(result[0, 1, 1034], 0)
        self.assertEqual(result[0, 2, 1033], 0)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(training, 'Student training module is not implemented')

    def test_labels_keep_all_correct_candidates_and_reject_reference_mismatch(self):
        bank = [question(['#### 2', '#### 2.0', '#### 3', 'nothing'])]
        refs = [{'id': 'synthetic', 'question': 'How many?', 'answer': '#### 2'}]
        np.testing.assert_array_equal(training.labels_from_references(bank, refs), [[1, 1, 0, 0]])
        with self.assertRaises(ValueError):
            training.labels_from_references(bank, [dict(refs[0], id='wrong')])

    def test_training_loader_refuses_test_and_validation_before_reading_files(self):
        for split in ['test', 'audit', 'validation']:
            with self.assertRaises(ValueError):
                training.load_training(split, Path('does-not-exist'))

    def test_pairwise_fit_learns_correct_direction_and_roundtrips_checkpoint(self):
        x = np.array([[[3.], [1.], [0.], [-1.]], [[-1.], [0.], [2.], [3.]]])
        y = np.array([[1., 0., 0., 0.], [0., 0., 1., 1.]])
        model = training.fit_ridge(x, y, np.ones_like(y, dtype=bool), alpha=0.1, objective='pairwise')
        scores = selection.linear_scores(x, model)
        self.assertEqual(scores.argmax(1).tolist(), [0, 3])
        with tempfile.TemporaryDirectory() as directory:
            training.save_model(Path(directory), model, {'feature_set': 'surface', 'pooling': 'none'})
            restored, config = selection.load_model(Path(directory))
            np.testing.assert_allclose(selection.linear_scores(x, restored), scores)
            self.assertEqual(config['pooling'], 'none')


if __name__ == '__main__':
    unittest.main()
