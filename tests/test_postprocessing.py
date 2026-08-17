import unittest

from student_eval.openrouter import _classify_http_error, parse_student_answer


class PostprocessingParserTests(unittest.TestCase):
    def test_classifies_provider_agnostic_json_schema_rejection(self):
        novita_error = (
            "Model does not support 'json_schema' response format. "
            "Supported formats: json_object."
        )
        self.assertEqual(
            _classify_http_error(400, novita_error),
            "unsupported_response_format",
        )

    def test_classifies_422_structured_output_rejection(self):
        self.assertEqual(
            _classify_http_error(422, "Structured output is invalid for this endpoint"),
            "unsupported_response_format",
        )

    def test_classifies_deepinfra_405_json_schema_rejection(self):
        deepinfra_error = (
            "json_schema response format is not supported for model: "
            "google/gemma-4-31B-it-turbo"
        )
        self.assertEqual(
            _classify_http_error(405, deepinfra_error),
            "unsupported_response_format",
        )

    def test_repairs_unescaped_newlines(self):
        parsed = parse_student_answer('{"reason": "line one\nline two", "choice": "B"}')
        self.assertEqual(parsed["choice"], "B")
        self.assertEqual(parsed["parse_method"], "control_character_repair")
        self.assertTrue(parsed["parse_repaired"])

    def test_recovers_explicit_choice_from_otherwise_malformed_json(self):
        parsed = parse_student_answer(
            '{"reason": "the model used an "unescaped quote"", "choice": "C"}'
        )
        self.assertEqual(parsed["choice"], "C")
        self.assertEqual(parsed["parse_method"], "explicit_choice_field_repair")

    def test_rejects_truncated_output_without_choice(self):
        with self.assertRaises(ValueError):
            parse_student_answer('{"reason": "unfinished analysis')


if __name__ == "__main__":
    unittest.main()
