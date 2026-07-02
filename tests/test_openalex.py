import json
import unittest
from unittest.mock import patch, MagicMock

from folium.tools.openalex import OpenAlexSearchTool, _format_authors
from folium.tools.base import ToolValidationError


class OpenAlexSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tool = OpenAlexSearchTool()

    def test_name_and_description(self):
        self.assertEqual(self.tool.name, "paper_search")
        self.assertIn("paper", self.tool.description.lower())

    def test_required_fields(self):
        with self.assertRaises(ToolValidationError) as ctx:
            self.tool.validate_arguments({})
        self.assertIn("query", str(ctx.exception))

    def test_valid_arguments(self):
        args = {"query": "machine learning"}
        result = self.tool.validate_arguments(args)
        self.assertEqual(result, args)

    def test_optional_fields(self):
        args = {"query": "LLM", "max_results": 10, "sort": "citations"}
        result = self.tool.validate_arguments(args)
        self.assertEqual(result, args)

    def test_unknown_field_rejected(self):
        with self.assertRaises(ToolValidationError):
            self.tool.validate_arguments({"query": "test", "unknown": "value"})

    def test_wrong_type_rejected(self):
        with self.assertRaises(ToolValidationError):
            self.tool.validate_arguments({"query": 123})


class FormatAuthorsTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_format_authors([]), "")

    def test_single_author(self):
        authors = [{"author": {"display_name": "Alice"}}]
        self.assertEqual(_format_authors(authors), "Alice")

    def test_multiple_authors(self):
        authors = [
            {"author": {"display_name": "Alice"}},
            {"author": {"display_name": "Bob"}},
        ]
        self.assertEqual(_format_authors(authors), "Alice, Bob")

    def test_more_than_five_truncated(self):
        authors = [{"author": {"display_name": f"Author{i}"}} for i in range(7)]
        result = _format_authors(authors)
        self.assertIn("et al.", result)
        self.assertIn("7 total", result)

    def test_missing_display_name(self):
        authors = [{"author": {}}]
        self.assertEqual(_format_authors(authors), "Unknown")


MOCK_RESPONSE = {
    "results": [
        {
            "title": "Attention Is All You Need",
            "authorships": [
                {"author": {"display_name": "Ashish Vaswani"}},
                {"author": {"display_name": "Noam Shazeer"}},
            ],
            "publication_year": 2017,
            "cited_by_count": 120000,
            "doi": "https://doi.org/10.48550/arXiv.1706.03762",
            "open_access": {"oa_url": "https://arxiv.org/pdf/1706.03762.pdf"},
            "primary_location": {
                "source": {"display_name": "NeurIPS"}
            },
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authorships": [
                {"author": {"display_name": "Jacob Devlin"}},
            ],
            "publication_year": 2019,
            "cited_by_count": 80000,
            "doi": "https://doi.org/10.48550/arXiv.1810.04805",
            "open_access": {"oa_url": ""},
            "primary_location": {"source": {}},
        },
    ]
}


class OpenAlexExecuteTests(unittest.TestCase):
    def setUp(self):
        self.tool = OpenAlexSearchTool()

    @patch("folium.tools.openalex.urllib.request.urlopen")
    def test_returns_formatted_results(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(MOCK_RESPONSE).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.tool.execute(query="attention mechanism", max_results=2)

        self.assertIn("Attention Is All You Need", result)
        self.assertIn("Ashish Vaswani", result)
        self.assertIn("120000", result)
        self.assertIn("NeurIPS", result)
        self.assertIn("arxiv.org/pdf", result)
        self.assertIn("BERT", result)

    @patch("folium.tools.openalex.urllib.request.urlopen")
    def test_empty_results(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"results": []}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.tool.execute(query="nonexistent topic xyz")
        self.assertIn("No papers found", result)

    def test_empty_query_returns_error(self):
        result = self.tool.execute(query="   ")
        self.assertIn("Error", result)

    @patch("folium.tools.openalex.urllib.request.urlopen")
    def test_sort_citations(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(MOCK_RESPONSE).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.tool.execute(query="test", sort="citations")
        call_args = mock_urlopen.call_args
        url = call_args[0][0].full_url
        self.assertIn("cited_by_count%3Adesc", url)


if __name__ == "__main__":
    unittest.main()
