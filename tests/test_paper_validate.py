import json
import unittest
from unittest.mock import MagicMock, patch

from folium.tools.base import ToolValidationError
from folium.tools.paper_validate import PaperValidateTool


OPENALEX_WORK = {
    "id": "https://openalex.org/W123",
    "title": "Attention Is All You Need",
    "authorships": [
        {"author": {"display_name": "Ashish Vaswani"}},
        {"author": {"display_name": "Noam Shazeer"}},
    ],
    "publication_year": 2017,
    "cited_by_count": 120000,
    "doi": "https://doi.org/10.48550/arXiv.1706.03762",
    "primary_location": {"source": {"display_name": "NeurIPS"}},
    "type": "journal-article",
}


class PaperValidateSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tool = PaperValidateTool()

    def test_name_and_description(self):
        self.assertEqual(self.tool.name, "paper_validate")
        self.assertIn("validate", self.tool.description.lower())

    def test_required_fields(self):
        with self.assertRaises(ToolValidationError) as ctx:
            self.tool.validate_arguments({})
        self.assertIn("papers", str(ctx.exception))

    def test_valid_arguments(self):
        args = {"papers": [{"title": "Attention Is All You Need"}]}
        self.assertEqual(self.tool.validate_arguments(args), args)


class PaperValidateExecuteTests(unittest.TestCase):
    def setUp(self):
        self.tool = PaperValidateTool()

    @patch("folium.tools.paper_validate.urllib.request.urlopen")
    def test_confirms_exact_doi_match(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"results": [OPENALEX_WORK]})

        result = self.tool.execute(papers=[{
            "title": "Attention Is All You Need",
            "doi": "10.48550/arXiv.1706.03762",
            "year": 2017,
            "venue": "NeurIPS",
        }])
        data = json.loads(result)
        item = data["results"][0]

        self.assertEqual(item["status"], "confirmed")
        self.assertEqual(item["issues"], [])
        self.assertEqual(item["lookup"], "doi")
        self.assertEqual(item["matched"]["openalex_id"], "https://openalex.org/W123")
        self.assertEqual(item["matched"]["authors"], ["Ashish Vaswani", "Noam Shazeer"])

        url = mock_urlopen.call_args[0][0].full_url
        self.assertIn("filter=doi%3a10.48550%2farxiv.1706.03762", url.lower())

    @patch("folium.tools.paper_validate.urllib.request.urlopen")
    def test_partial_when_metadata_conflicts(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"results": [OPENALEX_WORK]})

        result = self.tool.execute(papers=[{
            "title": "Attention Is All You Need",
            "year": 2020,
        }])
        item = json.loads(result)["results"][0]

        self.assertEqual(item["status"], "partial")
        self.assertIn("year mismatch", item["issues"])

    @patch("folium.tools.paper_validate.urllib.request.urlopen")
    def test_unverified_when_no_match(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"results": []})

        result = self.tool.execute(papers=[{"title": "Nonexistent Paper"}])
        item = json.loads(result)["results"][0]

        self.assertEqual(item["status"], "unverified")
        self.assertIn("no OpenAlex match found", item["issues"])

    def test_rejects_non_list_input(self):
        result = self.tool.execute(papers="not a list")
        self.assertIn("Error", result)


def _mock_response(data):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


if __name__ == "__main__":
    unittest.main()
