import unittest
from unittest.mock import patch, MagicMock

from folium.tools.arxiv_search import ArxivSearchTool
from folium.tools.base import ToolValidationError


MOCK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Event-Triggered Control for Multi-Agent Systems</title>
    <summary>A novel approach to event-triggered control.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2301.00001v1" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00002v1</id>
    <title>Networked Control Under DoS Attacks</title>
    <summary>Resilient control under denial-of-service.</summary>
    <published>2023-02-15T00:00:00Z</published>
    <author><name>Charlie Brown</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2301.00002v1" />
  </entry>
</feed>
"""


class ArxivSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tool = ArxivSearchTool()

    def test_name_and_description(self):
        self.assertEqual(self.tool.name, "arxiv_search")
        self.assertIn("arXiv", self.tool.description)

    def test_required_fields(self):
        with self.assertRaises(ToolValidationError):
            self.tool.validate_arguments({})

    def test_valid_arguments(self):
        args = {"query": "event-triggered control"}
        result = self.tool.validate_arguments(args)
        self.assertEqual(result, args)

    def test_optional_fields(self):
        args = {"query": "test", "max_results": 10, "sort": "date"}
        result = self.tool.validate_arguments(args)
        self.assertEqual(result, args)

    def test_unknown_field_rejected(self):
        with self.assertRaises(ToolValidationError):
            self.tool.validate_arguments({"query": "test", "unknown": "value"})


class ArxivExecuteTests(unittest.TestCase):
    def setUp(self):
        self.tool = ArxivSearchTool()

    @patch("folium.tools.arxiv_search.urllib.request.urlopen")
    def test_returns_formatted_results(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = MOCK_XML.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.tool.execute(
            query="event-triggered control",
            max_results=2,
            year_from=2023,
        )

        self.assertIn("Event-Triggered Control", result)
        self.assertIn("Alice Smith", result)
        self.assertIn("2301.00001", result)
        self.assertIn("pdf", result)
        self.assertIn("Networked Control", result)

    @patch("folium.tools.arxiv_search.urllib.request.urlopen")
    def test_empty_results(self, mock_urlopen):
        empty_xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        mock_resp = MagicMock()
        mock_resp.read.return_value = empty_xml.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.tool.execute(query="nonexistent topic xyz")
        self.assertIn("No arXiv papers found", result)

    def test_empty_query_returns_error(self):
        result = self.tool.execute(query="   ")
        self.assertIn("Error", result)


if __name__ == "__main__":
    unittest.main()
