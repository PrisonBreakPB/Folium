import unittest

from folium.encoding import repair_mojibake_payload, repair_mojibake_text


class EncodingRepairTests(unittest.TestCase):
    def test_repairs_utf8_decoded_as_gbk(self):
        self.assertEqual(repair_mojibake_text("涓枃杈撳嚭"), "中文输出")

    def test_repairs_lossy_historical_tool_output(self):
        self.assertEqual(
            repair_mojibake_text("1.鏈�绯荤粺闈㈠悜鍐呴儴"),
            "1.本系统面向内部",
        )
        self.assertEqual(
            repair_mojibake_text("鏄�鍚︽敮鎸侀渶姹傜殑绫诲瀷銆�"),
            "是否支持需求的类型。",
        )

    def test_keeps_normal_chinese_text(self):
        self.assertEqual(repair_mojibake_text("工具结果展示正常"), "工具结果展示正常")

    def test_repairs_nested_payload_text(self):
        payload = {"content": "涓枃杈撳嚭", "items": ["工具结果展示正常"]}

        repaired = repair_mojibake_payload(payload)

        self.assertEqual(repaired["content"], "中文输出")
        self.assertEqual(repaired["items"], ["工具结果展示正常"])


if __name__ == "__main__":
    unittest.main()
