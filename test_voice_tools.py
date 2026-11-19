"""Tests for VaaniAI local tools (time, calculator, notes)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import voice_tools


class CalculatorTest(unittest.TestCase):
    def test_basic_math(self):
        ok, out = voice_tools.execute("calculate", {"expression": "(12+8)*3/2"})
        self.assertTrue(ok)
        self.assertEqual(out, "(12+8)*3/2 = 30")

    def test_decimals_and_power(self):
        ok, out = voice_tools.execute("calculate", {"expression": "2**10"})
        self.assertTrue(ok)
        self.assertIn("1024", out)

    def test_rejects_non_arithmetic(self):
        for expr in ["__import__('os')", "os.getcwd", "1 if 2 else 3"]:
            ok, out = voice_tools.execute("calculate", {"expression": expr})
            self.assertFalse(ok, expr)
            self.assertIn("Error", out)

    def test_division_by_zero(self):
        ok, out = voice_tools.execute("calculate", {"expression": "5/0"})
        self.assertFalse(ok)
        self.assertIn("zero", out)


class TimeTest(unittest.TestCase):
    def test_time_format(self):
        ok, out = voice_tools.execute("get_current_time", {})
        self.assertTrue(ok)
        self.assertIn("It's", out)


class NotesTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        self.patcher = mock.patch.object(voice_tools, "NOTES_FILE",
                                         tmp / "notes.json")
        self.patcher.start()
        self.file = tmp / "notes.json"

    def tearDown(self):
        self.patcher.stop()

    def test_add_list_delete(self):
        ok, out = voice_tools.execute("add_note", {"text": "demo on Friday"})
        self.assertTrue(ok)
        self.assertIn("Noted", out)
        ok, out = voice_tools.execute("list_notes", {})
        self.assertTrue(ok)
        self.assertIn("demo on Friday", out)
        ok, out = voice_tools.execute("delete_note", {"note_id": 1})
        self.assertTrue(ok)
        ok, out = voice_tools.execute("list_notes", {})
        self.assertEqual(out, "No notes yet.")

    def test_delete_missing(self):
        ok, out = voice_tools.execute("delete_note", {"note_id": 99})
        self.assertFalse(ok)

    def test_empty_note_rejected(self):
        ok, _ = voice_tools.execute("add_note", {"text": "   "})
        self.assertFalse(ok)

    def test_notes_persist(self):
        voice_tools.execute("add_note", {"text": "persist me"})
        # re-read from disk through a fresh call
        ok, out = voice_tools.execute("list_notes", {})
        self.assertIn("persist me", out)
        self.assertTrue(self.file.exists())


class RegistryTest(unittest.TestCase):
    def test_schemas_and_unknown_tool(self):
        schemas = voice_tools.schemas()
        names = {s["function"]["name"] for s in schemas}
        self.assertEqual(names, {"get_current_time", "calculate", "add_note",
                                 "list_notes", "delete_note"})
        ok, out = voice_tools.execute("nope", {})
        self.assertFalse(ok)
        self.assertIn("unknown tool", out)


if __name__ == "__main__":
    unittest.main()
