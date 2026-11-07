import ast
from pathlib import Path
import unittest


APPLICATION_PATH = Path(__file__).resolve().parents[1] / "main.py"


class ApplicationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APPLICATION_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(APPLICATION_PATH))

    def test_application_title_is_present(self):
        self.assertIn("AI Vision – Object Recognition", self.source)
        self.assertIn('root.title("AI Vision – Object Recognition")', self.source)

    def test_analysis_and_exit_actions_are_available(self):
        button_labels = {
            keyword.value.value
            for call in ast.walk(self.tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "Button"
            for keyword in call.keywords
            if keyword.arg == "text"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        }
        self.assertIn("Start Analysis", button_labels)
        self.assertIn("Back to Live Preview", button_labels)
        self.assertIn("Exit", button_labels)

    def test_status_messages_cover_camera_and_analysis_states(self):
        for message in (
            "Ready",
            "Analyzing...",
            "Analysis completed",
            "Camera unavailable",
        ):
            with self.subTest(message=message):
                self.assertIn(message, self.source)

    def test_analysis_button_is_disabled_until_worker_finishes(self):
        self.assertIn("if analysis_running:", self.source)
        self.assertIn('analyze_button.configure(state="disabled"', self.source)
        self.assertIn(
            'text="Start Analysis"',
            self.source,
        )
        self.assertIn('state="normal" if camera_available else "disabled"', self.source)

    def test_live_preview_action_clears_overlay_and_allows_another_analysis(self):
        self.assertIn("def return_to_live_preview():", self.source)
        self.assertIn("analysis_overlay = None", self.source)
        self.assertIn("Live camera preview. Select Start Analysis", self.source)
        self.assertIn('command=return_to_live_preview', self.source)

    def test_layout_expands_with_the_window(self):
        self.assertIn("root.rowconfigure(1, weight=1)", self.source)
        self.assertIn(
            'content.columnconfigure(0, weight=3, uniform="panels")',
            self.source,
        )
        self.assertIn("canvas.grid(row=2, column=0, sticky=\"nsew\")", self.source)
        self.assertIn("canvas.create_text(", self.source)

    def test_mobilenet_classification_pipeline_is_preserved(self):
        self.assertIn("models.mobilenet_v3_small(weights=weights)", self.source)
        self.assertIn("probabilities = model(input_tensor).softmax(dim=1)[0]", self.source)
        self.assertIn("weights.meta[\"categories\"][index]", self.source)


if __name__ == "__main__":
    unittest.main()
