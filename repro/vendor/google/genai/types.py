"""Minimal stand-ins for the google-genai `types` used by the OmniVideo-100K
data-engine scripts. They are plain data holders; the shim Client (see
`__init__.py`) interprets them and forwards the request to OpenRouter."""


class Blob:
    def __init__(self, data=None, mime_type=None):
        self.data = data
        self.mime_type = mime_type


class Part:
    def __init__(self, text=None, inline_data=None):
        self.text = text
        self.inline_data = inline_data

    @classmethod
    def from_bytes(cls, data=None, mime_type=None):
        return cls(inline_data=Blob(data=data, mime_type=mime_type))

    @classmethod
    def from_text(cls, text=None):
        return cls(text=text)


class Content:
    def __init__(self, parts=None, role=None):
        self.parts = parts or []
        self.role = role
