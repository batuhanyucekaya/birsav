"""Bounded streaming multipart parser, including early request-size validation."""

from dataclasses import dataclass
import tempfile

from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header


class UploadError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass
class UploadPart:
    file: object
    filename: str | None
    size: int = 0


class UploadForm(dict):
    def close(self):
        for part in self.values():
            part.file.close()


def parse_form(stream, headers, max_bytes, allowed_fields=("image", "video", "gps")):
    if headers.get("Transfer-Encoding"):
        raise UploadError("Transfer-Encoding desteklenmiyor", 411)
    try:
        length = int(headers.get("Content-Length", ""))
    except ValueError:
        raise UploadError("Geçerli Content-Length gerekli", 411)
    if length < 1:
        raise UploadError("İstek gövdesi boş")
    if length > max_bytes:
        raise UploadError("Yükleme boyut sınırını aşıyor", 413)
    content_type, options = parse_options_header(headers.get("Content-Type", ""))
    boundary = options.get(b"boundary", b"")
    if content_type != b"multipart/form-data" or not 1 <= len(boundary) <= 200:
        raise UploadError("Geçerli multipart/form-data boundary gerekli")
    form = UploadForm()
    state = {"part": None, "count": 0, "ended": False, "header_bytes": 0}

    def begin():
        state["count"] += 1
        if state["count"] > len(allowed_fields):
            raise UploadError("Çok fazla form alanı")
        state.update(headers={}, field=bytearray(), value=bytearray(), header_bytes=0)

    def header_field(data, start, end):
        state["header_bytes"] += end - start
        if state["header_bytes"] > 8192:
            raise UploadError("Form başlıkları çok büyük")
        state["field"].extend(data[start:end])

    def header_value(data, start, end):
        state["header_bytes"] += end - start
        if state["header_bytes"] > 8192:
            raise UploadError("Form başlıkları çok büyük")
        state["value"].extend(data[start:end])

    def header_end():
        state["headers"][bytes(state["field"]).lower()] = bytes(state["value"])
        state["field"].clear()
        state["value"].clear()

    def headers_finished():
        disposition, params = parse_options_header(
            state["headers"].get(b"content-disposition", b"")
        )
        name = params.get(b"name", b"").decode("utf-8", errors="strict")
        if disposition != b"form-data" or name not in allowed_fields or name in form:
            raise UploadError("Bilinmeyen veya tekrarlanan form alanı")
        part = UploadPart(
            tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b"),
            params.get(b"filename", b"").decode("utf-8", errors="replace"),
        )
        form[name] = part
        state["part"], state["name"] = part, name

    def data_chunk(data, start, end):
        part = state["part"]
        if part is None:
            raise UploadError("Form alanı başlığı eksik")
        part.size += end - start
        limit = 2 * 1024 * 1024 if state["name"] == "gps" else max_bytes
        if part.size > limit:
            raise UploadError("Form alanı boyut sınırını aşıyor", 413)
        part.file.write(data[start:end])

    def part_end():
        state["part"].file.seek(0)
        state["part"] = None

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": begin,
            "on_header_field": header_field,
            "on_header_value": header_value,
            "on_header_end": header_end,
            "on_headers_finished": headers_finished,
            "on_part_data": data_chunk,
            "on_part_end": part_end,
            "on_end": lambda: state.update(ended=True),
        },
    )
    try:
        remaining = length
        while remaining:
            chunk = stream.read(min(64 * 1024, remaining))
            if not chunk:
                raise UploadError("Eksik istek gövdesi")
            parser.write(chunk)
            remaining -= len(chunk)
        parser.finalize()
        if not state["ended"]:
            raise UploadError("Tamamlanmamış multipart gövdesi")
        return form
    except Exception:
        form.close()
        raise
