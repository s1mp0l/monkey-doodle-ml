"""Редактор: сохранение/чтение через HTTP API, галерея, ревью, предсказание, параллельные клиенты."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from doodle.ndjson_io import read_records
from editor.server import App, serve


def _tiny_model(dirpath):
    """Необученная DoodleNet на 3 класса → ONNX + labels.json (как у export_onnx.py)."""
    import torch

    from doodle.model import DoodleNet, ExportWrapper

    net = DoodleNet(3, (8, 8, 8, 8)).eval()
    torch.onnx.export(ExportWrapper(net), (torch.zeros(1, 1, 64, 64),), str(dirpath / "doodle.onnx"), opset_version=15,
                      input_names=["input"], output_names=["probs"], dynamo=False)
    labels = {"run": "test", "preprocess": {"size": 64, "pad": 2.0, "line_width": 2.0, "epsilon": 2.0},
              "classes": [{"index": 0, "id": "banana", "ru": "Банан", "threshold": 0.0},
                          {"index": 1, "id": "other", "ru": "Другое", "threshold": None},
                          {"index": 2, "id": "rude", "ru": "Неприличное", "threshold": 0.5}]}
    (dirpath / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")


@pytest.fixture()
def server(tmp_path):
    custom = tmp_path / "custom"
    models = tmp_path / "model"
    models.mkdir()
    _tiny_model(models)
    app = App(custom_dir=custom, model_dir=models)
    httpd = serve(app, "127.0.0.1", 0)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", custom
    httpd.shutdown()
    httpd.server_close()


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
        return json.loads(raw) if r.headers.get_content_type() == "application/json" else raw


DRAWING = [[[10, 60, 110, 160], [100, 40, 40, 100], [0, 16, 33, 50]], [[85, 85], [40, 190], [300, 340]]]


def test_save_list_thumb_review_predict(server):
    base, custom = server
    cfg = call(base, "/api/config")
    assert cfg["default_class"] == "rude" and cfg["model"]["available"]
    r = call(base, "/api/samples", {"word": "rude", "author": "тест", "prompt": "маленький", "device": "pen",
                                    "canvas": {"w": 800, "h": 800}, "drawing": DRAWING})
    assert r["ok"] and r["count"]["active"] == 1
    recs = read_records(custom / "rude.ndjson")
    assert len(recs) == 1
    rec = recs[0]
    assert rec["key_id"] == r["key_id"] and rec["word"] == "rude" and rec["author"] == "тест"
    assert rec["device"] == "pen" and rec["prompt"] == "маленький" and rec["timestamp"]
    assert rec["drawing"] == DRAWING and rec["simplified"][0][0][0] == 0

    lst = call(base, "/api/samples?word=rude")
    assert lst["total"] == 1 and lst["items"][0]["key_id"] == r["key_id"] and lst["authors"] == ["тест"]
    png = call(base, f"/api/thumb?word=rude&key={r['key_id']}")
    assert png[:4] == b"\x89PNG"

    call(base, "/api/review", {"word": "rude", "key_id": r["key_id"], "action": "delete"})
    assert call(base, "/api/samples?word=rude")["total"] == 0
    assert call(base, "/api/samples?word=rude&status=deleted")["total"] == 1
    call(base, "/api/review", {"word": "rude", "key_id": r["key_id"], "action": "ok"})
    assert call(base, "/api/config")["counts"]["rude"]["active"] == 1

    p = call(base, "/api/predict", {"drawing": [s[:2] for s in DRAWING]})
    assert p["available"] and len(p["top"]) == 3 and abs(sum(t["p"] for t in p["top"]) - 1) < 1e-4
    assert p["preview"].startswith("data:image/png;base64,")


def test_rejects_bad_input(server):
    base, _ = server
    for body in ({"word": "../../etc", "drawing": DRAWING}, {"word": "rude", "drawing": []},
                 {"word": "rude", "drawing": [[[1, 2], [3]]]}, {"word": "rude", "drawing": [[["a"], [1], [0]]]}):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(base, "/api/samples", body)
        assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, "/static/../server.py")
    assert e.value.code == 404


def test_parallel_clients(server):
    base, custom = server
    errs = []

    def client(i):
        try:
            for k in range(10):
                call(base, "/api/samples", {"word": "banana", "author": f"c{i}", "drawing": DRAWING, "device": "touch"})
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    th = [threading.Thread(target=client, args=(i,)) for i in range(6)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert not errs
    recs = read_records(custom / "banana.ndjson")
    assert len(recs) == 60 and len({r["key_id"] for r in recs}) == 60
