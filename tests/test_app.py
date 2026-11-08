import io
from app import create_app


def client(tmp_path, model_path=None):
    config = {"TESTING": True, "DATABASE_URL": f"sqlite:///{tmp_path / 'test.db'}"}
    if model_path is not None:
        config["MODEL_PATH"] = str(model_path)
    return create_app(config).test_client()


def test_index_and_health(tmp_path):
    c = client(tmp_path, tmp_path / "missing.keras")
    assert c.get("/").status_code == 200
    assert c.get("/health").status_code == 503


def test_rejects_bad_extension(tmp_path):
    response = client(tmp_path).post(
        "/predict",
        data={"image": (io.BytesIO(b"x"), "bad.exe")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Unsupported" in response.data
